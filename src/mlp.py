import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import accuracy_score, roc_auc_score


class BinaryTabularDataset(Dataset):
    """Class dataset"""
    def __init__(self, X_path: str, y_path: str):
        self.X = pd.read_csv(X_path).values.astype(np.float32)
        self.y = pd.read_csv(y_path).values.astype(np.float32).flatten()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BinaryMLP(pl.LightningModule):
    """Class Model"""
    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout=0.2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev = h
        
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = lr
    
    def forward(self, x):
        return self.net(x).squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        
        # Log
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy_score(y.cpu(), preds.cpu()), prog_bar=True)
        self.log('val_auc', roc_auc_score(y.cpu(), probs.cpu()), prog_bar=True)
        
        return {'val_loss': loss, 'probs': probs, 'y': y}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def create_dataloaders(data_dir='data/processed', batch_size=64):
    """Crea DataLoader per training e validation"""
    X_train = os.path.join(data_dir, 'X_train.csv')
    X_val = os.path.join(data_dir, 'X_test.csv')
    y_train = os.path.join(data_dir, 'y_train.csv')
    y_val = os.path.join(data_dir, 'y_test.csv')
    
    train_ds = BinaryTabularDataset(X_train, y_train)
    val_ds = BinaryTabularDataset(X_val, y_val)
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    )


def train_model(data_dir='data/processed', max_epochs=50, batch_size=64):
    """Training loop"""

    
    train_loader, val_loader = create_dataloaders(data_dir, batch_size)
    input_dim = next(iter(train_loader))[0].shape[1]
    
    #istanciate the model
    model = BinaryMLP(input_dim=input_dim)
    
    # Callbacks (also implicti regularization)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, mode='min'),
        ModelCheckpoint(
            monitor='val_loss',
            filename='best-model-{epoch:02d}-{val_loss:.3f}',
            save_top_k=1,
            mode='min'
        ),
        ModelCheckpoint(
            monitor='val_auc',
            filename='best-auc-{epoch:02d}-{val_auc:.3f}',
            save_top_k=1,
            mode='max'
        )
    ]
    
    # Logging and save
    logger = CSVLogger('logs', name='binary_classifier')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Training loop
    trainer.fit(model, train_loader, val_loader)
    
    # Best model with callbacks API
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        model = BinaryMLP.load_from_checkpoint(best_model_path)
    
    return model, trainer


def predict(model, X_csv_path, batch_size=64):
    """Make prediction"""
    model.eval()
    
    X = pd.read_csv(X_csv_path).values.astype(np.float32)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = next(model.parameters()).device
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs)
    
    return np.array(predictions)


if __name__ == '__main__':

    model, trainer = train_model(
        data_dir='data/processed',
        max_epochs=30,
        batch_size=64
    )
    
    train_loader, val_loader = create_dataloaders()
    val_results = trainer.validate(model, val_loader)
    
    # Save Logging
    metrics = pd.read_csv(f'logs/binary_classifier/version_{trainer.logger.version}/metrics.csv')
    print(metrics.tail())
