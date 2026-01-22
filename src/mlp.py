import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


LOG_ROOT_DIR = './logs'                
RESULTS_DIR = './results/model_res'   


class BinaryTabularDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = pd.read_csv(X_path).values.astype(np.float32)
        self.y = pd.read_csv(y_path).values.astype(np.float32).flatten()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BinaryMLP(pl.LightningModule):
    def __init__(self, input_dim, results_dir, hidden_dims=(128, 64), dropout=0.2, lr=1e-3):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.results_dir = results_dir
        
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
        
        # Metriche
        self.train_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_acc = torchmetrics.classification.BinaryAccuracy()
        self.val_auc = torchmetrics.classification.BinaryAUROC()
        
        self.val_step_outputs = []

    def forward(self, x):
        return self.net(x).squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        
        self.val_acc(probs, y)
        self.val_auc(probs, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_step_outputs.append({
            'y_true': y.cpu(),
            'y_prob': probs.cpu()
        })
        return loss

    def on_validation_epoch_end(self):
        """Salva le predizioni nella cartella RESULTS_DIR separata dai log."""
        all_true = torch.cat([x['y_true'] for x in self.val_step_outputs])
        all_probs = torch.cat([x['y_prob'] for x in self.val_step_outputs])
        all_preds = (all_probs >= 0.5).float()
        
        df_res = pd.DataFrame({
            'y_true': all_true.numpy(),
            'y_prob': all_probs.numpy(),
            'y_pred': all_preds.numpy()
        })
        
        
        version = self.logger.version if self.logger else 'unknown'
        filename = f'preds_v{version}_epoch_{self.current_epoch}.csv'
        
        save_path = os.path.join(self.results_dir, filename)
        df_res.to_csv(save_path, index=False)
        
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def create_dataloaders(data_dir='data/processed', batch_size=64):
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        for name in ['X_train.csv', 'X_test.csv']:
            pd.DataFrame(np.random.randn(100, 10)).to_csv(f'{data_dir}/{name}', index=False)
        for name in ['y_train.csv', 'y_test.csv']:
            pd.DataFrame(np.random.randint(0, 2, 100)).to_csv(f'{data_dir}/{name}', index=False)

    train_ds = BinaryTabularDataset(f'{data_dir}/X_train.csv', f'{data_dir}/y_train.csv')
    val_ds = BinaryTabularDataset(f'{data_dir}/X_test.csv', f'{data_dir}/y_test.csv')
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    )

def train_model(data_dir='data/processed', max_epochs=50, batch_size=64):
    
    train_loader, val_loader = create_dataloaders(data_dir, batch_size)
    input_dim = next(iter(train_loader))[0].shape[1]
    
    
    model = BinaryMLP(input_dim=input_dim, results_dir=RESULTS_DIR)
    
    
    logger = CSVLogger(save_dir=LOG_ROOT_DIR, name='.')
    
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=None, 
        monitor='val_loss',
        filename='best-model-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[EarlyStopping('val_loss', patience=10), checkpoint_callback],
        logger=logger,
        accelerator='auto',
        devices=1,
        log_every_n_steps=1,
        enable_progress_bar=True
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    return trainer.logger.log_dir


if __name__ == '__main__':
    
    
    current_log_dir = train_model(max_epochs=5, batch_size=32)
    metrics_file = os.path.join(current_log_dir, 'metrics.csv')
    if os.path.exists(metrics_file):
        print(f"\nLettura {metrics_file}...")
        m = pd.read_csv(metrics_file)
        print(m.groupby('epoch').max()[['train_loss', 'val_loss', 'val_auc']].tail(1))
    
    
    res_files = os.listdir(RESULTS_DIR)
    if res_files:
        last_res = sorted(res_files)[-1]
        print(f"\nLettura ultimo file risultati in {RESULTS_DIR}/{last_res}...")
        print(pd.read_csv(os.path.join(RESULTS_DIR, last_res)).head())