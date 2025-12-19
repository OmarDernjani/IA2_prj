import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helper_fun import evaluate_models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PATH = 'data/processed/'
SAVING_PATH = 'results/model_res/'


X_train = pd.read_csv(PATH + 'X_train.csv')
X_test = pd.read_csv(PATH + 'X_test.csv')

y_train = pd.read_csv(PATH + 'y_train.csv').squeeze()
y_test = pd.read_csv(PATH + 'y_test.csv').squeeze()



models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        solver='liblinear'
    ),

    "Bernoulli NB": BernoulliNB(),

    "KNN": Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    ),

    "LDA": LinearDiscriminantAnalysis(),

    "QDA": QuadraticDiscriminantAnalysis(
        reg_param=0.1
    ),

    "Random Forest": RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
}


for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)


results = evaluate_models(models, X_test, y_test)

print("\nModel Results:")
print(results)


logreg = models["Logistic Regression"]

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': logreg.coef_[0],
})

feature_importance['Odds_Ratio'] = np.exp(feature_importance['Coefficient'])
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()

feature_importance = feature_importance.sort_values(
    by='Abs_Coefficient',
    ascending=False
)

print("\nTop 20 Features (LogReg):")
print(feature_importance.head(20))

feature_importance.to_csv(
    SAVING_PATH + 'feature_importance_logreg.csv',
    index=False
)

results.to_csv(
    SAVING_PATH + 'model_results.csv',
    index=False
)

metric_cols = [
    col for col in results.columns
    if 'accur' in col.lower()
    or 'auc' in col.lower()
    or 'f1' in col.lower()
]

if metric_cols:
    metric = metric_cols[0]
    best = results.loc[results[metric].idxmax()]

    print(f"\nBest model by {metric}:")
    print(best)

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score

# Caricamento dati (solo features principali)
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# Scorers
scorers = {
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    'accuracy': make_scorer(accuracy_score)
}

# 1️⃣ Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_leaf': [1, 5, 10]
}

rf_grid = GridSearchCV(
    rf, rf_params,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train, y_train)
print("Best RF params:", rf_grid.best_params_)
print("Best RF ROC AUC:", rf_grid.best_score_)

# 2️⃣ Logistic Regression
lr = LogisticRegression(max_iter=2000, solver='liblinear')

lr_params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

lr_grid = GridSearchCV(
    lr, lr_params,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)
lr_grid.fit(X_train, y_train)
print("Best LR params:", lr_grid.best_params_)
print("Best LR ROC AUC:", lr_grid.best_score_)

# 3️⃣ LDA
lda = LinearDiscriminantAnalysis()

lda_params = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': [None, 0.1, 0.5, 1]  # funziona solo con lsqr/eigen
}

lda_grid = GridSearchCV(
    lda, lda_params,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=1
)
lda_grid.fit(X_train, y_train)
print("Best LDA params:", lda_grid.best_params_)
print("Best LDA ROC AUC:", lda_grid.best_score_)