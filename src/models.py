import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from helper_fun import evaluate_models
import numpy as np

PATH = 'data/processed/'
SAVING_PATH = 'results/model_res'

#Load MLB datasets
X_train_mlb = pd.read_csv(PATH + 'X_train_MLB.csv')
X_test_mlb = pd.read_csv(PATH + 'X_test_MLB.csv')
y_train = pd.read_csv(PATH + 'y_train.csv')['Type of Answer']
y_test = pd.read_csv(PATH + 'y_test.csv')['Type of Answer']

#Load OHE datasets
X_train_ohe = pd.read_csv(PATH + 'X_train_OHE.csv')
X_test_ohe = pd.read_csv(PATH + 'X_test_OHE.csv')
y_train_ohe = pd.read_csv(PATH + 'y_train_OHE.csv')['Type of Answer']
y_test_ohe = pd.read_csv(PATH + 'y_test_OHE.csv')['Type of Answer']

# Instantiate models for MLB
models_mlb = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis()
}

# Train models on MLB data
for name, model in models_mlb.items():
    print(f"Training {name}...")
    model.fit(X_train_mlb, y_train)

# Evaluate MLB models
results_mlb = evaluate_models(models_mlb, X_test_mlb, y_test)
print("\nMLB Encoding Results:")
print(results_mlb)

# Feature importance for MLB
feature_importance_mlb = pd.DataFrame({
    'Feature': X_train_mlb.columns,
    'Coefficient': models_mlb["Logistic Regression"].coef_[0],
    'Odds_Ratio': np.exp(models_mlb["Logistic Regression"].coef_[0])
})

feature_importance_mlb['Abs_Coefficient'] = feature_importance_mlb['Coefficient'].abs()
print(feature_importance_mlb.sort_values(by='Abs_Coefficient', ascending=False).head(20))

# Instantiate models for OHE
models_ohe = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis()
}

# Train models on OHE data
for name, model in models_ohe.items():
    print(f"Training {name}...")
    model.fit(X_train_ohe, y_train_ohe)

# Evaluate OHE models
results_ohe = evaluate_models(models_ohe, X_test_ohe, y_test_ohe)
print("\nOHE Encoding Results:")
print(results_ohe)

# Feature importance for OHE
feature_importance_ohe = pd.DataFrame({
    'Feature': X_train_ohe.columns,
    'Coefficient': models_ohe["Logistic Regression"].coef_[0],
    'Odds_Ratio': np.exp(models_ohe["Logistic Regression"].coef_[0])
})


feature_importance_ohe['Abs_Coefficient'] = feature_importance_ohe['Coefficient'].abs()
print(feature_importance_ohe.sort_values(by='Abs_Coefficient', ascending=False).head(20))

# Save feature importance results
feature_importance_mlb.sort_values(by='Abs_Coefficient', ascending=False).to_csv(SAVING_PATH + 'feature_importance_MLB.csv', index=False)
feature_importance_ohe.sort_values(by='Abs_Coefficient', ascending=False).to_csv(SAVING_PATH + 'feature_importance_OHE.csv', index=False)

# Save model results
results_mlb.to_csv(SAVING_PATH + 'model_results_MLB.csv', index=False)
results_ohe.to_csv(SAVING_PATH + 'model_results_OHE.csv', index=False)

# Find the metric column (could be 'Accuracy', 'accuracy', or another metric)
metric_cols = [col for col in results_mlb.columns if 'accur' in col.lower() or 'score' in col.lower()]
if metric_cols:
    metric_col = metric_cols[0]
    print(f"\nUsing metric: {metric_col}")
    print("\nBest MLB Model:")
    print(results_mlb.loc[results_mlb[metric_col].idxmax()])
    print("\nBest OHE Model:")
    print(results_ohe.loc[results_ohe[metric_col].idxmax()])
else:
    print("\nMLB Results:")
    print(results_mlb)
    print("\nOHE Results:")
    print(results_ohe)