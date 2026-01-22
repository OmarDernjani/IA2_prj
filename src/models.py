import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from sklearn.dummy import DummyClassifier
from helper_fun import evaluate_models

#Global var

DATA_PATH = 'data/processed/'
RESULTS_PATH = 'results/model_res/'
RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS = 5
SCORING_METRIC = 'roc_auc'

#Load Dataset

def load_processed_data(path):
    """Load preprocessed training and test data."""
    X_train = pd.read_csv(path + 'X_train.csv')
    X_test = pd.read_csv(path + 'X_test.csv')
    y_train = pd.read_csv(path + 'y_train.csv').squeeze()
    y_test = pd.read_csv(path + 'y_test.csv').squeeze()
    
    print("="*70)
    print("DATA LOADED")
    print("="*70)
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def compute_baseline(X_train, X_test, y_train, y_test):
    """Compute baseline accuracy using dummy classifier."""
    print("\n" + "="*70)
    print("BASELINE ANALYSIS")
    print("="*70)
    
    # Class distribution
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    
    print("\nClass Distribution:")
    print(f"  Train - Class 0: {train_dist[0]:.4f} | Class 1: {train_dist[1]:.4f}")
    print(f"  Test  - Class 0: {test_dist[0]:.4f} | Class 1: {test_dist[1]:.4f}")
    
    # Dummy classifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    baseline_acc = dummy.score(X_test, y_test)
    
    print(f"\nDummy Classifier (most frequent):")
    print(f"  Baseline Accuracy: {baseline_acc:.4f}")
    print(f"  Target: Beat {baseline_acc:.4f} significantly")
    
    return baseline_acc


def get_baseline_models(random_state=42, n_jobs=-1):
    """Define baseline models with default hyperparameters."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            solver='liblinear',
            random_state=random_state
        ),
        
        "Bernoulli NB": BernoulliNB(),
        
        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]),
        
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=random_state
        ),
        
        "LDA": LinearDiscriminantAnalysis(),
        
        "QDA": QuadraticDiscriminantAnalysis(
            reg_param=0.1
        ),
        
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=50,
            random_state=random_state,
            n_jobs=n_jobs
        ),
        
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
    }
    
    return models


def train_baseline_models(models, X_train, y_train, X_test, y_test, results_path):
    """Train baseline models and evaluate performance."""
    print("\n" + "="*70)
    print("TRAINING BASELINE MODELS")
    print("="*70)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    
    # Evaluate all models
    # evaluate_models returns DataFrame with model names as index
    results = evaluate_models(models, X_test, y_test)
    
    print("\n" + "="*70)
    print("BASELINE RESULTS SUMMARY")
    print("="*70)
    print(results)
    
    # Save results
    results.to_csv(results_path + 'baseline_results.csv', index=True)
    
    # Find best model by accuracy
    if 'accuracy' in results.columns:
        best_idx = results['accuracy'].idxmax()
        best_acc = results.loc[best_idx, 'accuracy']
        
        print(f"\nBest baseline model by accuracy:")
        print(f"  Model: {best_idx}")
        print(f"  Accuracy: {best_acc:.4f}")
        
        # Show ROC AUC if available
        if 'roc_auc' in results.columns and pd.notna(results.loc[best_idx, 'roc_auc']):
            print(f"  ROC AUC: {results.loc[best_idx, 'roc_auc']:.4f}")
    
    return results


def analyze_feature_importance(model, feature_names, results_path, top_n=15):
    """Analyze and save feature importance from logistic regression."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0])
    })
    
    feature_importance = feature_importance.sort_values(
        by='Abs_Coefficient',
        ascending=False
    )
    
    print(f"\nTop {top_n} Most Important Features:")
    print(feature_importance.head(top_n).to_string(index=False))
    
    # Save full feature importance
    feature_importance.to_csv(
        results_path + 'feature_importance.csv',
        index=False
    )
    
    return feature_importance


def tune_random_forest(X_train, y_train, random_state=42, n_jobs=-1, 
                       scoring='roc_auc', cv=5):
    """Tune Random Forest hyperparameters."""
    
    rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_leaf': [1, 5, 10, 20]
    }
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV {scoring}: {grid_search.best_score_:.4f}")
    
    return grid_search


def tune_logistic_regression(X_train, y_train, random_state=42, n_jobs=-1,
                             scoring='roc_auc', cv=5):
    """Tune Logistic Regression hyperparameters."""
    
    lr = LogisticRegression(max_iter=2000, solver='liblinear', random_state=random_state)
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }
    
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV {scoring}: {grid_search.best_score_:.4f}")
    
    return grid_search


def tune_lda(X_train, y_train, n_jobs=-1, scoring='roc_auc', cv=5):
    """Tune Linear Discriminant Analysis hyperparameters."""
    
    lda = LinearDiscriminantAnalysis()
    
    param_grid = {
        'solver': ['lsqr', 'eigen'],
        'shrinkage': [None, 0.1, 0.3, 0.5, 0.7, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=lda,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV {scoring}: {grid_search.best_score_:.4f}")
    
    return grid_search


def tune_gradient_boosting(X_train, y_train, random_state=42, n_jobs=-1,
                           scoring='roc_auc', cv=5):
    """Tune Gradient Boosting hyperparameters."""
    
    gb = GradientBoostingClassifier(random_state=random_state)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV {scoring}: {grid_search.best_score_:.4f}")
    
    return grid_search



def evaluate_tuned_models(tuned_models, X_test, y_test, results_path):
    """Evaluate tuned models on test set."""
    results = evaluate_models(tuned_models, X_test, y_test)
    # Save results
    results.to_csv(results_path + 'tuned_results.csv', index=True)
    
    return results



def compare_results(baseline_results, tuned_results, results_path, baseline_acc):
    """Compare baseline and tuned models."""
    
    # Combine results - both have model names as index
    all_results = pd.concat([baseline_results, tuned_results])
    all_results.to_csv(results_path + 'all_results.csv', index=True)
    
    # Find best model by accuracy
    if 'accuracy' in all_results.columns:
        best_idx = all_results['accuracy'].idxmax()
        best_acc = all_results.loc[best_idx, 'accuracy']
        
        print(f"\nBest Overall Model:")
        print(f"  Model: {best_idx}")
        print(f"  Accuracy: {best_acc:.4f}")
        print(f"  Improvement over baseline: +{(best_acc - baseline_acc):.4f}")
        print(f"  Relative improvement: {((best_acc / baseline_acc - 1) * 100):.1f}%")
        
        # Show all metrics for best model
        print(f"\n  All metrics for best model:")
        for col in all_results.columns:
            if pd.notna(all_results.loc[best_idx, col]):
                print(f"    {col}: {all_results.loc[best_idx, col]:.4f}")
    
    return all_results

def main():
    """Main execution pipeline for model training and evaluation."""
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(DATA_PATH)
    
    # Compute baseline
    baseline_acc = compute_baseline(X_train, X_test, y_train, y_test)
    
    # Get and train baseline models
    baseline_models = get_baseline_models(RANDOM_STATE, N_JOBS)
    baseline_results = train_baseline_models(
        baseline_models, X_train, y_train, X_test, y_test, RESULTS_PATH
    )
    
    # Feature importance analysis
    logreg = baseline_models["Logistic Regression"]
    feature_importance = analyze_feature_importance(
        logreg, X_train.columns, RESULTS_PATH
    )
    
    # Hyperparameter tuning
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    
    rf_tuned = tune_random_forest(
        X_train, y_train, RANDOM_STATE, N_JOBS, SCORING_METRIC, CV_FOLDS
    )
    
    lr_tuned = tune_logistic_regression(
        X_train, y_train, RANDOM_STATE, N_JOBS, SCORING_METRIC, CV_FOLDS
    )
    
    lda_tuned = tune_lda(
        X_train, y_train, N_JOBS, SCORING_METRIC, CV_FOLDS
    )
    
    gb_tuned = tune_gradient_boosting(
        X_train, y_train, RANDOM_STATE, N_JOBS, SCORING_METRIC, CV_FOLDS
    )
    
    # Evaluate tuned models
    tuned_models = {
        "Random Forest (tuned)": rf_tuned.best_estimator_,
        "Logistic Regression (tuned)": lr_tuned.best_estimator_,
        "LDA (tuned)": lda_tuned.best_estimator_,
        "Gradient Boosting (tuned)": gb_tuned.best_estimator_
    }
    
    tuned_results = evaluate_tuned_models(tuned_models, X_test, y_test, RESULTS_PATH)
    
    # Final comparison
    all_results = compare_results(
        baseline_results, tuned_results, RESULTS_PATH, baseline_acc
    )
    

if __name__ == "__main__":
    main()