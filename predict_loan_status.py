
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import catboost as cb
import optuna
import pickle
import os
import matplotlib.pyplot as plt

train_path = "E:\\CSI_CB\\processed\\train_processed.csv"
test_path = "E:\\CSI_CB\\processed\\test_processed.csv"
submission_path = "E:\\CSI_CB\\submission.csv"
model_path = "E:\\CSI_CB\\best_model.pkl"
plot_path = "E:\\CSI_CB\\feature_importance.png"

try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Train and test data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Check class imbalance
print("\nLoan_Status Distribution (Before SMOTE):")
print(train_df['Loan_Status'].value_counts(normalize=True))

# Features
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 
            'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
            'Property_Rural', 'Property_Semiurban', 'Property_Urban', 'TotalIncome', 
            'LoanAmount_to_Income']
X = train_df[features].copy()
y = train_df['Loan_Status'].copy()
X_test = test_df[features].copy()

# Correlation matrix for income features
print("\nCorrelation Matrix for Income Features:")
corr_matrix = X[['ApplicantIncome', 'CoapplicantIncome', 'TotalIncome']].corr()
print(corr_matrix)

# Drop TotalIncome if highly correlated
if corr_matrix.loc['TotalIncome', ['ApplicantIncome', 'CoapplicantIncome']].abs().max() > 0.8:
    print("\nDropping TotalIncome due to high correlation.")
    features.remove('TotalIncome')
    X = X.drop('TotalIncome', axis=1)
    X_test = X_test.drop('TotalIncome', axis=1)

fill_values = {
    'Credit_History': 0, 'Gender': 0, 'Married': 0, 'Self_Employed': 0, 'Education': 0, 
    'Dependents': 0, 'CoapplicantIncome': 0, 'LoanAmount': X['LoanAmount'].median(), 
    'Loan_Amount_Term': X['Loan_Amount_Term'].median(), 'TotalIncome': 0, 
    'LoanAmount_to_Income': X['LoanAmount_to_Income'].median()
}
try:
    X = X.fillna(fill_values)
    X_test = X_test.fillna({**fill_values, 'LoanAmount': X_test['LoanAmount'].median(), 
                            'Loan_Amount_Term': X_test['Loan_Amount_Term'].median(), 
                            'LoanAmount_to_Income': X_test['LoanAmount_to_Income'].median()})
except Exception as e:
    print(f"Error in filling missing values: {e}")

# Feature engineering
try:
    X['Credit_History_LoanAmount'] = X['Credit_History'] * X['LoanAmount']
    X_test['Credit_History_LoanAmount'] = X_test['Credit_History'] * X_test['LoanAmount']
    features.append('Credit_History_LoanAmount')
except Exception as e:
    print(f"Error in feature engineering: {e}")
    features.append('Credit_History_LoanAmount')

try:
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Credit_History_LoanAmount']:
        cap = X[col].quantile(0.995)
        X.loc[:, col] = X[col].clip(upper=cap)
        X_test.loc[:, col] = X_test[col].clip(upper=cap)
except Exception as e:
    print(f"Error in outlier capping: {e}")

fill_values.update({'Credit_History_LoanAmount': 0})
try:
    X = X.fillna(fill_values)
    X_test = X_test.fillna({**fill_values, 'Credit_History_LoanAmount': X_test['Credit_History_LoanAmount'].median()})
except Exception as e:
    print(f"Error in filling missing values for new features: {e}")

try:
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    print("SMOTE applied successfully.")
    print("\nLoan_Status Distribution (After SMOTE):")
    print(pd.Series(y).value_counts(normalize=True))
except Exception as e:
    print(f"Error applying SMOTE: {e}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Validate feature consistency
if not (X_train.columns == X_val.columns).all() or not (X_val.columns == X_test.columns).all():
    print("Feature mismatch detected. Aligning columns...")
    common_cols = list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    X_test = X_test[common_cols]
    features = common_cols

# CatBoost Tuning with Optuna
cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                'Property_Rural', 'Property_Semiurban', 'Property_Urban']
try:
    def cat_objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1500),
            'depth': trial.suggest_int('depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 20),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0)
        }
        model = cb.CatBoostClassifier(**params, cat_features=cat_features, random_state=42, verbose=0)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
        return accuracy_score(y_val, model.predict(X_val))

    cat_study = optuna.create_study(direction='maximize')
    cat_study.optimize(cat_objective, n_trials=150)
    best_cat_params = cat_study.best_params
    best_cat = cb.CatBoostClassifier(**best_cat_params, cat_features=cat_features, random_state=42, verbose=0)
    best_cat.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)

    print(f"\nCatBoost Best Parameters: {best_cat_params}")
    cat_y_pred = best_cat.predict(X_val)
    cat_accuracy = accuracy_score(y_val, cat_y_pred)
    cat_cv_scores = cross_val_score(best_cat, X, y, cv=5, scoring='accuracy')
    print(f"CatBoost Validation Accuracy: {cat_accuracy:.4f}")
    print(f"CatBoost Cross-Validation Accuracy: {cat_cv_scores.mean():.4f} ± {cat_cv_scores.std():.4f}")

    cat_feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': best_cat.get_feature_importance()
    }).sort_values(by='Importance', ascending=False)
    print("\nCatBoost Feature Importance (Top 5):")
    print(cat_feature_importance.head())
except Exception as e:
    print(f"Error in CatBoost tuning: {e}")
    best_cat = cb.CatBoostClassifier(random_state=42, verbose=0)
    best_cat.fit(X_train, y_train)
    cat_accuracy = accuracy_score(y_val, best_cat.predict(X_val))
    cat_feature_importance = pd.DataFrame({'Feature': features, 'Importance': [0] * len(features)})

# Select best model
best_model = best_cat
best_model_name = 'CatBoost'
best_accuracy = cat_accuracy
best_importance = cat_feature_importance
print(f"\nBest Model: {best_model_name} with Validation Accuracy: {best_accuracy:.4f}")

try:
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to: {model_path}")
except Exception as e:
    print(f"Error saving model: {e}")

try:
    test_predictions = best_model.predict(X_test)
    submission = pd.DataFrame({
        'Loan_ID': test_df['Loan_ID'],
        'Loan_Status': test_predictions
    })
    submission['Loan_Status'] = submission['Loan_Status'].map({1: 'Y', 0: 'N'})
    submission.to_csv(submission_path, index=False)
    print(f"\nFirst 5 rows of submission.csv:")
    print(submission.head())
except Exception as e:
    print(f"Error in prediction or submission: {e}")

try:
    print(f"\nFeature Importance for {best_model_name} (Top 10):")
    print(best_importance.head(10))

    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = best_importance.head(10)
    ax.bar(top_features['Feature'], top_features['Importance'], 
           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_title(f'Top 10 Feature Importance ({best_model_name})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to: {plot_path}")
    plt.close()
except Exception as e:
    print(f"Error in feature importance or plotting: {e}")
