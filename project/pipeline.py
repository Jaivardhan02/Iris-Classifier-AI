"""
===============================================================================
HOW TO RUN THIS SCRIPT:
1. Open a terminal or command prompt.
2. Navigate to the directory containing this file: `cd d:\\IRIS\\project`
3. Execute the script: `python pipeline.py`
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

warnings.filterwarnings('ignore')

def create_dirs():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def load_data(filepath):
    print("--- Loading Data ---")
    df = pd.read_csv(filepath)
    print(f"Data shape: {df.shape}")
    return df

def perform_eda(df):
    print("\n--- Exploratory Data Analysis (EDA) ---")
    
    # 1. Data Types and Basic Statistics
    print("\nData Types:")
    print(df.dtypes)
    print("\nBasic Statistics:")
    print(df.describe())
    
    # 2. Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # 3. Handle 'Id' column (Feature Engineering: Dropping useless features)
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        print("\nDropped 'Id' column as it doesn't carry predictive power.")
    
    # 4. Distribution analysis
    plt.figure(figsize=(10, 8))
    for i, col in enumerate(df.columns[:-1]):
        plt.subplot(2, 2, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig('plots/distributions.png')
    plt.close()
    
    # 5. Correlation analysis
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    print("\nSaved correlation matrix plot.")
    
    # 6. Outlier detection using Boxplots
    plt.figure(figsize=(10, 8))
    for i, col in enumerate(df.columns[:-1]):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='Species', y=col, data=df)
        plt.title(f'{col} by Species')
    plt.tight_layout()
    plt.savefig('plots/boxplots.png')
    plt.close()
    
    return df

def preprocess_data(df):
    print("\n--- Data Preprocessing ---")
    
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    # Encode categorical variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train-test split (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale numerical features (important for SVM and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data split into train and test sets with stratification.")
    print(f"X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")
    
    # Save the scaler and encoder for deployment
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le

def build_and_evaluate_models(X_train, X_test, y_train, y_test, le):
    print("\n--- Model Building & Selection ---")
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    best_model = None
    best_acc = 0
    best_model_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} Performance:")
        print(f"Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name

    print(f"\nBest initial model: {best_model_name} with Accuracy: {best_acc:.4f}")
    
    # Hyperparameter tuning for SVM (assuming it's a good candidate)
    print("\n--- Hyperparameter Tuning for SVM ---")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    
    grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, verbose=0, cv=5)
    grid.fit(X_train, y_train)
    
    tuned_model = grid.best_estimator_
    print(f"Best SVM Parameters: {grid.best_params_}")
    
    y_pred_tuned = tuned_model.predict(X_test)
    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    print(f"Tuned SVM Accuracy: {tuned_acc:.4f}")
    
    final_model = tuned_model
    
    # Detailed Evaluation
    print("\n--- Model Evaluation (Best Model) ---")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_tuned, target_names=le.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_tuned)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    # ROC-AUC (One-vs-Rest for multi-class)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    y_score = final_model.predict_proba(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {le.classes_[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Multi-class)')
    plt.legend(loc="lower right")
    plt.savefig('plots/roc_curve.png')
    plt.close()

    # Save the final model
    joblib.dump(final_model, 'models/best_model.pkl')
    print("\nSaved best model to 'models/best_model.pkl'")
    
    return final_model

def plot_learning_curve(estimator, X, y):
    print("\n--- Generating Learning Curve ---")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score (Accuracy)")
    plt.legend(loc="best")
    plt.savefig('plots/learning_curve.png')
    plt.close()
    
    print("Learning curve saved. It demonstrates how the model balances bias and variance.")
    print("A small gap between training and validation scores indicates good generalization (low variance).")
    print("High scores for both indicate low bias.")

def main():
    print("Starting ML Pipeline...")
    create_dirs()
    
    # 1. Load Data
    filepath = r'd:\IRIS\dataset\Iris.csv'
    df = load_data(filepath)
    
    # 2. EDA
    df = perform_eda(df)
    
    # 3. Preprocessing
    X_train, X_test, y_train, y_test, le = preprocess_data(df)
    
    # 4. Model Building & Selection
    best_model = build_and_evaluate_models(X_train, X_test, y_train, y_test, le)
    
    # 5. Overfitting Prevention & Learning Curves
    # We use all scaled data and encoded labels to compute learning curve across folds
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    plot_learning_curve(best_model, X, y)
    
    print("\n--- Bias-Variance Trade-off & Overfitting Prevention Explanation ---")
    print("Bias vs Variance: Logistic Regression usually has higher bias and lower variance.")
    print("Random Forest has low bias but can have high variance (overfit) if not constrained.")
    print("SVM balances bias and variance through the 'C' regularization parameter.")
    print("Regularization: In our SVM, GridSearch tuned 'C'. A smaller 'C' creates a wider margin (more regularization), preventing overfitting.")
    print("Cross-validation: Used CV=5 during tuning and learning curve generation to ensure robust evaluation.")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
