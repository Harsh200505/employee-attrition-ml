"""Reproducible undergraduate ML experiment. Run: python train.py"""
from pathlib import Path
import hashlib
import json
import platform
import os

# Keep Matplotlib's optional cache in this project's writable directory.
os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd() / '.mplconfig'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, PrecisionRecallDisplay)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
OUT = ROOT / 'results'
SEED = 42


def load_data():
    df = pd.read_csv(DATA)
    if set(df['Attrition'].unique()) != {'Yes', 'No'}:
        raise ValueError('Expected Attrition labels Yes and No.')
    if df['EmployeeNumber'].duplicated().any():
        raise ValueError('Duplicate employee IDs require a grouped split.')
    return df


def split_data(df):
    # Explicit design choices, fixed before exploring relationships or test scores.
    # Remove the ID, documented constant fields, and gender/marital status.
    # Excluding two personal attributes does NOT establish fairness; proxies remain.
    excluded = ['Attrition', 'EmployeeNumber', 'EmployeeCount', 'StandardHours',
                'Over18', 'Gender', 'MaritalStatus']
    X = df.drop(columns=excluded)
    y = df['Attrition'].map({'No': 0, 'Yes': 1})
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def make_models(X_train):
    numeric = X_train.select_dtypes(include='number').columns.tolist()
    categorical = X_train.select_dtypes(exclude='number').columns.tolist()
    preprocessing = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ])
    estimators = {
        'Majority baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                               class_weight='balanced', random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8,
                                               min_samples_leaf=3, class_weight='balanced',
                                               random_state=SEED, n_jobs=1),
    }
    return {name: Pipeline([('prepare', clone(preprocessing)), ('model', model)])
            for name, model in estimators.items()}


def compare_models(models, X_train, y_train):
    # Every fold fits its own imputer, encoder and scaler via the pipeline.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {'average_precision': 'average_precision', 'f1': 'f1',
              'recall': 'recall', 'accuracy': 'accuracy'}
    rows = []
    for name, pipeline in models.items():
        result = cross_validate(pipeline, X_train, y_train, cv=folds,
                                scoring=scores, n_jobs=1, error_score='raise')
        row = {'model': name}
        for metric in scores:
            row[f'{metric}_mean'] = float(result[f'test_{metric}'].mean())
            row[f'{metric}_std'] = float(result[f'test_{metric}'].std())
        rows.append(row)
    return pd.DataFrame(rows).sort_values('average_precision_mean', ascending=False)


def exploratory_analysis(X_train, y_train):
    """Use training data for exploration so the test set stays held out."""
    OUT.mkdir(exist_ok=True)
    exploratory = X_train.assign(Attrition=y_train)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    tables = []
    for ax, column in zip(axes, ['OverTime', 'Department', 'JobSatisfaction']):
        table = exploratory.groupby(column)['Attrition'].agg(['size', 'sum', 'mean'])
        table.columns = ['employees', 'left', 'attrition_fraction']
        tables.append(table.reset_index().rename(columns={column: 'group'}).assign(feature=column))
        ax.bar(table.index.astype(str), table['attrition_fraction'] * 100, color='#236d93')
        ax.set_title(column)
        ax.set_ylabel('Attrition within group (%)')
        ax.tick_params(axis='x', labelrotation=25)
    fig.suptitle('Training data: associations, not causes')
    fig.tight_layout()
    fig.savefig(OUT / 'training_eda.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    pd.concat(tables, ignore_index=True).to_csv(OUT / 'training_group_rates.csv', index=False)


def final_evaluation(models, selected, X_train, X_test, y_train, y_test):
    """Evaluate the CV-selected model and predefined baseline at threshold 0.5."""
    results = {}
    fitted = None
    for name in dict.fromkeys(['Majority baseline', selected]):
        pipeline = models[name].fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        results[name] = {
            'accuracy': float(accuracy_score(y_test, predictions)),
            'precision': float(precision_score(y_test, predictions, zero_division=0)),
            'recall': float(recall_score(y_test, predictions, zero_division=0)),
            'f1': float(f1_score(y_test, predictions, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, probabilities)),
            'average_precision': float(average_precision_score(y_test, probabilities)),
            'confusion_matrix': confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        }
        if name == selected:
            fitted = pipeline
            pd.DataFrame({'source_row': X_test.index, 'actual_left': y_test,
                          'predicted_left': predictions, 'score': probabilities}).to_csv(
                              OUT / 'test_predictions.csv', index=False)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=[0, 1],
                display_labels=['Stayed', 'Left'], ax=axes[0], colorbar=False, cmap='Blues')
            PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=axes[1], name=selected)
            axes[1].axhline(y_test.mean(), linestyle='--', color='gray', label='Test prevalence')
            axes[1].legend()
            fig.suptitle(f'Held-out test: {selected}')
            fig.tight_layout()
            fig.savefig(OUT / 'test_evaluation.png', dpi=160, bbox_inches='tight')
            plt.close(fig)
    return results, fitted


def main():
    OUT.mkdir(exist_ok=True)
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    audit = {'rows': len(df), 'columns': len(df.columns),
             'missing_cells': int(df.isna().sum().sum()),
             'duplicate_rows': int(df.duplicated().sum()),
             'left': int((df.Attrition == 'Yes').sum()),
             'train_rows': len(X_train), 'test_rows': len(X_test),
             'train_left': int(y_train.sum()), 'test_left': int(y_test.sum())}
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(df)
    (OUT / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (OUT / 'split.json').write_text(json.dumps({'train': X_train.index.tolist(),
        'test': X_test.index.tolist()}, indent=2), encoding='utf-8')
    exploratory_analysis(X_train, y_train)
    models = make_models(X_train)
    comparison = compare_models(models, X_train, y_train)
    comparison.to_csv(OUT / 'cv_comparison.csv', index=False)
    selected = comparison.iloc[0]['model']
    results, fitted = final_evaluation(models, selected, X_train, X_test, y_train, y_test)
    manifest = {'selected_model': selected, 'selection_metric': 'training 5-fold average precision',
                'seed': SEED, 'threshold': 0.5, 'dataset_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
                'python': platform.python_version(), 'sklearn': sklearn.__version__,
                'pandas': pd.__version__, 'numpy': np.__version__, 'test_metrics': results}
    (OUT / 'metrics.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # Coefficients describe this fitted model; they are not causal effects.
    if hasattr(fitted.named_steps['model'], 'coef_'):
        pd.DataFrame({'feature': fitted.named_steps['prepare'].get_feature_names_out(),
            'coefficient': fitted.named_steps['model'].coef_[0]}).sort_values(
                'coefficient', ascending=False).to_csv(OUT / 'model_coefficients.csv', index=False)
    print('DATA AUDIT\n', json.dumps(audit, indent=2))
    print('\nTRAINING CROSS-VALIDATION\n', comparison.round(3).to_string(index=False))
    print('\nSELECTED:', selected)
    print('\nHELD-OUT TEST\n', json.dumps(results, indent=2))
    return manifest


if __name__ == '__main__':
    main()
