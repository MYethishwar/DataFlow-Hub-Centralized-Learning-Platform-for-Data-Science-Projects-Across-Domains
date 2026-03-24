# model_utils.py
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import shap
import optuna
from typing import Dict, List, Any, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def ensemble_feature_selection(X: pd.DataFrame, y: pd.Series, task_type: str, n_features: int = 10) -> Dict:
    """Perform ensemble feature selection using multiple methods"""
    try:
        # Correlation based selection
        corr_scores = {}
        if task_type == 'classification':
            mi_scores = mutual_info_classif(X, y)
        else:
            mi_scores = mutual_info_regression(X, y)
            
        feature_scores = {}
        for idx, col in enumerate(X.columns):
            feature_scores[col] = float(mi_scores[idx])
            
        # Sort features by importance
        sorted_features = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'selected_features': list(sorted_features.keys())[:n_features],
            'feature_scores': sorted_features,
            'selection_method': 'ensemble'
        }
    except Exception as e:
        raise Exception(f"Feature selection error: {str(e)}")

def detect_outliers(df: pd.DataFrame, methods: List[str] = ['zscore', 'isolation_forest']) -> Dict:
    """Detect outliers using multiple methods"""
    try:
        results = {}
        
        if 'zscore' in methods:
            z_scores = np.abs(StandardScaler().fit_transform(df.select_dtypes(include=[np.number])))
            results['zscore'] = (z_scores > 3).any(axis=1)
            
        if 'isolation_forest' in methods:
            iso_forest = IsolationForest(random_state=42)
            results['isolation_forest'] = iso_forest.fit_predict(df.select_dtypes(include=[np.number])) == -1
            
        if 'dbscan' in methods:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            results['dbscan'] = dbscan.fit_predict(StandardScaler().fit_transform(df.select_dtypes(include=[np.number]))) == -1
        
        # Combine results
        combined_outliers = pd.DataFrame(results).any(axis=1)
        
        return {
            'outlier_indices': combined_outliers[combined_outliers].index.tolist(),
            'outlier_counts': {method: results[method].sum() for method in methods},
            'total_outliers': combined_outliers.sum()
        }
    except Exception as e:
        raise Exception(f"Outlier detection error: {str(e)}")

def calculate_data_quality_score(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive data quality score"""
    try:
        scores = {
            'completeness': (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'uniqueness': (1 - df.duplicated().sum() / len(df)) * 100,
            'consistency': analyze_data_consistency(df),
            'validity': check_data_validity(df)
        }
        
        # Calculate overall score
        overall_score = np.mean(list(scores.values()))
        
        return {
            'overall_score': float(overall_score),
            'component_scores': scores,
            'recommendations': generate_quality_recommendations(scores)
        }
    except Exception as e:
        raise Exception(f"Data quality scoring error: {str(e)}")

def analyze_data_consistency(df: pd.DataFrame) -> float:
    """Analyze data consistency"""
    consistency_score = 100.0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Check for values outside 3 standard deviations
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(df[col] > mean + 3*std) | (df[col] < mean - 3*std)][col]
        consistency_score -= (len(outliers) / len(df)) * 10
    
    return max(0, consistency_score)

def check_data_validity(df: pd.DataFrame) -> float:
    """Check data validity"""
    validity_score = 100.0
    
    for col in df.columns:
        missing = df[col].isnull().sum()
        validity_score -= (missing / len(df)) * 10
        
        if df[col].dtype == 'object':
            # Check for inconsistent string patterns
            patterns = df[col].str.len().value_counts()
            if len(patterns) > 10:  # If too many different lengths
                validity_score -= 5
    
    return max(0, validity_score)

def generate_quality_recommendations(scores: Dict) -> List[str]:
    """Generate recommendations based on quality scores"""
    recommendations = []
    
    if scores['completeness'] < 95:
        recommendations.append("Consider handling missing values")
    if scores['uniqueness'] < 95:
        recommendations.append("Check for and handle duplicate records")
    if scores['consistency'] < 90:
        recommendations.append("Investigate and handle data inconsistencies")
    if scores['validity'] < 90:
        recommendations.append("Validate and clean data formats")
        
    return recommendations

def optimize_hyperparameters(model_class, X: pd.DataFrame, y: pd.Series, param_space: Dict) -> Dict:
    """Optimize model hyperparameters using Optuna"""
    def objective(trial):
        params = {}
        for param, space in param_space.items():
            if space['type'] == 'int':
                params[param] = trial.suggest_int(param, space['low'], space['high'])
            elif space['type'] == 'float':
                params[param] = trial.suggest_float(param, space['low'], space['high'])
            elif space['type'] == 'categorical':
                params[param] = trial.suggest_categorical(param, space['choices'])
        
        model = model_class(**params)
        score = cross_val_score(model, X, y, cv=5).mean()
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    return {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'optimization_history': study.trials_dataframe().to_dict()
    }

def analyze_model_errors(y_true: pd.Series, y_pred: pd.Series, X: pd.DataFrame) -> Dict:
    """Analyze prediction errors"""
    try:
        errors = y_true != y_pred if isinstance(y_true[0], str) else np.abs(y_true - y_pred)
        error_indices = np.where(errors)[0]
        
        error_analysis = {
            'error_count': len(error_indices),
            'error_rate': len(error_indices) / len(y_true),
            'feature_correlation': analyze_error_feature_correlation(X, errors),
            'error_distribution': analyze_error_distribution(errors)
        }
        
        return error_analysis
    except Exception as e:
        raise Exception(f"Error analysis failed: {str(e)}")

def analyze_error_feature_correlation(X: pd.DataFrame, errors: np.ndarray) -> Dict:
    """Analyze correlation between features and errors"""
    correlations = {}
    for column in X.columns:
        corr = np.corrcoef(X[column], errors)[0, 1]
        correlations[column] = float(corr)
    
    return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

def analyze_error_distribution(errors: np.ndarray) -> Dict:
    """Analyze the distribution of errors"""
    if isinstance(errors[0], bool):
        return {
            'true_count': int(sum(errors)),
            'false_count': int(sum(~errors)),
            'error_rate': float(sum(errors) / len(errors))
        }
    else:
        return {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'median': float(np.median(errors)),
            'max': float(np.max(errors)),
            'min': float(np.min(errors))
        }