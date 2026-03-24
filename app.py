from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file, current_app
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf.csrf import CSRFProtect
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from asgiref.sync import async_to_sync
from functools import wraps
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score, make_scorer
)
from sklearn.impute import SimpleImputer
import optuna
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import re
import warnings
import os
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from bson import ObjectId
import scipy.stats as stats

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════════════
#  BULLETPROOF DATA HELPERS  — handles ANY dataset, ANY dtype
# ═══════════════════════════════════════════════════════════════════

def fix_pandas_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Fix pandas 3.x StringDtype and all incompatible extension dtypes."""
    df = df.copy()
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if hasattr(df[col].dtype, 'storage') or dtype_str in ('string', 'StringDtype'):
            df[col] = df[col].astype(str).replace({'nan': np.nan, 'None': np.nan, '<NA>': np.nan})
        elif dtype_str.startswith('Int'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype_str.startswith('Float'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif dtype_str.startswith('boolean'):
            df[col] = df[col].astype(float)
    return df


def safe_read_csv(filepath: str) -> pd.DataFrame:
    """Read CSV with automatic separator, encoding and dtype detection."""
    for sep in [',', ';', '\t', '|']:
        for enc in ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig']:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=enc)
                if len(df.columns) > 1:
                    return fix_pandas_dtypes(df)
            except Exception:
                continue
    # Last resort
    try:
        df = pd.read_csv(filepath, sep=None, engine='python')
        return fix_pandas_dtypes(df)
    except Exception as e:
        raise ValueError(f"Cannot read CSV file: {e}")


def clean_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Remove special chars from column names (LightGBM requires clean names).
    Returns cleaned df and a mapping {clean_name: original_name}.
    """
    df = df.copy()
    mapping = {}
    new_cols = []
    seen = {}
    for orig in df.columns:
        clean = re.sub(r'[^A-Za-z0-9_]', '_', str(orig)).strip('_')
        if not clean:
            clean = f'col_{len(new_cols)}'
        if clean in seen:
            seen[clean] += 1
            clean = f'{clean}_{seen[clean]}'
        else:
            seen[clean] = 0
        mapping[clean] = orig
        new_cols.append(clean)
    df.columns = new_cols
    return df, mapping


def find_target_column(df: pd.DataFrame, target_column: str) -> str:
    """Find the actual target column name after column name cleaning."""
    if target_column in df.columns:
        return target_column
    clean_target = re.sub(r'[^A-Za-z0-9_]', '_', str(target_column)).strip('_')
    if clean_target in df.columns:
        return clean_target
    # Try case-insensitive match
    for col in df.columns:
        if col.lower() == target_column.lower():
            return col
        if col.lower() == clean_target.lower():
            return col
    # Try partial match
    for col in df.columns:
        if col.lower().startswith(clean_target[:4].lower()):
            return col
    raise ValueError(
        f"Target column '{target_column}' not found. "
        f"Available columns: {list(df.columns)}"
    )


def nuclear_clean(X: pd.DataFrame) -> pd.DataFrame:
    """
    Final safety net: guarantee NO NaN anywhere in X.
    Uses median for numeric, 0 for anything else.
    """
    X = X.copy()
    for col in X.columns:
        if X[col].isnull().any():
            try:
                fill = X[col].median()
                if pd.isna(fill):
                    fill = 0.0
            except Exception:
                fill = 0.0
            X[col] = X[col].fillna(fill)
    # Final check: replace inf values too
    X = X.replace([np.inf, -np.inf], 0.0)
    return X


def robust_preprocess(df: pd.DataFrame, target_column: str, task_type: str):
    """
    THE definitive preprocessing pipeline — handles ANY dataset:

    Steps:
    1. Fix pandas 3.x dtypes
    2. Clean column names (special chars removed)
    3. Find actual target column
    4. Drop rows where target is NaN
    5. Drop columns with >70% missing values
    6. Drop ID-like columns (all unique strings)
    7. Label encode all string/categorical columns
    8. Impute missing values (median for numeric, most_frequent for categorical)
    9. Convert everything to float64
    10. Nuclear NaN check — replace any remaining NaN/inf with 0
    11. Encode target (LabelEncoder for classification)

    Returns: X (clean DataFrame), y (array), encoders (dict), clean_target (str)
    """
    # Step 1: Fix dtypes
    df = fix_pandas_dtypes(df.copy())

    # Step 2: Clean column names
    df, col_mapping = clean_column_names(df)

    # Step 3: Find target
    clean_target = find_target_column(df, target_column)

    # Step 4: Drop rows where target is missing
    df = df.dropna(subset=[clean_target]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No data remaining after dropping rows with missing target values.")

    encoders = {}
    X = df.drop(columns=[clean_target]).copy()
    y_raw = df[clean_target].copy()

    # Step 5: Drop columns with >70% missing
    missing_frac = X.isnull().mean()
    cols_drop_missing = missing_frac[missing_frac > 0.7].index.tolist()
    X = X.drop(columns=cols_drop_missing)

    # Step 6: Drop ID-like columns (all unique string values with high cardinality)
    cols_drop_id = []
    for col in X.columns:
        if X[col].dtype == 'object':
            if X[col].nunique() / len(X) > 0.95:  # >95% unique = likely ID
                cols_drop_id.append(col)
    X = X.drop(columns=cols_drop_id)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after preprocessing.")

    # Step 7: Label encode all string/categorical columns
    for col in X.columns:
        col_dtype = str(X[col].dtype)
        if X[col].dtype == 'object' or col_dtype in ('string', 'StringDtype'):
            # Replace NaN with string 'missing' before encoding
            X[col] = X[col].astype(str).replace({'nan': 'missing', 'None': 'missing', '<NA>': 'missing'})
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col]).astype(float)
            encoders[col] = le
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Step 8: Fix imputation without using SimpleImputer (avoids shape mismatch)
    X = X.replace([np.inf, -np.inf], np.nan)  # inf -> NaN
    X = X.dropna(axis=1, how='all')            # drop all-NaN columns
    X = X.loc[:, X.isnull().mean() < 0.99]    # drop >99% NaN

    # Manual median imputation col by col (no shape mismatch possible)
    for col in list(X.columns):
        if X[col].isnull().any():
            med = X[col].median()
            X[col] = X[col].fillna(med if not pd.isna(med) else 0.0)

    X = X.astype(np.float64).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Step 11: Encode target
    if task_type == 'classification' or y_raw.dtype == 'object' or str(y_raw.dtype) in ('string', 'StringDtype'):
        y_raw = y_raw.astype(str).replace({'nan': 'missing', 'None': 'missing'})
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_raw).astype(np.int64)
        encoders['target'] = le_target
    else:
        y = pd.to_numeric(y_raw, errors='coerce').values.astype(np.float64)
        # Remove rows where y is NaN (shouldn't happen but safety)
        valid_mask = ~np.isnan(y)
        if not valid_mask.all():
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask]

    return X, y, encoders, clean_target


# ═══════════════════════════════════════════════════════════════════
#  SUPPORTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def calculate_feature_importance(X: pd.DataFrame, y, task_type: str) -> dict:
    try:
        if task_type == 'classification':
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            importance_metric = mutual_info_classif
        else:
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            importance_metric = mutual_info_regression

        rf.fit(X, y)
        rf_scores = dict(zip(X.columns, rf.feature_importances_))
        mi_scores_arr = importance_metric(X, y)
        mi_scores = dict(zip(X.columns, mi_scores_arr))

        feature_ranks = {
            feat: (rf_scores.get(feat, 0) + mi_scores.get(feat, 0)) / 2
            for feat in X.columns
        }
        return {
            'detailed_scores': {'random_forest': rf_scores, 'mutual_information': mi_scores},
            'aggregate_ranks': dict(sorted(feature_ranks.items(), key=lambda x: x[1], reverse=True))
        }
    except Exception as e:
        logging.warning(f"Feature importance failed: {e}")
        return {'detailed_scores': {}, 'aggregate_ranks': {}}


def analyze_target_distribution(df: pd.DataFrame, target_column: str) -> dict:
    if target_column not in df.columns:
        return {'type': 'unknown', 'unique_count': 0, 'missing_count': 0}
    target_analysis = {
        'type': 'categorical' if df[target_column].dtype == 'object' else 'numeric',
        'unique_count': int(df[target_column].nunique()),
        'missing_count': int(df[target_column].isnull().sum())
    }
    if target_analysis['type'] == 'numeric':
        try:
            s = df[target_column].describe()
            target_analysis.update({
                'mean': float(s['mean']), 'std': float(s['std']),
                'min': float(s['min']), 'max': float(s['max']),
                'quartiles': {'25%': float(s['25%']), '50%': float(s['50%']), '75%': float(s['75%'])}
            })
        except Exception:
            pass
    else:
        try:
            vc = df[target_column].value_counts()
            target_analysis.update({
                'value_counts': {str(k): int(v) for k, v in vc.items()},
                'percentages': {str(k): float(v / len(df) * 100) for k, v in vc.items()}
            })
        except Exception:
            pass
    return target_analysis


def analyze_correlations(df: pd.DataFrame) -> dict:
    try:
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if len(numeric_df.columns) < 2:
            return {'has_correlations': False, 'message': 'Not enough numeric columns'}
        correlation_matrix = numeric_df.corr().round(4)
        return {'has_correlations': True, 'correlation_matrix': correlation_matrix.to_dict()}
    except Exception:
        return {'has_correlations': False, 'message': 'Correlation analysis failed'}


def perform_enhanced_eda(df: pd.DataFrame, target_column: str) -> dict:
    missing_values = {}
    for col in df.columns:
        mc = df[col].isnull().sum()
        if mc > 0:
            missing_values[col] = {'count': int(mc), 'percentage': float(mc / len(df) * 100)}

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    eda_results = {
        'basic_info': {
            'total_rows': len(df), 'total_columns': len(df.columns),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'duplicate_rows': int(df.duplicated().sum())
        },
        'missing_values': missing_values,
        'numerical_cols': list(numerical_cols),
        'categorical_cols': list(categorical_cols),
        'correlations': analyze_correlations(df),
        'numerical_stats': {},
        'categorical_stats': {},
        'target_analysis': analyze_target_distribution(df, target_column)
    }

    for col in numerical_cols:
        try:
            s = df[col].describe()
            eda_results['numerical_stats'][col] = {
                'mean': float(s['mean']), 'std': float(s['std']),
                'min': float(s['min']), 'max': float(s['max']),
                'quartiles': {'25%': float(s['25%']), '50%': float(s['50%']), '75%': float(s['75%'])}
            }
        except Exception:
            pass

    for col in categorical_cols:
        try:
            vc = df[col].value_counts()
            eda_results['categorical_stats'][col] = {
                'unique_values': int(vc.count()),
                'top_values': {str(k): int(v) for k, v in vc.head(5).items()},
                'value_counts': {str(k): int(v) for k, v in vc.items()}
            }
        except Exception:
            pass

    return eda_results


def preprocess_dataset(df: pd.DataFrame, target_column: str, task_type: str):
    """Legacy wrapper — calls robust_preprocess internally."""
    try:
        X, y, encoders, clean_target = robust_preprocess(df, target_column, task_type)
        preprocessing_steps = {
            'steps_taken': ['Robust universal preprocessing applied'],
            'dropped_columns': [],
            'encoded_columns': [k for k in encoders if k not in ('target', 'imputer')],
            'scaled_columns': list(X.columns),
            'final_features': list(X.columns)
        }
        return X, y, preprocessing_steps, encoders
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")


# ═══════════════════════════════════════════════════════════════════
#  FLASK APP INIT
# ═══════════════════════════════════════════════════════════════════

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from utils.user_friendly_utils import UserFriendlyWrapper
from config import Config
from database import db_manager
from profiling import DataProfiler
from ai_assistant import AIAssistant

base_dir = os.path.dirname(os.path.abspath(__file__))
for directory in ['uploads', 'models', 'logs']:
    dp = os.path.join(base_dir, directory)
    if not os.path.exists(dp):
        os.makedirs(dp)
        os.chmod(dp, 0o755)

app = Flask(__name__)
app.config.from_object(Config)

ai_assistant = AIAssistant()

csrf = CSRFProtect(app)
app.config['WTF_CSRF_ENABLED'] = False
app.config['WTF_CSRF_SECRET_KEY'] = os.urandom(32)
app.config['WTF_CSRF_TIME_LIMIT'] = 3600

LOG_DIR = os.path.join(str(Path.home()), 'automl_logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, 'app.log')
_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
app.logger.addHandler(_handler)
app.logger.setLevel(logging.INFO)

app.config.update(
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1)
)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


def async_route(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        return async_to_sync(f)(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════
#  MODELS / CLASSES
# ═══════════════════════════════════════════════════════════════════

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['id'])
        self.username = user_data['username']
        self.email = user_data.get('email')
        self.user_data = user_data

    def get_id(self):
        return str(self.id)


class RobustClassificationPipeline:
    def __init__(self, df, target_column='Failure Type'):
        self.df = df.copy()
        self.target_column = target_column
        self.label_encoders = {}
        self.logger = logging.getLogger(__name__)

    def preprocess_data(self):
        try:
            X, y, encoders, _ = robust_preprocess(self.df, self.target_column, 'classification')
            self.label_encoders = encoders
            return X, y
        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            raise

    def train_model(self):
        try:
            X, y = self.preprocess_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                'train_score': float(model.score(X_train, y_train)),
                'test_score': float(model.score(X_test, y_test)),
                'feature_importance': dict(zip(X.columns, model.feature_importances_.tolist()))
            }
            self.model = model
            return metrics
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise


class EnhancedModelTraining:
    def __init__(self, api_key: str = None):
        self.logger = logging.getLogger(__name__)
        self.best_preprocessing = None
        self.feature_importance = None
        self.ai_assistant = AIAssistant()

    async def get_optimal_preprocessing(self, df: pd.DataFrame, target_column: str) -> dict:
        return self._get_default_preprocessing(df)

    def _get_default_preprocessing(self, df: pd.DataFrame) -> dict:
        return {
            'missing_values': {'method': 'mean', 'columns': df.columns[df.isnull().any()].tolist()},
            'scaling': {'method': 'standard', 'columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist()},
            'encoding': {'method': 'label', 'columns': df.select_dtypes(include=['object']).columns.tolist()},
            'feature_selection': {'method': 'mutual_info', 'n_features': min(df.shape[1] - 1, 20)},
            'outlier_treatment': {'method': 'clip', 'columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist()}
        }

    def preprocess_data(self, df: pd.DataFrame, target_column: str, preprocessing_config: dict) -> Tuple[pd.DataFrame, dict]:
        df_processed = df.copy()
        preprocessing_info = {'steps': []}
        for col in preprocessing_config['missing_values']['columns']:
            if col in df_processed.columns:
                method = preprocessing_config['missing_values']['method']
                if method == 'mean' and pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                else:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median() if pd.api.types.is_numeric_dtype(df_processed[col]) else df_processed[col].mode().iloc[0] if len(df_processed[col].mode()) > 0 else 0)
                preprocessing_info['steps'].append(f"Filled missing values in {col}")
        numeric_cols = [c for c in preprocessing_config['scaling']['columns'] if c in df_processed.columns]
        scaler = StandardScaler()
        if numeric_cols:
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
            preprocessing_info['steps'].append("Applied standard scaling")
        if preprocessing_config['outlier_treatment']['method'] == 'clip':
            for col in preprocessing_config['outlier_treatment']['columns']:
                if col in df_processed.columns:
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df_processed[col] = df_processed[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        return df_processed, preprocessing_info

    def select_features(self, X: pd.DataFrame, y, task_type: str, n_features: int) -> List[str]:
        k = min(n_features, X.shape[1])
        if task_type == 'classification':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X, y)
        feature_scores = dict(zip(X.columns, selector.scores_))
        self.feature_importance = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in self.feature_importance[:k]]

    def _calculate_metrics(self, y_true, y_pred, task_type: str) -> dict:
        if task_type == 'classification':
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            }
        else:
            return {
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'r2_score': float(r2_score(y_true, y_pred))
            }


class ModelOptimizer:
    def __init__(self, model=None):
        self.model = model
        self.ai_assistant = AIAssistant()

    def get_optimal_params(self, df, target_column, task_type, model_name):
        fallback = {
            'random_forest': {'n_estimators': 200, 'max_depth': 15},
            'gradient_boosting': {'n_estimators': 150, 'learning_rate': 0.1},
            'xgboost': {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.1},
            'lightgbm': {'n_estimators': 200, 'num_leaves': 31}
        }
        try:
            prompt = f"Task: {task_type}, Model: {model_name}. Return ONLY a JSON object with hyperparameter names and values."
            response = self.ai_assistant.generate_response(prompt)
            return json.loads(response)
        except Exception:
            return fallback.get(model_name, {'iterations': 200, 'depth': 10})

    def apply_feature_engineering(self, df, steps):
        df_new = df.copy()
        for feat1, feat2 in steps.get('interactions', []):
            if feat1 in df.columns and feat2 in df.columns:
                df_new[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
        for feat in steps.get('polynomials', []):
            if feat in df.columns:
                df_new[f'{feat}_squared'] = df[feat] ** 2
        for feat, transform in steps.get('transformations', {}).items():
            if feat in df.columns:
                if transform == 'log':
                    df_new[f'{feat}_log'] = np.log1p(df[feat].clip(lower=0))
                elif transform == 'sqrt':
                    df_new[f'{feat}_sqrt'] = np.sqrt(df[feat].clip(lower=0))
        for feat in steps.get('bin_features', []):
            if feat in df.columns:
                try:
                    df_new[f'{feat}_bins'] = pd.qcut(df[feat], q=5, labels=False, duplicates='drop')
                except Exception:
                    pass
        return df_new


def process_maintenance_data(filepath):
    try:
        df = safe_read_csv(filepath)
        required_columns = ['Failure Type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        pipeline = RobustClassificationPipeline(df)
        results = pipeline.train_model()
        return {'success': True, 'model': pipeline.model, 'encoders': pipeline.label_encoders, 'metrics': results}
    except Exception as e:
        logging.error(f"Failed to process maintenance data: {str(e)}")
        return {'success': False, 'error': str(e)}


@login_manager.user_loader
def load_user(user_id):
    try:
        conn = db_manager.get_mysql_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()
        return User(user_data) if user_data else None
    except Exception as e:
        app.logger.error(f"Error loading user: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = db_manager.get_mysql_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            cursor.close()
        finally:
            conn.close()
        if user and check_password_hash(user['password'], password):
            login_user(User(user))
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        name = request.form.get('name')
        organization = request.form.get('organization')
        conn = db_manager.get_mysql_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            if cursor.fetchone():
                flash('Username or email already exists', 'error')
                return render_template('register.html')
            hashed_password = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, password, email, name, organization, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                (username, hashed_password, email, name, organization, datetime.utcnow())
            )
            conn.commit()
            cursor.close()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/dashboard')
@login_required
def dashboard():
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            files = list(mongo_db.uploads.find({"username": current_user.username}))
            return render_template('dashboard.html', files=files, username=current_user.username)
    except Exception as e:
        app.logger.error(f"Dashboard error: {e}")
        flash('Error loading dashboard', 'error')
        return redirect(url_for('home'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        app.logger.info("Upload request received")
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        file = request.files['file']
        task_type = request.form.get('task_type')
        target_column = request.form.get('target_column')

        app.logger.info(f"Received file: {file.filename}, task_type: {task_type}, target_column: {target_column}")

        if not file or not file.filename:
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        if not task_type or not target_column:
            return jsonify({'success': False, 'error': 'Task type and target column are required'}), 400
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            file.save(filepath)
            app.logger.info(f"File saved to {filepath}")
        except Exception as e:
            return jsonify({'success': False, 'error': 'Error saving file'}), 500

        try:
            df = safe_read_csv(filepath)
            # Check target column exists (fuzzy match)
            try:
                actual_target = find_target_column(df, target_column)
            except ValueError:
                os.remove(filepath)
                app.logger.error(f"Target column {target_column} not found in CSV")
                return jsonify({
                    'success': False,
                    'error': f'Target column "{target_column}" not found. Available: {list(df.columns)}'
                }), 400
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            app.logger.error(f"Error reading CSV: {e}")
            return jsonify({'success': False, 'error': f'Invalid CSV: {str(e)}'}), 400

        try:
            file_info = {
                "user_id": current_user.id,
                "username": current_user.username,
                "filename": unique_filename,
                "original_filename": filename,
                "filepath": filepath,
                "task_type": task_type,
                "target_column": target_column,
                "upload_date": datetime.utcnow(),
                "status": "uploaded",
                "columns": list(df.columns),
                "rows": len(df)
            }
            with db_manager.get_mongo_connection() as mongo_db:
                result = mongo_db.uploads.insert_one(file_info)
                app.logger.info(f"File info saved to MongoDB with id: {result.inserted_id}")
            return jsonify({'success': True, 'file_id': str(result.inserted_id), 'message': 'File uploaded successfully'})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': 'Error saving to database'}), 500

    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/generate_profile/<file_id>')
@login_required
@async_route
async def generate_profile(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(file_id), "username": current_user.username})
            if not file_info:
                return jsonify({'error': 'File not found'}), 404
            df = safe_read_csv(file_info['filepath'])
            profiler = DataProfiler(df)
            html_report = profiler.generate_html_report()
            report_path = os.path.join(app.config['UPLOAD_FOLDER'], f'profile_{file_id}.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            return send_file(report_path, as_attachment=False)
    except Exception as e:
        app.logger.error(f"Profile generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/process-explainer')
def process_explainer():
    return render_template('process_explainer.html')


@app.route('/train/<file_id>', methods=['GET', 'POST'])
@login_required
@async_route
async def train_model(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(file_id), "username": current_user.username})
            if not file_info:
                return jsonify({'error': 'File not found'}), 404

            df = safe_read_csv(file_info['filepath'])
            target_column = file_info['target_column']

            # Resolve actual target column
            try:
                actual_target = find_target_column(df, target_column)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400

            # Auto-detect task type
            target_series = df[actual_target].dropna()
            is_categorical = (
                target_series.dtype == 'object' or
                str(target_series.dtype) in ('string', 'StringDtype') or
                target_series.nunique() <= 10
            )
            task_type = 'classification' if is_categorical else 'regression'

            mongo_db.uploads.update_one({"_id": ObjectId(file_id)}, {"$set": {"task_type": task_type}})
            file_info['task_type'] = task_type

            # ── GET: show training page ──
            if request.method == 'GET':
                try:
                    analysis_result = await ai_assistant.analyze_data(df, target_column, task_type)
                    eda_text = analysis_result.get('analysis', '') if isinstance(analysis_result, dict) else ''
                except Exception:
                    eda_text = ''
                return render_template('train_model.html', file_info=file_info, eda_results=eda_text)

            # ── POST: train ──
            app.logger.info(f"Starting model training for file {file_id}")
            training_start_time = datetime.utcnow()

            # ── BULLETPROOF PREPROCESSING ──
            try:
                X, y, encoders, clean_target_col = robust_preprocess(df, target_column, task_type)
                app.logger.info(f"Preprocessing done: {X.shape[0]} rows, {X.shape[1]} features, task={task_type}")
            except Exception as prep_err:
                import traceback
                app.logger.error(f"Preprocessing failed: {prep_err}")
                app.logger.error(f"Full traceback:\n{traceback.format_exc()}")
                return jsonify({'error': f'Preprocessing failed: {str(prep_err)}', 'traceback': traceback.format_exc()}), 500

            # One more nuclear clean just before model training
            X = nuclear_clean(X)

            preprocessing_info = {
                'steps_taken': [
                    'Auto-detected CSV separator and encoding',
                    'Fixed pandas 3.x StringDtype',
                    'Cleaned column names (removed special characters)',
                    'Dropped columns with >70% missing values',
                    'Dropped ID-like columns (>95% unique)',
                    'Label encoded all categorical/string columns',
                    'Imputed all missing values (SimpleImputer median)',
                    'Replaced all Inf/-Inf values with 0',
                    'Converted all features to float64',
                    'Applied StandardScaler'
                ],
                'encoded_columns': [k for k in encoders if k not in ('target', 'imputer')],
                'final_features': list(X.columns),
                'dropped_columns': [],
                'scaled_columns': list(X.columns),
                'n_samples': int(len(X)),
                'n_features': int(X.shape[1])
            }

            # Scale features
            scaler = StandardScaler()
            X_scaled_arr = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled_arr, columns=X.columns, dtype=np.float64)
            # Final nuclear clean after scaling
            X_scaled = nuclear_clean(X_scaled)

            # Safe CV split count
            n_samples = len(X_scaled)
            n_splits = 5
            if n_samples < 50:
                n_splits = max(2, n_samples // 10)
            elif n_samples < 100:
                n_splits = 3

            # Train / test split
            try:
                if task_type == 'classification' and len(np.unique(y)) >= 2:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled.values, y, test_size=0.2, random_state=42, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled.values, y, test_size=0.2, random_state=42
                    )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled.values, y, test_size=0.2, random_state=42
                )

            # Convert to DataFrames for feature names (LightGBM needs them)
            X_train_df = pd.DataFrame(X_train, columns=X_scaled.columns)
            X_test_df = pd.DataFrame(X_test, columns=X_scaled.columns)

            # Final nuclear clean on splits
            X_train_df = nuclear_clean(X_train_df)
            X_test_df = nuclear_clean(X_test_df)

            # ── DEFINE MODELS ──
            if task_type == 'classification':
                models = {
                    'random_forest': RandomForestClassifier(
                        n_estimators=100, random_state=42, n_jobs=-1
                    ),
                    'gradient_boosting': GradientBoostingClassifier(
                        n_estimators=100, random_state=42
                    ),
                    'xgboost': XGBClassifier(
                        random_state=42, eval_metric='logloss',
                        verbosity=0, use_label_encoder=False,
                        tree_method='hist'
                    ),
                    'lightgbm': LGBMClassifier(
                        random_state=42, verbose=-1,
                        min_data_in_leaf=1, min_child_samples=1,
                        n_jobs=-1
                    ),
                    'catboost': CatBoostClassifier(
                        random_state=42, verbose=False,
                        allow_const_label=True
                    )
                }
            else:
                models = {
                    'random_forest': RandomForestRegressor(
                        n_estimators=100, random_state=42, n_jobs=-1
                    ),
                    'gradient_boosting': GradientBoostingRegressor(
                        n_estimators=100, random_state=42
                    ),
                    'xgboost': XGBRegressor(
                        random_state=42, verbosity=0,
                        tree_method='hist'
                    ),
                    'lightgbm': LGBMRegressor(
                        random_state=42, verbose=-1,
                        min_data_in_leaf=1, min_child_samples=1,
                        n_jobs=-1
                    ),
                    'catboost': CatBoostRegressor(
                        random_state=42, verbose=False
                    )
                }

            results = {}
            fold_scores = []
            fold_precisions = []
            trained_models = {}

            for name, model in models.items():
                try:
                    app.logger.info(f"Training {name} model...")
                    model.fit(X_train_df, y_train)
                    y_pred = model.predict(X_test_df)

                    if task_type == 'classification':
                        metrics = {
                            'accuracy': float(accuracy_score(y_test, y_pred)),
                            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                            'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                        }
                        try:
                            if len(np.unique(y_train)) >= n_splits:
                                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                            else:
                                cv = n_splits
                            cv_scores = cross_val_score(model, X_train_df, y_train, cv=cv, scoring='accuracy', error_score='raise')
                            try:
                                cv_precisions = cross_val_score(model, X_train_df, y_train, cv=cv, scoring='precision_weighted', error_score='raise')
                            except Exception:
                                cv_precisions = cv_scores.copy()
                            fold_scores.extend(cv_scores.tolist())
                            fold_precisions.extend(cv_precisions.tolist())
                        except Exception as cv_err:
                            app.logger.warning(f"CV failed for {name}: {cv_err}")
                            cv_scores = np.array([metrics['accuracy']])
                            cv_precisions = np.array([metrics['precision']])
                    else:
                        metrics = {
                            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                            'mae': float(mean_absolute_error(y_test, y_pred)),
                            'r2': float(r2_score(y_test, y_pred))
                        }
                        try:
                            cv_scores = cross_val_score(
                                model, X_train_df, y_train,
                                cv=n_splits, scoring='neg_root_mean_squared_error',
                                error_score='raise'
                            )
                            fold_scores.extend(cv_scores.tolist())
                        except Exception as cv_err:
                            app.logger.warning(f"CV failed for {name}: {cv_err}")
                            cv_scores = np.array([-metrics['rmse']])

                    metrics['cv_score_mean'] = float(np.mean(np.abs(cv_scores)))
                    metrics['cv_score_std'] = float(np.std(np.abs(cv_scores)))
                    results[name] = metrics
                    trained_models[name] = model
                    app.logger.info(f"Successfully trained {name} model")

                except Exception as model_error:
                    app.logger.error(f"Error training {name} model: {str(model_error)}")
                    continue

            if not results:
                return jsonify({'error': 'All models failed to train. Please check your dataset.'}), 500

            # Feature importance
            try:
                feature_importance = calculate_feature_importance(X_scaled, y, task_type)
            except Exception:
                feature_importance = {'detailed_scores': {}, 'aggregate_ranks': {}}

            # Best model selection
            if task_type == 'classification':
                best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
            else:
                best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]

            # Save best model
            model_filepath = os.path.join(
                app.config['MODELS_FOLDER'],
                f"{file_id}_{best_model_name}.joblib"
            )
            joblib.dump(trained_models[best_model_name], model_filepath)

            # Performance timeline
            if task_type == 'classification':
                performance_timeline = [
                    {
                        'name': f'Fold {i+1}',
                        'accuracy': float(fold_scores[i]),
                        'precision': float(fold_precisions[i]) if i < len(fold_precisions) else 0.0
                    }
                    for i in range(len(fold_scores))
                ]
            else:
                performance_timeline = [
                    {'name': f'Fold {i+1}', 'rmse': float(abs(fold_scores[i]))}
                    for i in range(len(fold_scores))
                ]

            training_end_time = datetime.utcnow()
            training_duration = (training_end_time - training_start_time).total_seconds()

            dashboard_data = {
                'best_model': {
                    'name': best_model_name,
                    'metrics': results[best_model_name],
                    'parameters': trained_models[best_model_name].get_params()
                },
                'feature_count': int(X.shape[1]),
                'training_time': float(training_duration),
                'cv_score': float(results[best_model_name]['cv_score_mean']),
                'model_comparison': [{'name': n, 'metrics': m} for n, m in results.items()],
                'performance_timeline': performance_timeline,
                'feature_importance': [
                    {'feature': str(feat), 'importance': float(score)}
                    for feat, score in feature_importance['aggregate_ranks'].items()
                ]
            }

            report_data = {
                'user_id': current_user.id,
                'username': current_user.username,
                'file_id': file_id,
                'preprocessing_info': preprocessing_info,
                'feature_importance': feature_importance,
                'results': results,
                'model_filepath': model_filepath,
                'created_at': training_start_time,
                'completed_at': training_end_time,
                'training_duration': float(training_duration),
                'status': 'completed',
                'task_type': task_type,
                'target_column': target_column,
                'dashboard_data': dashboard_data,
                'best_model': {'name': best_model_name, 'metrics': results[best_model_name]}
            }

            report_id = mongo_db.model_reports.insert_one(report_data).inserted_id
            mongo_db.uploads.update_one(
                {"_id": ObjectId(file_id)},
                {"$set": {
                    "status": "trained",
                    "last_training": training_end_time,
                    "best_model": best_model_name,
                    "latest_report_id": str(report_id)
                }}
            )

            return jsonify({
                'success': True,
                'report_id': str(report_id),
                'results': results,
                'dashboard_data': dashboard_data,
                'message': f'Models trained successfully. Best model: {best_model_name}'
            })

    except Exception as e:
        app.logger.error(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/ai/analyze_data/<file_id>')
@login_required
@async_route
async def ai_analyze_data(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(file_id), "username": current_user.username})
            if not file_info:
                return jsonify({'error': 'File not found'}), 404
            df = safe_read_csv(file_info['filepath'])
            dataset_info = {
                'basic_info': {
                    'rows': len(df), 'columns': len(df.columns),
                    'task_type': file_info['task_type'],
                    'target_column': file_info['target_column']
                },
                'columns': {}
            }
            for col in df.columns:
                try:
                    dataset_info['columns'][col] = {
                        'type': str(df[col].dtype),
                        'unique_values': int(df[col].nunique()),
                        'missing_values': int(df[col].isnull().sum())
                    }
                    if df[col].dtype in [np.float64, np.int64]:
                        s = df[col].describe()
                        dataset_info['columns'][col].update({
                            'mean': float(s['mean']), 'std': float(s['std']),
                            'min': float(s['min']), 'max': float(s['max'])
                        })
                except Exception:
                    pass
            prompt = f"""
Analyze this dataset for machine learning:
Task Type: {file_info['task_type']}, Target: {file_info['target_column']}
Rows: {dataset_info['basic_info']['rows']}, Columns: {dataset_info['basic_info']['columns']}
Column Details: {json.dumps(dataset_info['columns'], indent=2)}
Provide:
1. Data Quality Assessment
2. Feature Engineering Suggestions
3. Preprocessing Recommendations
4. Modeling Approach
5. Potential Challenges
Format with markdown headings and bullet points.
"""
            try:
                response = ai_assistant.generate_response(prompt)
                return jsonify({'success': True, 'analysis': response})
            except Exception as e:
                return jsonify({'success': False, 'error': 'Error generating AI analysis'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


@app.route('/process-explanation')
def process_explanation():
    return render_template('includes/process_explainer.html')


@app.route('/insights/<report_id>')
@login_required
@async_route
async def get_training_insights(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({"_id": ObjectId(report_id), "username": current_user.username})
            if not report:
                return jsonify({'error': 'Report not found'}), 404
            analysis_data = {
                'best_model': report.get('best_model', {}),
                'feature_importance': report.get('feature_importance', {}),
                'preprocessing_steps': report.get('preprocessing_info', {}).get('steps_taken', []),
                'model_metrics': report.get('results', {}),
                'training_duration': report.get('training_duration', 0)
            }
            prompt = f"""
Analyze these ML training results:
{json.dumps(analysis_data, indent=2)}
Provide insights about:
1. Model Performance Analysis
2. Feature Importance Interpretation
3. Preprocessing Effectiveness
4. Recommendations for Improvement
Format with markdown.
"""
            response = ai_assistant.generate_response(prompt)
            mongo_db.model_reports.update_one(
                {"_id": ObjectId(report_id)},
                {"$set": {"ai_insights": response, "insights_generated_at": datetime.utcnow()}}
            )
            return jsonify({'success': True, 'insights': response})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/ai/model_insights/<report_id>')
@login_required
@async_route
async def get_model_insights(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({"_id": ObjectId(report_id), "username": current_user.username})
            if not report:
                return jsonify({'error': 'Report not found'}), 404
            model_summary = {
                'task_type': report['task_type'],
                'target_column': report['target_column'],
                'models': report['results'],
                'preprocessing': report['preprocessing_info'],
                'feature_importance': report.get('feature_importance', {})
            }
            prompt = f"""
Analyze these ML model results:
Task: {model_summary['task_type']}, Target: {model_summary['target_column']}
Model Performance: {json.dumps(model_summary['models'], indent=2)}
Feature Importance summary: {json.dumps(list(model_summary['feature_importance'].get('aggregate_ranks', {}).items())[:10], indent=2)}
Provide:
1. Model Performance Comparison
2. Key Feature Importance Insights
3. Potential Areas for Improvement
4. Cross-validation Stability Analysis
5. Optimization Recommendations
Format with clear markdown sections.
"""
            try:
                response = ai_assistant.generate_response(prompt)
                return jsonify({'success': True, 'insights': response})
            except Exception as e:
                return jsonify({'success': False, 'error': 'Error generating model insights'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


@app.route('/ai/feature_recommendations/<file_id>')
@login_required
@async_route
async def get_feature_recommendations(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(file_id), "username": current_user.username})
            if not file_info:
                return jsonify({'error': 'File not found'}), 404
            df = safe_read_csv(file_info['filepath'])
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            correlations = {}
            if len(numeric_cols) > 1:
                try:
                    correlations = df[numeric_cols].corr().round(4).to_dict()
                except Exception:
                    pass
            prompt = f"""
Generate feature engineering recommendations:
Task: {file_info['task_type']}, Target: {file_info['target_column']}
Numeric Features: {', '.join(numeric_cols[:20])}
Categorical Features: {', '.join(categorical_cols[:20])}
Suggest:
1. Feature transformations
2. Meaningful feature interactions
3. Dimensionality reduction if needed
4. Categorical variable handling
5. New derived features
6. Feature selection strategies
Format with clear sections and examples.
"""
            try:
                response = ai_assistant.generate_response(prompt)
                return jsonify({'success': True, 'recommendations': response})
            except Exception as e:
                return jsonify({'success': False, 'error': 'Error generating feature recommendations'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500


@app.route('/delete_dataset/<file_id>', methods=['POST'])
@login_required
def delete_dataset(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(file_id), "username": current_user.username})
            if not file_info:
                return jsonify({'error': 'Dataset not found'}), 404
            if os.path.exists(file_info['filepath']):
                os.remove(file_info['filepath'])
            mongo_db.uploads.delete_one({"_id": ObjectId(file_id)})
            mongo_db.model_reports.delete_many({"file_id": file_id})
            return jsonify({'success': True, 'message': 'Dataset and associated reports deleted successfully'})
    except Exception as e:
        app.logger.error(f"Error deleting dataset: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/view_dataset/<file_id>')
@login_required
def view_dataset(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(file_id), "username": current_user.username})
            if not file_info:
                flash('Dataset not found', 'error')
                return redirect(url_for('dashboard'))
            df = safe_read_csv(file_info['filepath'])
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            stats = {}
            for col in numeric_cols:
                try:
                    stats[col] = {
                        'mean': float(df[col].mean()), 'std': float(df[col].std()),
                        'min': float(df[col].min()), 'max': float(df[col].max())
                    }
                except Exception:
                    pass
            past_reports = list(mongo_db.model_reports.find(
                {"file_id": str(file_id), "username": current_user.username, "status": "completed"}
            ).sort("created_at", -1))
            return render_template(
                'view_dataset.html',
                file_info=file_info,
                preview_data=df.head(10).to_dict('records'),
                columns=list(df.columns),
                stats=stats,
                shape=df.shape,
                dtypes=df.dtypes.astype(str).to_dict(),
                past_reports=past_reports
            )
    except Exception as e:
        app.logger.error(f"Error viewing dataset: {e}")
        flash('Error loading dataset', 'error')
        return redirect(url_for('dashboard'))


@app.route('/download/preprocessed/<file_id>')
@login_required
def download_preprocessed_data(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(file_id), "username": current_user.username})
            if not file_info:
                flash('File not found', 'error')
                return redirect(url_for('dashboard'))
            df = safe_read_csv(file_info['filepath'])
            try:
                task_type = file_info.get('task_type', 'classification')
                X, y, _, clean_target = robust_preprocess(df, file_info['target_column'], task_type)
                preprocessed_df = X.copy()
                preprocessed_df[clean_target] = y
            except Exception as e:
                preprocessed_df = df.copy()
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'preprocessed_{file_info["filename"]}')
            preprocessed_df.to_csv(temp_path, index=False)
            try:
                return send_file(temp_path, mimetype='text/csv', as_attachment=True,
                                 download_name=f'preprocessed_{file_info["filename"]}')
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    except Exception as e:
        app.logger.error(f"Download error: {str(e)}")
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('dashboard'))


@app.route('/preprocessing_status/<file_id>')
@login_required
def get_preprocessing_status(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(file_id), "username": current_user.username})
            if not file_info:
                return jsonify({'error': 'File not found'}), 404
            return jsonify({'status': 'completed', 'progress': 100})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_model_features/<report_id>')
@login_required
def get_model_features(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({"_id": ObjectId(report_id), "username": current_user.username})
            if not report:
                return jsonify({'error': 'Report not found'}), 404
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(report['file_id'])})
            if not file_info:
                return jsonify({'error': 'Dataset not found'}), 404
            features = report.get('preprocessing_info', {}).get('final_features', [])
            return jsonify({
                'success': True,
                'features': features,
                'feature_importance': {
                    model_name: result.get('feature_importance', {})
                    for model_name, result in report['results'].items()
                    if 'feature_importance' in result
                }
            })
    except Exception as e:
        app.logger.error(f"Error getting model features: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/view_report/<report_id>')
@login_required
def view_report(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({"_id": ObjectId(report_id), "username": current_user.username})
            if not report:
                flash('Report not found', 'error')
                return redirect(url_for('dashboard'))
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(report['file_id'])})
            if not file_info:
                flash('Associated dataset not found', 'error')
                return redirect(url_for('dashboard'))
            return render_template('view_report.html', report=report, file_info=file_info, model_results=report['results'])
    except Exception as e:
        app.logger.error(f"Error viewing report: {e}")
        flash('Error loading report', 'error')
        return redirect(url_for('dashboard'))


@app.route('/deploy/<report_id>', methods=['POST'])
@login_required
def deploy_model(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({"_id": ObjectId(report_id), "username": current_user.username})
            if not report:
                return jsonify({'error': 'Report not found'}), 404
            best_model = None
            best_score = -float('inf')
            for model_name, metrics in report['results'].items():
                score = metrics.get('accuracy', metrics.get('r2', 0))
                if score > best_score:
                    best_model = model_name
                    best_score = score
            deployment_info = {
                'report_id': str(report_id), 'model_name': best_model,
                'username': current_user.username, 'task_type': report['task_type'],
                'target_column': report['target_column'], 'score': float(best_score),
                'deployment_date': datetime.utcnow(), 'status': 'active',
                'preprocessing_info': report['preprocessing_info']
            }
            deployment = mongo_db.deployments.insert_one(deployment_info)
            mongo_db.model_reports.update_one(
                {"_id": ObjectId(report_id)},
                {"$set": {"deployment_id": str(deployment.inserted_id)}}
            )
            return jsonify({
                'success': True,
                'message': f'Model {best_model} deployed successfully',
                'deployment_id': str(deployment.inserted_id)
            })
    except Exception as e:
        app.logger.error(f"Deployment error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/analyze/quality/<file_id>')
@login_required
def analyze_data_quality(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(file_id), "username": current_user.username})
            if not file_info:
                return jsonify({'error': 'File not found'}), 404
            df = safe_read_csv(file_info['filepath'])
            n_cells = df.shape[0] * df.shape[1]
            completeness = float((1 - df.isnull().sum().sum() / n_cells) * 100) if n_cells > 0 else 100.0
            uniqueness = float((1 - df.duplicated().sum() / len(df)) * 100) if len(df) > 0 else 100.0
            quality_scores = {
                'overall_score': float((completeness + uniqueness) / 2),
                'component_scores': {'completeness': completeness, 'uniqueness': uniqueness},
                'recommendations': []
            }
            if completeness < 95:
                quality_scores['recommendations'].append("Consider handling missing values")
            if uniqueness < 95:
                quality_scores['recommendations'].append("Check for and remove duplicate records")
            return jsonify(quality_scores)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze/errors/<report_id>')
@login_required
def analyze_prediction_errors(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({"_id": ObjectId(report_id), "username": current_user.username})
            if not report:
                return jsonify({'error': 'Report not found'}), 404
            file_info = mongo_db.uploads.find_one({"_id": ObjectId(report['file_id'])})
            if not file_info:
                return jsonify({'error': 'Original file not found'}), 404
            error_analysis = {
                'model_performance': report['results'],
                'error_distribution': {
                    model: {
                        'correct_predictions': float(metrics.get('accuracy', metrics.get('r2', 0)) * 100),
                        'incorrect_predictions': float((1 - metrics.get('accuracy', metrics.get('r2', 0))) * 100)
                    }
                    for model, metrics in report['results'].items()
                },
                'feature_importance': report.get('feature_importance', {})
            }
            return jsonify(error_analysis)
    except Exception as e:
        app.logger.error(f"Error analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/download_report/<report_id>')
@login_required
def download_report(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({
                "_id": ObjectId(report_id),
                "username": current_user.username
            })
            if not report:
                return jsonify({'error': 'Report not found'}), 404

            results = report.get('results', {})
            task_type = report.get('task_type', 'classification')
            best_model = report.get('best_model', {})
            feature_importance = report.get('feature_importance', {}).get('aggregate_ranks', {})
            preprocessing = report.get('preprocessing_info', {})

            # Build metrics rows
            metrics_rows = ''
            for model_name, metrics in results.items():
                is_best = model_name == best_model.get('name', '')
                if task_type == 'classification':
                    metrics_rows += f"""
                    <tr style="{'background:#e0faf5;font-weight:bold;' if is_best else ''}">
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">
                            {model_name} {'⭐ Best' if is_best else ''}
                        </td>
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">{metrics.get('accuracy',0)*100:.2f}%</td>
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">{metrics.get('precision',0)*100:.2f}%</td>
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">{metrics.get('recall',0)*100:.2f}%</td>
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">{metrics.get('f1',0)*100:.2f}%</td>
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">{metrics.get('cv_score_mean',0)*100:.2f}%</td>
                    </tr>"""
                else:
                    metrics_rows += f"""
                    <tr style="{'background:#e0faf5;font-weight:bold;' if is_best else ''}">
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">
                            {model_name} {'⭐ Best' if is_best else ''}
                        </td>
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">{metrics.get('rmse',0):.4f}</td>
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">{metrics.get('mae',0):.4f}</td>
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">{metrics.get('r2',0):.4f}</td>
                        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">{metrics.get('cv_score_mean',0):.4f}</td>
                    </tr>"""

            # Build feature importance rows
            feature_rows = ''
            for i, (feat, score) in enumerate(list(feature_importance.items())[:10], 1):
                bar_width = int(score / max(feature_importance.values(), default=1) * 100)
                feature_rows += f"""
                <tr>
                    <td style="padding:6px 12px;border-bottom:1px solid #f3f4f6">{i}</td>
                    <td style="padding:6px 12px;border-bottom:1px solid #f3f4f6">{feat}</td>
                    <td style="padding:6px 12px;border-bottom:1px solid #f3f4f6">{score:.4f}</td>
                    <td style="padding:6px 12px;border-bottom:1px solid #f3f4f6">
                        <div style="background:#0dbfa0;height:10px;border-radius:4px;width:{bar_width}%"></div>
                    </td>
                </tr>"""

            # Header row based on task type
            if task_type == 'classification':
                header_row = '<tr style="background:#065e56;color:#fff"><th style="padding:10px 12px;text-align:left">Model</th><th style="padding:10px 12px;text-align:left">Accuracy</th><th style="padding:10px 12px;text-align:left">Precision</th><th style="padding:10px 12px;text-align:left">Recall</th><th style="padding:10px 12px;text-align:left">F1</th><th style="padding:10px 12px;text-align:left">CV Score</th></tr>'
            else:
                header_row = '<tr style="background:#065e56;color:#fff"><th style="padding:10px 12px;text-align:left">Model</th><th style="padding:10px 12px;text-align:left">RMSE</th><th style="padding:10px 12px;text-align:left">MAE</th><th style="padding:10px 12px;text-align:left">R²</th><th style="padding:10px 12px;text-align:left">CV Score</th></tr>'

            steps_html = ''.join([f'<li style="margin-bottom:4px">✓ {s}</li>' for s in preprocessing.get('steps_taken', [])])

            html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>ML Report — {report.get('target_column','')}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 32px; color: #1f2937; }}
  h1 {{ color: #065e56; border-bottom: 3px solid #0dbfa0; padding-bottom: 10px; }}
  h2 {{ color: #0a6b62; margin-top: 32px; font-size: 16px; border-left: 4px solid #0dbfa0; padding-left: 10px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:bold; }}
  .badge-blue {{ background:#dbeafe; color:#1e40af; }}
  .badge-purple {{ background:#ede9fe; color:#5b21b6; }}
  @media print {{ body {{ padding: 16px; }} }}
</style>
</head>
<body>
<h1>📊 Machine Learning Report</h1>

<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:20px 0;background:#f0fdfb;padding:16px;border-radius:8px">
  <div><p style="color:#6b7280;font-size:12px;margin:0">Target Column</p><p style="font-weight:bold;margin:4px 0">{report.get('target_column','—')}</p></div>
  <div><p style="color:#6b7280;font-size:12px;margin:0">Task Type</p><span class="badge {'badge-blue' if task_type=='classification' else 'badge-purple'}">{task_type}</span></div>
  <div><p style="color:#6b7280;font-size:12px;margin:0">Trained On</p><p style="font-weight:bold;margin:4px 0">{report.get('created_at','').strftime('%Y-%m-%d %H:%M') if report.get('created_at') else '—'}</p></div>
  <div><p style="color:#6b7280;font-size:12px;margin:0">Best Model</p><p style="font-weight:bold;color:#0dbfa0;margin:4px 0">{best_model.get('name','—')} ⭐</p></div>
  <div><p style="color:#6b7280;font-size:12px;margin:0">Features Used</p><p style="font-weight:bold;margin:4px 0">{len(preprocessing.get('final_features',[]))}</p></div>
  <div><p style="color:#6b7280;font-size:12px;margin:0">Training Time</p><p style="font-weight:bold;margin:4px 0">{report.get('training_duration',0):.1f}s</p></div>
</div>

<h2>Model Performance Comparison</h2>
<table><thead>{header_row}</thead><tbody>{metrics_rows}</tbody></table>

<h2>Top 10 Important Features</h2>
<table>
  <thead><tr style="background:#f3f4f6"><th style="padding:8px 12px;text-align:left">#</th><th style="padding:8px 12px;text-align:left">Feature</th><th style="padding:8px 12px;text-align:left">Score</th><th style="padding:8px 12px;text-align:left">Importance</th></tr></thead>
  <tbody>{feature_rows}</tbody>
</table>

<h2>Preprocessing Steps Applied</h2>
<ul style="font-size:14px;line-height:1.8;color:#374151">{steps_html}</ul>

<div style="margin-top:40px;padding-top:16px;border-top:1px solid #e5e7eb;font-size:12px;color:#9ca3af;text-align:center">
  Generated by DataFlow HUB Platform &nbsp;|&nbsp; {report.get('created_at','').strftime('%Y-%m-%d') if report.get('created_at') else ''}
</div>
</body></html>"""

            # Save and send as downloadable HTML file
            report_path = os.path.join(app.config['MODELS_FOLDER'], f'report_{report_id}.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html)

            return send_file(
                report_path,
                as_attachment=True,
                download_name=f'ml_report_{report.get("target_column","report")}.html',
                mimetype='text/html'
            )

    except Exception as e:
        app.logger.error(f"Report download error: {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/download_model/<report_id>/<model_name>')
@login_required
def download_model(report_id, model_name):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({"_id": ObjectId(report_id), "username": current_user.username})
            if not report:
                return jsonify({'error': 'Report not found'}), 404
            model_info = {
                'model_name': model_name, 'task_type': report['task_type'],
                'target_column': report['target_column'],
                'performance_metrics': report['results'].get(model_name, {}),
                'preprocessing_info': report['preprocessing_info']
            }
            model_file = f"model_{model_name}_{report_id}.json"
            model_path = os.path.join(app.config['MODELS_FOLDER'], model_file)
            with open(model_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            return send_file(model_path, as_attachment=True, download_name=model_file)
    except Exception as e:
        app.logger.error(f"Model download error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/download_model_pkl/<report_id>/<model_name>')
@login_required
def download_model_pkl(report_id, model_name):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({"_id": ObjectId(report_id), "username": current_user.username})
            if not report:
                return jsonify({'error': 'Report not found'}), 404
            model_dir = os.path.join(app.config['MODELS_FOLDER'], str(report_id))
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            task_type = report.get('task_type', 'classification')
            model_map = {
                'classification': {
                    'random_forest': RandomForestClassifier(),
                    'gradient_boosting': GradientBoostingClassifier(),
                    'xgboost': XGBClassifier(),
                    'lightgbm': LGBMClassifier(),
                    'catboost': CatBoostClassifier()
                },
                'regression': {
                    'random_forest': RandomForestRegressor(),
                    'gradient_boosting': GradientBoostingRegressor(),
                    'xgboost': XGBRegressor(),
                    'lightgbm': LGBMRegressor(),
                    'catboost': CatBoostRegressor()
                }
            }
            model = model_map.get(task_type, model_map['classification']).get(model_name)
            if model is None:
                return jsonify({'error': f'Unknown model: {model_name}'}), 400
            joblib.dump(model, model_path)
            return send_file(model_path, as_attachment=True,
                             download_name=f"{model_name}_{report_id}.pkl",
                             mimetype='application/octet-stream')
    except Exception as e:
        app.logger.error(f"Model download error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/ai/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.get_json()
        message = data.get('message')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        try:
            response = ai_assistant.generate_response(message)
            return jsonify({'success': True, 'response': response})
        except Exception as e:
            app.logger.error(f"AI Assistant error: {str(e)}")
            return jsonify({'success': False, 'error': 'Error generating response'}), 500
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='127.0.0.1', port=5000)