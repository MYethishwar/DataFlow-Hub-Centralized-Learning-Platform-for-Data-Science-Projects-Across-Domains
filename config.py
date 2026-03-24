import os
from datetime import timedelta

class Config:
    # Debug mode
    DEBUG = True
    
    # File configurations
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    ALLOWED_EXTENSIONS = {'csv'}
    
    # Security configurations
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    CSRF_ENABLED = True
    WTF_CSRF_SECRET_KEY = os.environ.get('WTF_CSRF_SECRET_KEY') or os.urandom(24)
    
    # Session Configuration
    SESSION_TYPE = 'filesystem'
    SESSION_COOKIE_NAME = 'automl_session'
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    
    # MongoDB Configuration
    MONGODB_URI = "mongodb://localhost:27017/"
    MONGODB_DB = "automl"
    MONGODB_SETTINGS = {
        'host': MONGODB_URI,
        'db': MONGODB_DB,
        'serverSelectionTimeoutMS': 5000,
        'connectTimeoutMS': 5000
    }
    
    # MySQL Configuration
    # MySQL Configuration
    MYSQL_CONFIG = {
        'user': os.environ.get('MYSQL_USER', 'root'),
        'password': os.environ.get('MYSQL_PASSWORD', ''),
        'host': os.environ.get('MYSQL_HOST', '127.0.0.1'),
        'database': os.environ.get('MYSQL_DB', 'automl_db'),
        'auth_plugin': 'mysql_native_password',
        'raise_on_warnings': True
    }
    
    # MySQL Pool Config
    MYSQL_POOL_CONFIG = {
        'pool_name': 'mypool',
        'pool_size': 5,
        'pool_reset_session': True
    }
    
    # ML Configuration
    ML_CONFIG = {
        'max_training_time': 3600,  # Maximum training time in seconds
        'supported_algorithms': {
            'classification': [
                'random_forest',
                'xgboost',
                'lightgbm'
            ],
            'regression': [
                'random_forest',
                'xgboost',
                'lightgbm'
            ]
        },
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5
    }
    
    # Logging Configuration
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'logs/app.log'
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    @staticmethod
    def init_app(app):
        """Initialize application configuration"""
        # Create required directories if they don't exist
        for folder in [Config.UPLOAD_FOLDER, Config.MODELS_FOLDER, 'logs']:
            folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                os.chmod(folder_path, 0o755)  # Set proper permissions
        
        # Set up basic logging
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        return app

    @staticmethod
    def get_mysql_uri():
        """Get MySQL URI from configuration"""
        return (f"mysql://{Config.MYSQL_CONFIG['user']}:{Config.MYSQL_CONFIG['password']}@"
                f"{Config.MYSQL_CONFIG['host']}/{Config.MYSQL_CONFIG['database']}")

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = False
    TESTING = True
    WTF_CSRF_ENABLED = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True  # Requires HTTPS
    
    # Override with production database settings
    MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
    MYSQL_CONFIG = {
        'user': os.environ.get('MYSQL_USER', 'root'),
        'port': int(os.environ.get('MYSQL_PORT', 3307)),
        'password': os.environ.get('MYSQL_PASSWORD', ''),
        'host': os.environ.get('MYSQL_HOST', 'localhost'),
        'database': os.environ.get('MYSQL_DB', 'automl_prod')
    }

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}