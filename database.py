from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import mysql.connector
from mysql.connector import Error as MySQLError, pooling
from contextlib import contextmanager
import logging
import os
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DatabaseManager:
    def __init__(self):
        self.mongo_client = None
        self.mysql_pool = None
        self.logger = logging.getLogger(__name__)
        self._init_connections()

    def _init_connections(self):
        try:
            self.mongo_client = MongoClient(
                Config.MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            self.mongo_client.admin.command('ping')
            self.logger.info("MongoDB connection successful")

            db = self.mongo_client[Config.MONGODB_DB]
            for collection in ['uploads', 'model_reports']:
                if collection not in db.list_collection_names():
                    db.create_collection(collection)

            try:
                dbconfig = {
                    'user': Config.MYSQL_CONFIG['user'],
                    'password': Config.MYSQL_CONFIG['password'],
                    'host': Config.MYSQL_CONFIG['host'],
                    'database': Config.MYSQL_CONFIG['database'],
                    'raise_on_warnings': True,
                    'auth_plugin': 'mysql_native_password',
                    'pool_name': 'mypool',
                    'pool_size': 5
                }
                self.mysql_pool = mysql.connector.pooling.MySQLConnectionPool(**dbconfig)
                self.logger.info("MySQL connection pool initialized")
            except Exception as mysql_error:
                self.logger.warning(f"MySQL unavailable: {str(mysql_error)}")
                self.mysql_pool = None

        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            if self.mongo_client:
                self.mongo_client.close()
            raise

    @contextmanager
    def get_mongo_connection(self):
        try:
            if not self.mongo_client:
                self._init_connections()
            db = self.mongo_client[Config.MONGODB_DB]
            yield db
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.error(f"MongoDB connection error: {e}")
            self.mongo_client = None
            raise ConnectionError(f"MongoDB connection failed: {str(e)}")
        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            raise

    def get_mysql_connection(self):
        if not self.mysql_pool:
            raise RuntimeError("MySQL is not available.")
        return self.mysql_pool.get_connection()

    def initialize_database(self):
        if not self.mysql_pool:
            self.logger.warning("MySQL not available, skipping table initialization")
            return True
        try:
            conn = self.mysql_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = %s AND table_name = 'users'
            """, (Config.MYSQL_CONFIG['database'],))
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    CREATE TABLE users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        password VARCHAR(255) NOT NULL,
                        email VARCHAR(120) UNIQUE NOT NULL,
                        name VARCHAR(100),
                        organization VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP NULL,
                        active BOOLEAN DEFAULT TRUE
                    )
                """)
                conn.commit()
                self.logger.info("Users table created successfully")
            else:
                self.logger.info("Users table already exists")
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            return False

    def close_connections(self):
        try:
            if self.mongo_client:
                self.mongo_client.close()
                self.mongo_client = None
            self.logger.info("All database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")

db_manager = DatabaseManager()

if not db_manager.initialize_database():
    logging.warning("Database initialization incomplete - some features may not work")