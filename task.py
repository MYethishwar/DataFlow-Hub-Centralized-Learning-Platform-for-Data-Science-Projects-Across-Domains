from celery import Celery
from config import Config

celery_app = Celery('tasks', broker=Config.CELERY_BROKER_URL, backend=Config.CELERY_RESULT_BACKEND)
celery_app.conf.update(task_serializer='json', accept_content=['json'], result_serializer='json')
