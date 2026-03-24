# model_deployment.py
import joblib
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json

class ModelDeployment:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.deployed_models = {}
        
    def deploy_model(self, model_id: str, model_info: Dict) -> Dict:
        """Deploy a trained model"""
        try:
            # Load model
            model = joblib.load(os.path.join(self.model_path, f"{model_id}.joblib"))
            
            # Store model info
            self.deployed_models[model_id] = {
                'model': model,
                'info': model_info,
                'deployment_time': datetime.utcnow(),
                'status': 'active'
            }
            
            return {
                'success': True,
                'model_id': model_id,
                'deployment_time': self.deployed_models[model_id]['deployment_time']
            }
        except Exception as e:
            raise Exception(f"Model deployment failed: {str(e)}")
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a deployed model"""
        if model_id not in self.deployed_models:
            raise Exception("Model not found")
            
        return {
            'model_id': model_id,
            'info': self.deployed_models[model_id]['info'],
            'deployment_time': self.deployed_models[model_id]['deployment_time'],
            'status': self.deployed_models[model_id]['status']
        }
    
    def predict(self, model_id: str, data: Dict) -> Dict:
        """Make predictions using deployed model"""
        if model_id not in self.deployed_models:
            raise Exception("Model not found")
            
        try:
            model = self.deployed_models[model_id]['model']
            prediction = model.predict([list(data.values())])
            
            return {
                'success': True,
                'prediction': prediction.tolist()[0],
                'model_id': model_id,
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")