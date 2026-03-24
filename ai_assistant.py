from groq import Groq
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIAssistant:
    def __init__(self):
        """Initialize AI Assistant using Groq API."""
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model = "llama-3.3-70b-versatile"

    def generate_response(self, prompt: str, max_tokens: int = 100):
        """Generate response using Groq API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Groq API error: {str(e)}")
            return f"Error generating response: {str(e)}"

    async def analyze_data(self, df, target_column: str, task_type: str):
        """Perform AI-based dataset analysis"""
        try:
            dataset_info = {
                'basic_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'task_type': task_type,
                    'target_column': target_column
                },
                'columns': {
                    col: {
                        'type': str(df[col].dtype),
                        'unique_values': int(df[col].nunique()),
                        'missing_values': int(df[col].isnull().sum())
                    }
                    for col in df.columns
                }
            }

            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                stats = df[col].describe()
                dataset_info['columns'][col].update({
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max'])
                })

            prompt = f"""
            Analyze this dataset for machine learning:

            Basic Information:
            - Task Type: {task_type}
            - Target Column: {target_column}
            - Number of Rows: {dataset_info['basic_info']['rows']}
            - Number of Columns: {dataset_info['basic_info']['columns']}

            Column Details:
            {json.dumps(dataset_info['columns'], indent=2)}

            Provide:
            1. Data Quality Assessment
            2. Feature Engineering Suggestions
            3. Preprocessing Recommendations
            4. Modeling Approach
            5. Potential Challenges

            Format the response with markdown headings and bullet points.
            """
            response = self.generate_response(prompt)
            return {'success': True, 'analysis': response}

        except Exception as e:
            logging.error(f"Data analysis error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def get_model_recommendations(self, df: pd.DataFrame, task_type: str, target_column: str) -> Dict:
        """Get AI recommendations for model selection and hyperparameters"""
        try:
            data_summary = {
                'shape': list(df.shape),
                'numeric_features': len(df.select_dtypes(include=['int64', 'float64']).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns),
                'missing_values': int(df.isnull().sum().sum()),
                'target_distribution': df[target_column].value_counts().to_dict() if task_type == 'classification' else {
                    'mean': float(df[target_column].mean()),
                    'std': float(df[target_column].std())
                }
            }

            prompt = f"""
            As an ML expert, recommend the best 5 models and their hyperparameters for this dataset:

            Task Type: {task_type}
            Dataset Info: {json.dumps(data_summary, indent=2)}

            For each model provide:
            1. Model name and rationale
            2. Optimal hyperparameters
            3. Expected performance characteristics
            4. Potential challenges

            Return as a valid JSON object only, no extra text.
            """
            response = self.generate_response(prompt)
            try:
                recommendations = json.loads(response)
            except:
                recommendations = {"raw": response}
            return {'success': True, 'recommendations': recommendations}

        except Exception as e:
            logging.error(f"Model recommendation error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def get_model_insights(self, results: Dict, task_type: str) -> Dict:
        """Get insights about model performance"""
        try:
            prompt = f"""
            Analyze these model results for a {task_type} task:
            {json.dumps(results, indent=2)}

            Provide insights about:
            - Model performance comparison
            - Areas for improvement
            - Feature importance analysis
            - Optimization suggestions

            Format the response with markdown headings and bullet points.
            """
            response = self.generate_response(prompt)
            return {'success': True, 'insights': response}

        except Exception as e:
            logging.error(f"Model insights error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def get_feature_recommendations(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Get feature engineering recommendations"""
        try:
            correlations = {}
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                correlations = corr_matrix.to_dict()

            df_info = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'correlations': correlations,
                'target_column': target_column
            }

            prompt = f"""
            Based on this dataset information:
            {json.dumps(df_info, indent=2)}

            Suggest:
            - Feature transformations
            - Feature interactions
            - Feature selection approaches
            - New features that could be created

            Format the response with markdown headings and bullet points.
            """
            response = self.generate_response(prompt)
            return {'success': True, 'recommendations': response}

        except Exception as e:
            logging.error(f"Feature recommendations error: {str(e)}")
            return {'success': False, 'error': str(e)}