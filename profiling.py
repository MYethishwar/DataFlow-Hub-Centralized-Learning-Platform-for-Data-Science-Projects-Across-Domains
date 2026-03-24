import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

class DataProfiler:
    def __init__(self, df):
        self.df = df
        self.profile = {}
        
    def generate_profile(self):
        """Generate comprehensive data profile"""
        self.profile = {
            'basic_stats': self._get_basic_stats(),
            'column_stats': self._get_column_stats(),
            'correlations': self._get_correlations(),
            'missing_data': self._get_missing_data(),
            'distributions': self._get_distributions()
        }
        return self.profile
    
    def _get_basic_stats(self):
        """Get basic dataset statistics"""
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'total_cells': len(self.df) * len(self.df.columns),
            'duplicate_rows': len(self.df[self.df.duplicated()]),
            'total_memory': self.df.memory_usage(deep=True).sum()
        }
    
    def _get_column_stats(self):
        """Get detailed statistics for each column"""
        stats = {}
        for col in self.df.columns:
            col_type = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            missing_count = self.df[col].isnull().sum()
            
            col_stats = {
                'type': col_type,
                'unique_count': int(unique_count),
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_count / len(self.df) * 100)
            }
            
            if np.issubdtype(self.df[col].dtype, np.number):
                desc = self.df[col].describe()
                col_stats.update({
                    'mean': float(desc['mean']),
                    'std': float(desc['std']),
                    'min': float(desc['min']),
                    'max': float(desc['max']),
                    'quartiles': {
                        '25%': float(desc['25%']),
                        '50%': float(desc['50%']),
                        '75%': float(desc['75%'])
                    }
                })
            elif self.df[col].dtype == 'object':
                value_counts = self.df[col].value_counts()
                col_stats.update({
                    'top_values': value_counts.head(10).to_dict()
                })
                
            stats[col] = col_stats
        return stats
    
    def _get_correlations(self):
        """Get correlation matrix for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().round(4)
            return corr_matrix.to_dict()
        return {}
    
    def _get_missing_data(self):
        """Analyze missing data"""
        missing = self.df.isnull().sum()
        return {
            'total_missing': int(missing.sum()),
            'missing_by_column': missing.to_dict()
        }
    
    def _get_distributions(self):
        """Get distribution information for numeric columns"""
        distributions = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            hist, bins = np.histogram(self.df[col].dropna(), bins='auto')
            distributions[col] = {
                'histogram': hist.tolist(),
                'bins': bins.tolist()
            }
        return distributions
    
    def generate_html_report(self):
        """Generate an HTML report with interactive visualizations"""
        profile = self.generate_profile()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Profile Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        </head>
        <body class="bg-gray-100 p-8">
            <div class="max-w-7xl mx-auto">
                <h1 class="text-3xl font-bold mb-8">Data Profile Report</h1>
                
                <!-- Basic Statistics -->
                <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                    <h2 class="text-xl font-bold mb-4">Basic Statistics</h2>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                            <p class="text-gray-600">Rows</p>
                            <p class="text-2xl font-bold">{profile['basic_stats']['rows']:,}</p>
                        </div>
                        <div>
                            <p class="text-gray-600">Columns</p>
                            <p class="text-2xl font-bold">{profile['basic_stats']['columns']:,}</p>
                        </div>
                        <div>
                            <p class="text-gray-600">Duplicate Rows</p>
                            <p class="text-2xl font-bold">{profile['basic_stats']['duplicate_rows']:,}</p>
                        </div>
                        <div>
                            <p class="text-gray-600">Memory Usage</p>
                            <p class="text-2xl font-bold">{profile['basic_stats']['total_memory'] / 1024 / 1024:.2f} MB</p>
                        </div>
                    </div>
                </div>
                
                <!-- Column Statistics -->
                <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                    <h2 class="text-xl font-bold mb-4">Column Statistics</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {self._generate_column_cards()}
                    </div>
                </div>
                
                <!-- Correlation Matrix -->
                <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                    <h2 class="text-xl font-bold mb-4">Correlation Matrix</h2>
                    <div id="correlationMatrix"></div>
                </div>
                
                <!-- Missing Data -->
                <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                    <h2 class="text-xl font-bold mb-4">Missing Data</h2>
                    <div id="missingData"></div>
                </div>
            </div>
            
            <script>
                {self._generate_plotly_scripts()}
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_column_cards(self):
        """Generate HTML cards for column statistics"""
        cards = []
        for col, stats in self.profile['column_stats'].items():
            card = f"""
            <div class="bg-gray-50 rounded p-4">
                <h3 class="font-bold text-lg mb-2">{col}</h3>
                <p><span class="font-medium">Type:</span> {stats['type']}</p>
                <p><span class="font-medium">Unique Values:</span> {stats['unique_count']:,}</p>
                <p><span class="font-medium">Missing:</span> {stats['missing_count']:,} ({stats['missing_percentage']:.2f}%)</p>
            """
            
            if 'mean' in stats:
                card += f"""
                <p><span class="font-medium">Mean:</span> {stats['mean']:.2f}</p>
                <p><span class="font-medium">Std:</span> {stats['std']:.2f}</p>
                <p><span class="font-medium">Range:</span> [{stats['min']:.2f}, {stats['max']:.2f}]</p>
                """
            
            card += "</div>"
            cards.append(card)
        
        return "\n".join(cards)
    
    def _generate_plotly_scripts(self):
        scripts = []
    # Correlation Matrix
        if self.profile['correlations']:
            corr_keys = list(self.profile['correlations'].keys())
            corr_values = [[self.profile['correlations'][i][j] for j in corr_keys] 
                        for i in corr_keys]
        
            corr_data = {
                'z': corr_values,
                'x': corr_keys,
                'y': corr_keys,
                'type': 'heatmap',
                'colorscale': 'RdBu'
            }
            
            scripts.append(f"""
                Plotly.newPlot('correlationMatrix', [{json.dumps(corr_data)}], {{
                    title: 'Feature Correlation Matrix',
                    width: document.getElementById('correlationMatrix').offsetWidth,
                    height: 500,
                    margin: {{
                        l: 100,
                        r: 50,
                        b: 100,
                        t: 50,
                        pad: 4
                    }}
                }});
            """)
        
        # Missing Data
        missing_data = self.profile['missing_data']['missing_by_column']
        missing_plot_data = {
            'x': list(missing_data.keys()),
            'y': list(missing_data.values()),
            'type': 'bar'
        }
        
        scripts.append(f"""
            Plotly.newPlot('missingData', [{json.dumps(missing_plot_data)}], {{
                title: 'Missing Values by Column',
                width: document.getElementById('missingData').offsetWidth,
                height: 400,
                xaxis: {{
                    tickangle: -45
                }},
                margin: {{
                    l: 50,
                    r: 50,
                    b: 100,
                    t: 50,
                    pad: 4
                }}
            }});
            
            // Add resize handlers
            window.addEventListener('resize', function() {{
                Plotly.relayout('correlationMatrix', {{
                    width: document.getElementById('correlationMatrix').offsetWidth
                }});
                Plotly.relayout('missingData', {{
                    width: document.getElementById('missingData').offsetWidth
                }});
            }});
        """)
        
        return "\n".join(scripts)