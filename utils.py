import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Dict, Any

class LBVPredictor:
    def __init__(self, model_path='lbv_model.pkl'):
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_hydrocarbon_options(self):
        """Get available hydrocarbon options"""
        if self.label_encoder is not None:
            return list(self.label_encoder.classes_)
        return []
    
    def predict_lbv(self, hydrocarbon, temperature, equiv_ratio, pressure):
        """Predict LBV for given inputs"""
        try:
            # Encode hydrocarbon
            hydrocarbon_encoded = self.label_encoder.transform([hydrocarbon])[0]
            
            # Prepare input features
            features = np.array([[hydrocarbon_encoded, temperature, equiv_ratio, pressure]])
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def get_hydrocarbon_ranges(self, hydrocarbon):
        """Get valid ranges for a specific hydrocarbon from the dataset"""
        try:
            import pandas as pd
            df = pd.read_csv('data/final_dataset_after_preprocessing (2.0)_1750055394475.csv')
            hydrocarbon_data = df[df['Hydrocarbon'] == hydrocarbon]
            
            if len(hydrocarbon_data) == 0:
                # Return default ranges if hydrocarbon not found
                return {
                    'temperature': {'min': 300, 'max': 750},
                    'equiv_ratio': {'min': 0.1, 'max': 2.4},
                    'pressure': {'min': 1.0, 'max': 10.0}
                }
            
            return {
                'temperature': {
                    'min': float(hydrocarbon_data['Ti (K)'].min()),
                    'max': float(hydrocarbon_data['Ti (K)'].max())
                },
                'equiv_ratio': {
                    'min': float(hydrocarbon_data['equivalent ratio'].min()),
                    'max': float(hydrocarbon_data['equivalent ratio'].max())
                },
                'pressure': {
                    'min': float(hydrocarbon_data['Pressure (atm)'].min()),
                    'max': float(hydrocarbon_data['Pressure (atm)'].max())
                }
            }
        except Exception as e:
            print(f"Error getting ranges: {e}")
            return {
                'temperature': {'min': 300, 'max': 750},
                'equiv_ratio': {'min': 0.1, 'max': 2.4},
                'pressure': {'min': 1.0, 'max': 10.0}
            }

    def validate_inputs(self, hydrocarbon, temperature, equiv_ratio, pressure):
        """Validate input parameters based on hydrocarbon-specific ranges"""
        errors = []
        
        # Check hydrocarbon
        if hydrocarbon not in self.get_hydrocarbon_options():
            errors.append(f"Invalid hydrocarbon type. Available options: {', '.join(self.get_hydrocarbon_options())}")
            return errors
        
        # Get hydrocarbon-specific ranges
        ranges = self.get_hydrocarbon_ranges(hydrocarbon)
        
        # Check temperature range
        temp_min, temp_max = ranges['temperature']['min'], ranges['temperature']['max']
        if not (temp_min <= temperature <= temp_max):
            errors.append(f"Temperature must be between {temp_min:.1f} and {temp_max:.1f} K for {hydrocarbon}")
        
        # Check equivalence ratio range
        eq_min, eq_max = ranges['equiv_ratio']['min'], ranges['equiv_ratio']['max']
        if not (eq_min <= equiv_ratio <= eq_max):
            errors.append(f"Equivalence ratio must be between {eq_min:.2f} and {eq_max:.2f} for {hydrocarbon}")
        
        # Check pressure range
        press_min, press_max = ranges['pressure']['min'], ranges['pressure']['max']
        if not (press_min <= pressure <= press_max):
            errors.append(f"Pressure must be between {press_min:.1f} and {press_max:.1f} atm for {hydrocarbon}")
        
        return errors

def convert_units(value, from_unit, to_unit):
    """Convert between cm/s and m/s"""
    if from_unit == "cm/s" and to_unit == "m/s":
        return value / 100
    elif from_unit == "m/s" and to_unit == "cm/s":
        return value * 100
    else:
        return value

def get_input_ranges():
    """Get the valid input ranges for the application"""
    return {
        'temperature': {'min': 300, 'max': 750, 'unit': 'K'},
        'equiv_ratio': {'min': 0.1, 'max': 2.4, 'unit': ''},
        'pressure': {'min': 1.0, 'max': 10.0, 'unit': 'atm'}
    }

def format_prediction_result(lbv_value, unit="cm/s", confidence=None):
    """Format the prediction result for display"""
    if lbv_value is None:
        return "Prediction failed"
    
    formatted_value = f"{lbv_value:.2f} {unit}"
    
    if confidence is not None:
        return f"{formatted_value} (confidence: {confidence:.1f}%)"
    
    return formatted_value

def get_model_info():
    """Get information about the model"""
    return {
        'algorithm': 'Random Forest Regression',
        'features': ['Hydrocarbon Type', 'Temperature (K)', 'Equivalence Ratio', 'Pressure (atm)'],
        'target': 'Laminar Burning Velocity (cm/s)',
        'dataset_size': '6000+ data points',
        'optimization': 'Hyperparameter tuning with GridSearchCV'
    }

def initialize_model():
    """Initialize the model and train if not exists"""
    import os
    
    model_path = 'lbv_model.pkl'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("Model not found. Please run training first.")
        return False
    else:
        print("Model found and ready to use.")
        return True
