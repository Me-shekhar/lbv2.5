import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os

class LBVModelTrainer:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = ['Hydrocarbon_encoded', 'Ti (K)', 'equivalent ratio', 'Pressure (atm)']
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the dataset"""
        try:
            df = pd.read_csv(csv_path)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Check for missing values
            print(f"Missing values:\n{df.isnull().sum()}")
            
            # Remove rows with missing values
            df = df.dropna()
            
            # Encode hydrocarbon types
            self.label_encoder = LabelEncoder()
            df['Hydrocarbon_encoded'] = self.label_encoder.fit_transform(df['Hydrocarbon'])
            
            # Prepare features and target
            X = df[['Hydrocarbon_encoded', 'Ti (K)', 'equivalent ratio', 'Pressure (atm)']]
            y = df['LBV (cm/s)']
            
            print(f"Unique hydrocarbon types: {list(self.label_encoder.classes_)}")
            print(f"Feature ranges:")
            print(f"Temperature: {X['Ti (K)'].min():.1f} - {X['Ti (K)'].max():.1f} K")
            print(f"Equivalence ratio: {X['equivalent ratio'].min():.2f} - {X['equivalent ratio'].max():.2f}")
            print(f"Pressure: {X['Pressure (atm)'].min():.1f} - {X['Pressure (atm)'].max():.1f} atm")
            print(f"LBV range: {y.min():.2f} - {y.max():.2f} cm/s")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def train_model(self, X, y):
        """Train Random Forest model with hyperparameter tuning"""
        print("Starting model training...")
        
        # Split the data without stratification to avoid issues with small classes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Use a simpler approach with good default parameters for faster training
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f} cm/s")
        print(f"MAE: {mae:.4f} cm/s")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R² Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        print(feature_importance)
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
    
    def save_model(self, model_path='lbv_model.pkl'):
        """Save the trained model and label encoder"""
        try:
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path='lbv_model.pkl'):
        """Load a pre-trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            
            print(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Main function to train and save the model"""
    trainer = LBVModelTrainer()
    
    # Load and preprocess data
    csv_path = 'data/final_dataset_after_preprocessing (2.0)_1750055394475.csv'
    X, y = trainer.load_and_preprocess_data(csv_path)
    
    if X is not None and y is not None:
        # Train model
        metrics = trainer.train_model(X, y)
        
        # Save model
        trainer.save_model('lbv_model.pkl')
        
        print("\nModel training completed successfully!")
        return metrics
    else:
        print("Failed to load and preprocess data!")
        return None

if __name__ == "__main__":
    main()
