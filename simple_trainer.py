import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

def train_simple_model():
    """Train a simple but effective Random Forest model quickly"""
    print("Loading dataset...")
    
    # Load data
    df = pd.read_csv('data/final_dataset_after_preprocessing (2.0)_1750055394475.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Clean and prepare data
    df = df.dropna()
    
    # Encode hydrocarbon types
    label_encoder = LabelEncoder()
    df['Hydrocarbon_encoded'] = label_encoder.fit_transform(df['Hydrocarbon'])
    
    # Prepare features and target
    X = df[['Hydrocarbon_encoded', 'Ti (K)', 'equivalent ratio', 'Pressure (atm)']]
    y = df['LBV (cm/s)']
    
    print(f"Training data: {X.shape[0]} samples")
    print(f"Unique hydrocarbons: {len(label_encoder.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train optimized model for better accuracy
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=300,  # Increased for better accuracy
        max_depth=25,      # Deeper trees for complex patterns
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Model R² Score: {r2:.4f}")
    print(f"Model RMSE: {rmse:.2f} cm/s")
    
    # Save model
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': ['Hydrocarbon_encoded', 'Ti (K)', 'equivalent ratio', 'Pressure (atm)']
    }
    
    with open('lbv_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully!")
    return r2, rmse

if __name__ == "__main__":
    train_simple_model()