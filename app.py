import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Laminar Burning Velocity Prediction", page_icon="🔥", layout="wide")

# Title of the app
st.title("🔥 Laminar Burning Velocity Prediction")

# Global variables
MODEL_PATH = "lbv_model.pkl"
model = None
prediction_history = []

# List of hydrocarbons (adjust based on your model's training data)
hydrocarbons = ["Methane", "Ethane", "Propane", "Butane", "Ethylene", "Propylene"]

# Function to initialize the model
def initialize_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False

# Function to predict laminar burning velocity
def predict_lbv(hydrocarbon, temperature, equivalence_ratio, pressure):
    try:
        # Prepare input data for the model
        input_data = pd.DataFrame({
            'Hydrocarbon': [hydrocarbon],
            'Temperature': [temperature],
            'Equivalence_Ratio': [equivalence_ratio],
            'Pressure': [pressure]
        })
        # Predict using the model
        prediction = model.predict(input_data)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Main function to run the app
def main():
    model_path = "lbv_model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure lbv_model.pkl is in the repository.")
        return

    if not initialize_model():
        st.error("Failed to initialize model. Please check the data file.")
        return

    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    
    # Hydrocarbon selection
    hydrocarbon = st.sidebar.selectbox("Select Hydrocarbon", hydrocarbons)
    
    # Temperature input
    temperature = st.sidebar.slider("Initial Temperature (K)", min_value=300.0, max_value=750.0, value=400.0, step=0.1)
    
    # Equivalence ratio input
    equivalence_ratio = st.sidebar.slider("Equivalence Ratio (φ)", min_value=0.1, max_value=2.4, value=1.0, step=0.01)
    
    # Pressure input
    pressure = st.sidebar.slider("Pressure (atm)", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
    
    # Unit conversion option
    convert_to_ms = st.sidebar.checkbox("Convert to m/s", value=False)
    
    # Prediction button
    if st.sidebar.button("🔥 Predict LBV"):
        with st.spinner("Calculating LBV..."):
            lbv = predict_lbv(hydrocarbon, temperature, equivalence_ratio, pressure)
            if lbv is not None:
                # Convert units if requested
                if convert_to_ms:
                    lbv = lbv / 100  # Convert cm/s to m/s
                    unit = "m/s"
                else:
                    unit = "cm/s"
                st.success(f"Predicted Laminar Burning Velocity: **{lbv:.2f} {unit}**")
                
                # Add to prediction history
                prediction_history.append({
                    "Hydrocarbon": hydrocarbon,
                    "Temperature": temperature,
                    "Equivalence_Ratio": equivalence_ratio,
                    "Pressure": pressure,
                    "LBV": lbv,
                    "Unit": unit
                })

    # Sidebar for history toggle
    show_history = st.sidebar.checkbox("Show Prediction History", value=True)

    # Display prediction history
    if show_history and prediction_history:
        st.sidebar.subheader("📜 Prediction History")
        history_df = pd.DataFrame(prediction_history)
        st.sidebar.table(history_df)
        
        # Clear history button
        if st.sidebar.button("🗑️ Clear History"):
            prediction_history.clear()
            st.sidebar.success("History cleared!")

    # Sidebar sections for additional information
    with st.sidebar.expander("📘 About"):
        st.write("""
        This app predicts the Laminar Burning Velocity (LBV) of various hydrocarbons using a machine learning model.
        - **Algorithm**: Random Forest Regression
        - **Trained on**: Experimental data for hydrocarbons
        - **Input Ranges**:
          - Temperature: 300–750 K
          - Equivalence Ratio: 0.1–2.4
          - Pressure: 1–10 atm
        """)

    with st.sidebar.expander("👨‍💻 Developers"):
        st.write("""
        - Shekhar Sonar
        - Sujal Fiske
        - Karan Shinde
        """)

    with st.sidebar.expander("🎓 Mentor / Project Guide"):
        st.write("Shawnam, IIT Bombay")

    with st.sidebar.expander("📚 Resources"):
        st.write("""
        - [Paper on Laminar Burning Velocity](https://example.com)
        - [Dataset Source](https://example.com)
        """)

if __name__ == "__main__":
    main()
