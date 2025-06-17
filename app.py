import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils import LBVPredictor, convert_units, get_input_ranges, format_prediction_result, get_model_info, initialize_model
import os

# Page configuration
st.set_page_config(
    page_title="🔥 Laminar Burning Velocity Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern teal/cyan theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .input-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid rgba(6, 182, 212, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .result-container {
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .result-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
        border: 1px solid rgba(6, 182, 212, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-title {
        color: #e2e8f0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .metric-value {
        color: #06b6d4;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Remove dots from number inputs */
    .stNumberInput > div > div > input::-webkit-outer-spin-button,
    .stNumberInput > div > div > input::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    
    .stNumberInput > div > div > input[type=number] {
        -moz-appearance: textfield;
    }
    
    /* Input styling */
    .stSelectbox > div > div > div {
        background-color: #1e293b;
        border: 1px solid #475569;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        background-color: #1e293b;
        border: 1px solid #475569;
        color: white;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: bold;
        font-size: 1.1rem;
        width: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d9488 0%, #0ea5e9 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
    }
    
    
    /* Toggle switch custom styling */
    .stToggle > label > div:first-child {
        background-color: #0f766e !important;
        border-color: #0f766e !important;
    }
    .stToggle > label > div:first-child:hover {
        background-color: #0d9488 !important;
    }

/* Hide captions initially */
    .caption-hidden {
        display: none;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize model
    if not initialize_model():
        st.error("Failed to initialize model. Please check the data file.")
        return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔥 Laminar Burning Velocity Prediction</h1>
        <p>Predict the Laminar Burning Velocity (LBV) for different hydrocarbons based on temperature, equivalence ratio, and pressure.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load predictor
    try:
        predictor = LBVPredictor()
        hydrocarbon_options = predictor.get_hydrocarbon_options()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Sidebar with menu layout
    with st.sidebar:
        # Menu header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%); 
                    padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0;">📋 Menu</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # History toggle switch
        if 'show_history' not in st.session_state:
            st.session_state.show_history = True
            
        show_history = st.toggle("Show Prediction History", value=st.session_state.show_history)
        st.session_state.show_history = show_history
        
        st.markdown("---")
        
        # About section (Model Performance)
        with st.expander("📘 About", expanded=False):
            st.write("**Algorithm:** Random Forest Regression")
            st.write("**Dataset Size:** 6000+ data points")
            st.write("**R² Score:** 0.903")
            st.write("**RMSE:** 12.73 cm/s")
            st.write("**Features:** 4 input parameters")
            st.write("**Hydrocarbon Types:** 46 different fuels")
            st.write("**Training Data:** Real experimental measurements")
            st.write("**Validation:** Cross-validated for robustness")
        
        # Developers section
        with st.expander("👨‍💻 Developers", expanded=False):
            st.subheader("Development Team")
            st.write("**Final Year Mechanical Engineering Students**")
            st.write("Pimpri Chinchwad College of Engineering, Ravet")
            st.write("Pune, Maharashtra")
            
            st.markdown("---")
            st.subheader("Team Members")
            
            st.write("**Shekhar Sonar**")
            st.write("📧 shekharsonar641@gmail.com")
            
            st.write("**Sujal Fiske**")  
            st.write("📧 sujal.fiske_mech24@pccoer.in")
            
            st.write("**Karan Shinde**")
            st.write("📧 karan.shinde_mech23@pccoer.in")
        
        # Mentor/Project Guide section
        with st.expander("🎓 Mentor / Project Guide", expanded=False):
            st.subheader("Project Supervisor")
            st.write("**Shawnam**")
            st.write("📧 shawnam.ae111@gmail.com")
            st.write("🏢 Department of Aerospace Engineering")
            st.write("🏛️ Indian Institute of Technology Bombay")
            st.write("📍 Mumbai 400076, India")
        
        # Resources section (References)
        with st.expander("📚 Resources", expanded=False):
            st.subheader("Textbook:")
            st.write("Turns, S. R., 2020, *An Introduction to Combustion: Concepts and Applications*, McGraw-Hill Education.")
            
            st.subheader("Key Research Papers:")
            st.write("1. **Laminar burning velocity measurements of ethyl valerate-air flames at elevated temperatures with mechanism modifications**")
            st.write("*Authors: Shawnam, Rohit Kumar, E.V. Jithin, Ratna Kishore Velamati, Sudarshan Kumar*")
            
            st.write("2. **Experimental measurements of laminar burning velocity of propane-air flames at higher pressure and temperature conditions**")
            st.write("*Authors: Vijay Shinde, Amardeep Fulzele, Sudarshan Kumar*")
            
            st.write("3. **Laminar burning velocity measurements of NH3/N2/Ar mixtures at elevated temperatures**")
            st.write("*Authors: Shawnam, Pragya Berwal, Muskaan Singh, Sudarshan Kumar*")
            
            st.write("4. **Combustion of N-Decane/air Mixtures To Investigate Laminar Burning Velocity Measurements At Elevated Temperatures**")
            st.write("*Authors: Rohit Kumar, Ratna Kishore Velamati, Sudarshan Kumar*")
            
            st.write("**Additional Sources:**")
            st.write("• 30 more research papers on laminar burning velocity measurements")
    
    # Main content - single column layout
        st.subheader("🔧 Input Parameters")
    
    # First row - Hydrocarbon selection (full width)
    hydrocarbon = st.selectbox(
        "Hydrocarbon",
        options=hydrocarbon_options,
        index=0 if hydrocarbon_options else 0,
        help="Select the type of hydrocarbon fuel"
    )
    
    # Get dynamic ranges for selected hydrocarbon
    if hydrocarbon:
        ranges = predictor.get_hydrocarbon_ranges(hydrocarbon)
        temp_min, temp_max = ranges['temperature']['min'], ranges['temperature']['max']
        eq_min, eq_max = ranges['equiv_ratio']['min'], ranges['equiv_ratio']['max']  
        press_min, press_max = ranges['pressure']['min'], ranges['pressure']['max']
    else:
        temp_min, temp_max = 300.0, 750.0
        eq_min, eq_max = 0.1, 2.4
        press_min, press_max = 1.0, 10.0
    
    # Second row - Three equal columns for parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temperature = st.number_input(
            "Initial Temperature (K)",
            min_value=temp_min,
            max_value=temp_max,
            value=min(450.0, temp_max) if temp_max >= 450.0 else temp_min,
            step=1.0,
            help=f"Temperature range for {hydrocarbon}: {temp_min:.1f}-{temp_max:.1f} K"
        )
        st.caption(f"Valid range: {temp_min:.1f} - {temp_max:.1f} K")
    
    with col2:
        equiv_ratio = st.number_input(
            "Equivalence Ratio (φ)",
            min_value=eq_min,
            max_value=eq_max,
            value=min(0.7, eq_max) if eq_max >= 0.7 else eq_min,
            step=0.01,
            help=f"Equivalence ratio range for {hydrocarbon}: {eq_min:.2f}-{eq_max:.2f}"
        )
        st.caption(f"Valid range: {eq_min:.2f} - {eq_max:.2f}")
    
    with col3:
        pressure = st.number_input(
            "Pressure (atm)",
            min_value=press_min,
            max_value=press_max,
            value=min(1.0, press_max) if press_max >= 1.0 else press_min,
            step=0.1,
            help=f"Pressure range for {hydrocarbon}: {press_min:.1f}-{press_max:.1f} atm"
        )
        st.caption(f"Valid range: {press_min:.1f} - {press_max:.1f} atm")
    
    # Third row - Controls
    col4, col5, col6 = st.columns([1, 1, 2])
    
    with col4:
        # Unit conversion toggle
        convert_to_ms = st.toggle("Convert to m/s", value=False, help="Toggle between cm/s and m/s units")
    
    with col6:
        # Predict button
        predict_clicked = st.button("🔥 Predict LBV", type="primary")
    
        
    # Handle prediction
    if predict_clicked:
        # Validate inputs
        errors = predictor.validate_inputs(hydrocarbon, temperature, equiv_ratio, pressure)
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            # Make prediction
            with st.spinner("Calculating LBV..."):
                lbv_prediction = predictor.predict_lbv(hydrocarbon, temperature, equiv_ratio, pressure)
            
            if lbv_prediction is not None:
                # Convert units if needed
                if convert_to_ms:
                    lbv_display = convert_units(lbv_prediction, "cm/s", "m/s")
                    unit = "m/s"
                else:
                    lbv_display = lbv_prediction
                    unit = "cm/s"
                
                # Store results in session state
                st.session_state.lbv_result = lbv_display
                st.session_state.lbv_unit = unit
                st.session_state.prediction_made = True
                st.session_state.current_inputs = {
                    'hydrocarbon': hydrocarbon,
                    'temperature': temperature,
                    'equiv_ratio': equiv_ratio,
                    'pressure': pressure
                }
                
                # Add to prediction history
                if 'prediction_history' not in st.session_state:
                    st.session_state.prediction_history = []
                
                # Create history entry
                history_entry = {
                    'hydrocarbon': hydrocarbon,
                    'temperature': temperature,
                    'equiv_ratio': equiv_ratio,
                    'pressure': pressure,
                    'lbv_value': lbv_display,
                    'unit': unit,
                    'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Add to history (no limit)
                st.session_state.prediction_history.insert(0, history_entry)
            else:
                st.error("Prediction failed. Please check your inputs.")
    
    # Display results at the bottom if prediction was made
    if st.session_state.get('prediction_made', False):
        lbv_result = st.session_state.get('lbv_result')
        lbv_unit = st.session_state.get('lbv_unit')
        inputs = st.session_state.get('current_inputs', {})
        
        st.markdown(f"""
        <div class="result-container">
            <h2>🎯 Prediction Result</h2>
            <div class="result-value">{lbv_result:.2f} {lbv_unit}</div>
            <p>Laminar Burning Velocity</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display input summary
        st.subheader("📋 Input Summary")
        
        col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
        
        with col_summary1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Hydrocarbon</div>
                <div class="metric-value">{inputs.get('hydrocarbon', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_summary2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Temperature</div>
                <div class="metric-value">{inputs.get('temperature', 0):.1f} K</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_summary3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Equivalence Ratio</div>
                <div class="metric-value">{inputs.get('equiv_ratio', 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_summary4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Pressure</div>
                <div class="metric-value">{inputs.get('pressure', 0):.1f} atm</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display prediction history if enabled
        if st.session_state.get('show_history', True) and 'prediction_history' in st.session_state and len(st.session_state.prediction_history) > 0:
            st.markdown("---")
            st.subheader("📜 Prediction History")
            
            # Create a nice table for history
            history_data = []
            for i, entry in enumerate(st.session_state.prediction_history):
                history_data.append({
                    "No.": i + 1,
                    "Hydrocarbon": entry['hydrocarbon'][:20] + "..." if len(entry['hydrocarbon']) > 20 else entry['hydrocarbon'],
                    "Temp (K)": f"{entry['temperature']:.1f}",
                    "φ": f"{entry['equiv_ratio']:.2f}",
                    "P (atm)": f"{entry['pressure']:.1f}",
                    "LBV": f"{entry['lbv_value']:.2f} {entry['unit']}",
                    : entry['timestamp'].split()[1][:5]  # Show only HH:MM
                })
            
            # Display as a styled table
            if history_data:
                df_history = pd.DataFrame(history_data)
                st.dataframe(
                    df_history,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "No.": st.column_config.NumberColumn("No.", width="small"),
                        "Hydrocarbon": st.column_config.TextColumn("Hydrocarbon", width="large"),
                        "Temp (K)": st.column_config.TextColumn("Temp (K)", width="small"),
                        "φ": st.column_config.TextColumn("φ", width="small"),
                        "P (atm)": st.column_config.TextColumn("P (atm)", width="small"),
                        "LBV": st.column_config.TextColumn("LBV", width="medium"),
                                            }
                )
                
                # Clear history button
                if st.button("🗑️ Clear History", help="Clear all prediction history"):
                    st.session_state.prediction_history = []
                    st.rerun()
    
    else:
        # Placeholder when no prediction made
        st.markdown("""
        <div class="input-container">
            <h3>🎯 Prediction Results</h3>
            <p>Enter your parameters and click "Predict LBV" to see the results.</p>
            <br>
            <h4>📈 How it works:</h4>
            <ul>
                <li>Select your hydrocarbon fuel type</li>
                <li>Set the initial temperature (300-750 K)</li>
                <li>Choose the equivalence ratio (0.1-2.4)</li>
                <li>Set the pressure (1-10 atm)</li>
                <li>Get instant LBV prediction!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show history even when no current prediction if toggle is on
        if st.session_state.get('show_history', True) and 'prediction_history' in st.session_state and len(st.session_state.prediction_history) > 0:
            st.markdown("---")
            st.subheader("📜 Previous Predictions")
            
            # Create a nice table for history
            history_data = []
            for i, entry in enumerate(st.session_state.prediction_history):
                history_data.append({
                    "No.": i + 1,
                    "Hydrocarbon": entry['hydrocarbon'][:20] + "..." if len(entry['hydrocarbon']) > 20 else entry['hydrocarbon'],
                    "Temp (K)": f"{entry['temperature']:.1f}",
                    "φ": f"{entry['equiv_ratio']:.2f}",
                    "P (atm)": f"{entry['pressure']:.1f}",
                    "LBV": f"{entry['lbv_value']:.2f} {entry['unit']}",
                    : entry['timestamp'].split()[1][:5]  # Show only HH:MM
                })
            
            # Display as a styled table
            if history_data:
                df_history = pd.DataFrame(history_data)
                st.dataframe(
                    df_history,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "No.": st.column_config.NumberColumn("No.", width="small"),
                        "Hydrocarbon": st.column_config.TextColumn("Hydrocarbon", width="large"),
                        "Temp (K)": st.column_config.TextColumn("Temp (K)", width="small"),
                        "φ": st.column_config.TextColumn("φ", width="small"),
                        "P (atm)": st.column_config.TextColumn("P (atm)", width="small"),
                        "LBV": st.column_config.TextColumn("LBV", width="medium"),
                                            }
                )
                
                # Clear history button
                if st.button("🗑️ Clear History", help="Clear all prediction history"):
                    st.session_state.prediction_history = []
                    st.rerun()

if __name__ == "__main__":
    main()
