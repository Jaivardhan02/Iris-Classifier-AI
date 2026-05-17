"""
===============================================================================
HOW TO RUN THIS APP:
1. Open a terminal or command prompt.
2. Navigate to the directory containing this file: `cd d:\\IRIS\\project`
3. Execute the script: `streamlit run app.py`
===============================================================================
"""
import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image

# ---------------------------------------------------------
# App Configuration (Must be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Iris Classifier AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Premium Custom UI Design
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #a78bfa !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Card-like containers for text inputs and elements */
    .st-emotion-cache-16txtl3 {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Styling the main Button */
    .stButton>button {
        background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
        background: linear-gradient(90deg, #7c3aed 0%, #2563eb 100%);
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Custom divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Core Functionality
# ---------------------------------------------------------
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        encoder = joblib.load('models/label_encoder.pkl')
        return model, scaler, encoder
    except FileNotFoundError:
        st.error("⚠️ Error: Model files not found. Please run `python pipeline.py` first.")
        return None, None, None

def main():
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🔮 Iris Classifier AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>An advanced machine learning engine for precise botanical classification.</p>", unsafe_allow_html=True)
    
    model, scaler, encoder = load_model_components()
    
    if model is None:
        return
        
    st.sidebar.markdown("## ⚙️ Inference Engine")
    st.sidebar.markdown("Select your data source to begin prediction.")
    
    input_method = st.sidebar.radio("Data Input Method", ("🧬 Manual Parameters", "📁 Batch Upload (CSV)"))
    
    input_data = None
    
    if input_method == "🧬 Manual Parameters":
        st.sidebar.markdown("---")
        sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
        sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 5.0, 3.4, 0.1)
        petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.5, 0.1)
        petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 3.0, 0.2, 0.1)
        
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                                  columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
        
    elif input_method == "📁 Batch Upload (CSV)":
        st.sidebar.markdown("---")
        uploaded_file = st.sidebar.file_uploader("Upload dataset", type="csv")
        if uploaded_file is not None:
            try:
                input_data = pd.read_csv(uploaded_file)
                expected_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
                if not all(col in input_data.columns for col in expected_cols):
                    st.sidebar.error(f"Missing required columns: {expected_cols}")
                    input_data = None
                else:
                    input_data = input_data[expected_cols]
                    st.sidebar.success("✅ Dataset loaded")
            except Exception as e:
                st.sidebar.error(f"File error: {e}")
    
    # Main Layout
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 📊 Data Pipeline")
        if input_data is not None:
            st.dataframe(input_data, use_container_width=True)
        else:
            st.info("Awaiting data input...")
            
    with col2:
        st.markdown("### 🎯 Model Output")
        if input_data is not None:
            if st.button("Initialize Prediction Sequence 🚀"):
                try:
                    with st.spinner("Executing neural inference..."):
                        scaled_data = scaler.transform(input_data)
                        predictions = model.predict(scaled_data)
                        predicted_classes = encoder.inverse_transform(predictions)
                        probs = model.predict_proba(scaled_data)
                        
                    if len(predicted_classes) == 1:
                        st.success(f"**Identified Classification:** {predicted_classes[0]}")
                        st.markdown("**Confidence Matrix:**")
                        prob_df = pd.DataFrame(probs, columns=encoder.classes_)
                        st.bar_chart(prob_df.T)
                    else:
                        st.success(f"Successfully classified {len(predicted_classes)} samples.")
                        input_data['Predicted_Species'] = predicted_classes
                        st.dataframe(input_data, use_container_width=True)
                        
                        csv = input_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Inference Results",
                            data=csv,
                            file_name='neural_predictions.csv',
                            mime='text/csv',
                        )
                except Exception as e:
                    st.error(f"Inference failure: {e}")

    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>🔬 Evaluation Telemetry</h2>", unsafe_allow_html=True)
    
    plots = {
        "Confusion Matrix": "plots/confusion_matrix.png",
        "ROC Curve Analysis": "plots/roc_curve.png",
        "Model Learning Curve": "plots/learning_curve.png",
        "Feature Distributions": "plots/distributions.png",
        "Correlation Topology": "plots/correlation_matrix.png",
        "Outlier Boxplots": "plots/boxplots.png"
    }
    
    plot_choice = st.selectbox("Select telemetry visualizer:", list(plots.keys()))
    
    plot_path = plots.get(plot_choice)
    if plot_path and os.path.exists(plot_path):
        st.image(plot_path, caption=plot_choice, use_container_width=True)
    else:
        st.warning("Visualizer offline. Ensure training pipeline was executed.")

if __name__ == "__main__":
    main()
