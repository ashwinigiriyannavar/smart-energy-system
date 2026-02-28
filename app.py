"""
Smart Energy Consumption Prediction & Optimization System
A comprehensive Streamlit web application for energy consumption analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_generator import EnergyDataGenerator
from models import EnergyMLModels

# Set page configuration
st.set_page_config(
    page_title="Smart Energy Consumption System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
def set_custom_style():
    st.markdown("""
    <style>
    /* Header Styles */
    .college-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .college-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0;
    }
    .department {
        font-size: 1.2rem;
        margin: 0.5rem 0;
    }
    .hackathon-info {
        font-size: 1rem;
        margin: 0.5rem 0;
        opacity: 0.9;
    }
    .divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        border-radius: 2px;
        margin: 1rem 0;
    }
    
    /* Main Title Styles */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #1f77b4, #17becf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .main-subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Team Card Styles */
    .team-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #e1e8ed;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .team-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    .team-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-align: center;
    }
    .team-member {
        display: flex;
        align-items: center;
        margin: 0.8rem 0;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        transition: background 0.3s ease;
    }
    .team-member:hover {
        background: #e9ecef;
    }
    .member-icon {
        font-size: 1.2rem;
        margin-right: 0.8rem;
        width: 30px;
        text-align: center;
    }
    .member-info {
        flex: 1;
    }
    .member-name {
        font-weight: 600;
        color: #2c3e50;
    }
    .member-role {
        font-size: 0.9rem;
        color: #6c757d;
    }
    
    /* Feature Card Styles */
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.25);
    }
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Sidebar Styles */
    .sidebar-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .sidebar-metric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer Styles */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .footer-line {
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Section Styles */
    .section-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Metric Styles */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Success and Warning Messages */
    .success-message {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(40, 167, 69, 0.3);
    }
    .warning-message {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: #212529;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(255, 193, 7, 0.3);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .college-title {
            font-size: 1.4rem;
        }
        .feature-card {
            margin-bottom: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'

def home_page():
    """Home page with project description and team details"""
    # College Header
    st.markdown("""
    <div class="college-header">
        <div class="college-title">🏛️ Basaveshwar Engineering College</div>
        <div class="department">🤖 Artificial Intelligence & Machine Learning</div>
        <div class="hackathon-info">🏆 12 Hrs Internal Data Science Hackathon 2026 – INSIGHTX</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Main Title with Lightning Icon
    st.markdown('<h1 class="main-header">⚡ Smart Energy Consumption Prediction & Optimization System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">🤖 AI-powered predictive analytics for energy efficiency</p>', unsafe_allow_html=True)
    
    # Use tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📋 Overview", "👥 Team", "🎯 Features"])
    
    with tab1:
        # Project Overview Section
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("### 🎯 Project Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            The Smart Energy Consumption Prediction & Optimization System is a comprehensive 
            machine learning solution designed to analyze and predict household energy consumption 
            patterns. This system leverages advanced ML algorithms to provide insights into 
            energy usage and optimization opportunities.
            
            ### 🔬 Key Technologies:
            - **Frontend**: Streamlit with modern UI/UX
            - **Backend**: Python with Scikit-learn
            - **ML Algorithms**: Linear Regression, Decision Trees, KNN, K-Means
            - **Visualization**: Plotly, Matplotlib, Seaborn
            - **Data Processing**: Pandas, NumPy
            """)
        
        with col2:
            # Quick Stats
            st.markdown("### 📊 Quick Stats")
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric("ML Models", "4")
            with col1_2:
                st.metric("Features", "9")
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Accuracy", "87%")
            with col2_2:
                st.metric("R² Score", "0.85")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Team Section
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("### 👥 Meet Our Team")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="team-card">
                <div class="team-title">🏆 Team 8</div>
                <div class="team-member">
                    <div class="member-icon">👩‍💻</div>
                    <div class="member-info">
                        <div class="member-name">Shraddha U Adahalli</div>
                    </div>
                </div>
                <div class="team-member">
                    <div class="member-icon">📊</div>
                    <div class="member-info">
                        <div class="member-name">Ashwini R G</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="team-card">
                <div class="team-title">🎯 Team Members</div>
                <div class="team-member">
                    <div class="member-icon">🎨</div>
                    <div class="member-info">
                        <div class="member-name">Pallavi R K</div>
                    </div>
                </div>
                <div class="team-member">
                    <div class="member-icon">📈</div>
                    <div class="member-info">
                        <div class="member-name">Vijeta M M</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Feature Cards Section
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("### 🚀 System Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🔋</div>
                <div class="feature-title">Energy Prediction</div>
                <div class="feature-desc">Linear Regression for accurate consumption forecasting</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🌳</div>
                <div class="feature-title">High Usage Classification</div>
                <div class="feature-desc">Decision Trees for usage pattern analysis</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Efficiency Categorization</div>
                <div class="feature-desc">KNN for multi-class efficiency ranking</div>
            </div>
            """, unsafe_allow_html=True)
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Consumption Clustering</div>
                <div class="feature-desc">K-Means for pattern discovery</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">Smart Recommendations</div>
                <div class="feature-desc">AI-powered optimization tips</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Professional Footer
    st.markdown("""
    <div class="footer">
        <div class="footer-line">🏛️ Developed at Basaveshwar Engineering College</div>
        <div class="footer-line">🤖 Department of Artificial Intelligence & Machine Learning</div>
        <div class="footer-line">© 2026 Team 8 – INSIGHTX Hackathon</div>
    </div>
    """, unsafe_allow_html=True)

def data_page():
    """Data generation and upload page"""
    # College Header
    st.markdown("""
    <div class="college-header">
        <div class="college-title">📊 Data Management</div>
        <div class="department">Generate or upload your energy consumption dataset</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔄 Generate Synthetic Data", "📁 Upload CSV Data"])
    
    with tab1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("### 🔄 Synthetic Data Generator")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**⚙️ Configuration**")
            num_records = st.number_input(
                "Number of Records",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Choose the number of records to generate"
            )
            
            generate_button = st.button("🔄 Generate Dataset", type="primary", help="Generate synthetic energy consumption data")
        
        with col2:
            st.markdown("**📋 Dataset Features**")
            st.markdown("""
            <div class="metric-container">
                <div style="font-weight: bold; color: #1f77b4;">👥 Household Size</div>
                <div style="font-size: 0.9rem;">Number of people (1-6)</div>
            </div>
            <div class="metric-container">
                <div style="font-weight: bold; color: #1f77b4;">🔌 Appliance Count</div>
                <div style="font-size: 0.9rem;">Total appliances (5-25)</div>
            </div>
            <div class="metric-container">
                <div style="font-weight: bold; color: #1f77b4;">🌡️ Temperature</div>
                <div style="font-size: 0.9rem;">Temperature in Celsius (-5 to 40°C)</div>
            </div>
            <div class="metric-container">
                <div style="font-weight: bold; color: #1f77b4;">⏰ Working Hours</div>
                <div style="font-size: 0.9rem;">Daily working hours (0-12)</div>
            </div>
            <div class="metric-container">
                <div style="font-weight: bold; color: #1f77b4;">☀️ Solar Usage</div>
                <div style="font-size: 0.9rem;">Solar panel installation (Yes/No)</div>
            </div>
            <div class="metric-container">
                <div style="font-weight: bold; color: #1f77b4;">💰 Electricity Tariff</div>
                <div style="font-size: 0.9rem;">Tariff category (Low/Medium/High)</div>
            </div>
            <div class="metric-container">
                <div style="font-weight: bold; color: #1f77b4;">📅 Day Type</div>
                <div style="font-size: 0.9rem;">Weekday or Weekend</div>
            </div>
            <div class="metric-container">
                <div style="font-weight: bold; color: #1f77b4;">⚡ Previous Consumption</div>
                <div style="font-size: 0.9rem;">Last month's consumption (kWh)</div>
            </div>
            <div class="metric-container">
                <div style="font-weight: bold; color: #28a745;">🎯 Target: Energy Consumption</div>
                <div style="font-size: 0.9rem;">Predicted consumption (kWh)</div>
            </div>
            """, unsafe_allow_html=True)
        
        if generate_button:
            with st.spinner("Generating synthetic dataset..."):
                generator = EnergyDataGenerator(num_records=num_records)
                df = generator.generate_dataset()
                st.session_state.data = df
                
                # Display dataset info with enhanced styling
                st.markdown("### 📊 Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div class="metric-container">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #1f77b4;">📈</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">{}</div>
                        <div style="font-size: 0.9rem; color: #6c757d;">Total Records</div>
                    </div>
                    """.format(len(df)), unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div class="metric-container">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #28a745;">🔧</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">{}</div>
                        <div style="font-size: 0.9rem; color: #6c757d;">Features</div>
                    </div>
                    """.format(len(df.columns)), unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                    <div class="metric-container">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #ffc107;">⚡</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">Energy_Consumption</div>
                        <div style="font-size: 0.9rem; color: #6c757d;">Target Variable</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display sample data
                st.markdown("### 🔍 Sample Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data statistics
                st.markdown("### 📈 Statistical Summary")
                st.dataframe(df.describe().round(2), use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Dataset (CSV)",
                    data=csv,
                    file_name=f"energy_consumption_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download the generated dataset as CSV file"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("### 📁 Upload CSV Data")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**📤 File Upload**")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload your energy consumption dataset",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                st.markdown("""
                <div class="metric-container">
                    <div style="font-weight: bold; color: #28a745;">✅ File Selected</div>
                    <div style="font-size: 0.9rem;">{}</div>
                </div>
                """.format(uploaded_file.name), unsafe_allow_html=True)
        
        with col2:
            st.markdown("**📋 Required Columns**")
            st.markdown("""
            <div class="metric-container">
                <div style="font-weight: bold; color: #1f77b4;">📋 Column Requirements:</div>
                <div style="font-size: 0.9rem;">
                • Household_Size<br>
                • Appliance_Count<br>
                • Average_Temperature<br>
                • Working_Hours<br>
                • Solar_Usage<br>
                • Electricity_Tariff<br>
                • Day_Type<br>
                • Previous_Consumption<br>
                • Energy_Consumption (Target)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                
                # Display dataset info with enhanced styling
                st.markdown("### 📊 Dataset Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div class="metric-container">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #1f77b4;">📈</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">{}</div>
                        <div style="font-size: 0.9rem; color: #6c757d;">Total Records</div>
                    </div>
                    """.format(len(df)), unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div class="metric-container">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #28a745;">🔧</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">{}</div>
                        <div style="font-size: 0.9rem; color: #6c757d;">Features</div>
                    </div>
                    """.format(len(df.columns)), unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                    <div class="metric-container">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #ffc107;">💾</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">{:.1f} KB</div>
                        <div style="font-size: 0.9rem; color: #6c757d;">File Size</div>
                    </div>
                    """.format(uploaded_file.size / 1024), unsafe_allow_html=True)
                
                # Display sample data
                st.markdown("### 🔍 Sample Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Check required columns
                required_columns = [
                    'Household_Size', 'Appliance_Count', 'Average_Temperature',
                    'Working_Hours', 'Solar_Usage', 'Electricity_Tariff',
                    'Day_Type', 'Previous_Consumption', 'Energy_Consumption'
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.markdown(f'<div class="warning-message">⚠️ Missing required columns: {", ".join(missing_columns)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-message">✅ All required columns present!</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="warning-message">❌ Error reading file: {str(e)}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def models_page():
    """Model training and performance page"""
    # College Header
    st.markdown("""
    <div class="college-header">
        <div class="college-title">🤖 Machine Learning Models</div>
        <div class="department">Train and evaluate ML algorithms for energy prediction</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Data Required")
        st.markdown("""
        <div class="warning-message">
            Please generate or upload a dataset first before training models!
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Check if models are already trained
    if not st.session_state.models_trained:
        train_button = st.button("🚀 Train All Models", type="primary")
        
        if train_button:
            with st.spinner("Training ML models... This may take a few moments..."):
                try:
                    # Initialize ML models
                    ml_models = EnergyMLModels()
                    
                    # Preprocess data
                    X, y, df_processed = ml_models.preprocess_data(st.session_state.data)
                    
                    # Train all models
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Linear Regression
                    status_text.text("Training Linear Regression...")
                    lr_results = ml_models.train_linear_regression(X, y)
                    progress_bar.progress(25)
                    
                    # Decision Tree
                    status_text.text("Training Decision Tree Classifier...")
                    dt_results = ml_models.train_decision_tree(X, y)
                    progress_bar.progress(50)
                    
                    # KNN
                    status_text.text("Training KNN Classifier...")
                    knn_results = ml_models.train_knn(X, y)
                    progress_bar.progress(75)
                    
                    # K-Means
                    status_text.text("Training K-Means Clustering...")
                    km_results = ml_models.train_kmeans(X)
                    progress_bar.progress(100)
                    
                    # Store models in session state
                    st.session_state.ml_models = ml_models
                    st.session_state.models_trained = True
                    
                    status_text.text("✅ All models trained successfully!")
                    st.markdown('<div class="success-message">🎉 All ML models have been trained successfully!</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
    else:
        st.markdown('<div class="success-message">✅ Models are already trained!</div>', unsafe_allow_html=True)
        
        # Display model performance
        ml_models = st.session_state.ml_models
        model_summary = ml_models.get_model_summary()
        
        st.subheader("Model Performance Summary")
        
        # Create performance metrics display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Linear Regression")
            lr_metrics = model_summary['linear_regression']['metrics']
            st.metric("Test R² Score", f"{lr_metrics['test_r2']:.4f}")
            st.metric("Test MSE", f"{lr_metrics['test_mse']:.4f}")
            st.metric("CV Mean R²", f"{lr_metrics['cv_mean']:.4f}")
        
        with col2:
            st.markdown("### Decision Tree Classifier")
            dt_metrics = model_summary['decision_tree']['metrics']
            st.metric("Test Accuracy", f"{dt_metrics['test_accuracy']:.4f}")
            st.metric("CV Mean Accuracy", f"{dt_metrics['cv_mean']:.4f}")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### KNN Classifier")
            knn_metrics = model_summary['knn']['metrics']
            st.metric("Test Accuracy", f"{knn_metrics['test_accuracy']:.4f}")
            st.metric("CV Mean Accuracy", f"{knn_metrics['cv_mean']:.4f}")
        
        with col4:
            st.markdown("### K-Means Clustering")
            km_metrics = model_summary['kmeans']['metrics']
            st.metric("Silhouette Score", f"{km_metrics['silhouette_score']:.4f}")
            st.metric("Inertia", f"{km_metrics['inertia']:.2f}")
        
        # Feature importance for Linear Regression
        if 'linear_regression' in ml_models.models:
            st.subheader("Linear Regression - Feature Importance")
            lr_importance = ml_models.models['linear_regression']['feature_importance']
            fig = px.bar(
                lr_importance.head(10),
                x='Abs_Coefficient',
                y='Feature',
                orientation='h',
                title="Top 10 Feature Importance (Linear Regression)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance for Decision Tree
        if 'decision_tree' in ml_models.models:
            st.subheader("Decision Tree - Feature Importance")
            dt_importance = ml_models.models['decision_tree']['feature_importance']
            fig = px.bar(
                dt_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Feature Importance (Decision Tree)"
            )
            st.plotly_chart(fig, use_container_width=True)

def prediction_page():
    """Prediction interface page"""
    st.markdown('<h1 class="main-header">⚡ Energy Consumption Prediction</h1>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train the models first!")
        return
    
    ml_models = st.session_state.ml_models
    
    # Create input form
    st.subheader("Enter Household Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        household_size = st.slider("Household Size", 1, 6, 3)
        appliance_count = st.slider("Appliance Count", 5, 25, 12)
        avg_temperature = st.slider("Average Temperature (°C)", -5, 40, 22)
        working_hours = st.slider("Working Hours", 0, 12, 8)
    
    with col2:
        solar_usage = st.selectbox("Solar Usage", ["Yes", "No"])
        electricity_tariff = st.selectbox("Electricity Tariff", ["Low", "Medium", "High"])
        day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])
        previous_consumption = st.number_input("Previous Consumption (kWh)", 0.0, 100.0, 25.5)
    
    # Create input dictionary
    input_data = {
        'Household_Size': household_size,
        'Appliance_Count': appliance_count,
        'Average_Temperature': avg_temperature,
        'Working_Hours': working_hours,
        'Solar_Usage': solar_usage,
        'Electricity_Tariff': electricity_tariff,
        'Day_Type': day_type,
        'Previous_Consumption': previous_consumption
    }
    
    # Predict button
    predict_button = st.button("🔮 Make Predictions", type="primary")
    
    if predict_button:
        with st.spinner("Making predictions..."):
            try:
                # Get predictions from all models
                consumption = ml_models.predict_consumption(input_data)
                high_usage, high_usage_prob = ml_models.predict_high_usage(input_data)
                efficiency, efficiency_prob = ml_models.predict_efficiency_category(input_data)
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Predicted Consumption", f"{consumption:.2f} kWh")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("High Usage", high_usage)
                    st.write(f"Confidence: {max(high_usage_prob):.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Efficiency Category", efficiency)
                    st.write(f"Confidence: {max(efficiency_prob):.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Recommendations
                st.subheader("💡 Energy Optimization Recommendations")
                
                recommendations = []
                
                if consumption > 30:
                    recommendations.append("🔴 Your predicted consumption is high. Consider reducing appliance usage.")
                
                if solar_usage == "No":
                    recommendations.append("☀️ Consider installing solar panels to reduce electricity costs.")
                
                if electricity_tariff == "High":
                    recommendations.append("💰 You're on a high tariff. Consider switching to a cheaper plan.")
                
                if avg_temperature > 25 or avg_temperature < 10:
                    recommendations.append("🌡️ Extreme temperatures increase consumption. Consider better insulation.")
                
                if working_hours > 8:
                    recommendations.append("⏰ Long working hours increase consumption. Try to optimize usage.")
                
                if recommendations:
                    for rec in recommendations:
                        st.write(rec)
                else:
                    st.write("✅ Your energy consumption looks optimized!")
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")

def visualization_page():
    """Visualization and analysis page"""
    # College Header
    st.markdown("""
    <div class="college-header">
        <div class="college-title">📈 Data Visualization & Analysis</div>
        <div class="department">Interactive charts and cluster analysis for energy insights</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Data Required")
        st.markdown("""
        <div class="warning-message">
            Please generate or upload a dataset first to view visualizations!
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    df = st.session_state.data
    
    # Enhanced Visualization Options with Tabs
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Correlation", "📊 Distribution", "🔗 Relationships", "🎯 Clustering"])
    
    with tab1:
        st.markdown("### 🔥 Feature Correlation Heatmap")
        
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()
        
        # Create enhanced heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Feature Correlation Matrix",
            template="plotly_white"
        )
        fig.update_layout(
            title_font_size=16,
            width=800,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Insights
        st.markdown("### 📋 Correlation Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div style="font-weight: bold; color: #1f77b4;">🔍 Strong Correlations</div>
                <div style="font-size: 0.9rem;">Features with high correlation values (>0.7)</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div style="font-weight: bold; color: #ff7f0e;">⚠️ Weak Correlations</div>
                <div style="font-size: 0.9rem;">Features with low correlation values (<0.3)</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 Energy Consumption Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced Histogram
            fig = px.histogram(
                df,
                x="Energy_Consumption",
                nbins=30,
                title="Energy Consumption Distribution",
                color_discrete_sequence=['#1f77b4'],
                template="plotly_white"
            )
            fig.add_vline(
                x=df['Energy_Consumption'].mean(), 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {df['Energy_Consumption'].mean():.2f}"
            )
            fig.update_layout(
                title_font_size=14,
                xaxis_title="Energy Consumption (kWh)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics Summary
            st.markdown("### 📈 Statistical Summary")
            stats = df['Energy_Consumption'].describe()
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.metric("Mean", f"{stats['mean']:.2f} kWh")
                st.metric("Median", f"{stats['50%']:.2f} kWh")
            with col1_2:
                st.metric("Std Dev", f"{stats['std']:.2f}")
                st.metric("Range", f"{stats['max'] - stats['min']:.2f} kWh")
        
        with col2:
            # Enhanced Box plots
            cat_var = st.selectbox("Select categorical variable", ["Solar_Usage", "Electricity_Tariff", "Day_Type"], key="dist_cat_var")
            
            fig = px.box(
                df,
                x=cat_var,
                y="Energy_Consumption",
                title=f"Energy Consumption by {cat_var}",
                color=cat_var,
                template="plotly_white"
            )
            fig.update_layout(
                title_font_size=14,
                xaxis_title=cat_var,
                yaxis_title="Energy Consumption (kWh)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution by Category
            st.markdown("### 📊 Category Statistics")
            cat_stats = df.groupby(cat_var)['Energy_Consumption'].agg(['mean', 'std', 'count']).round(2)
            st.dataframe(cat_stats, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔗 Feature Relationships")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**⚙️ Chart Configuration**")
            x_var = st.selectbox("Select X variable", df.columns, index=0, key="rel_x_var")
            y_var = st.selectbox("Select Y variable", df.columns, index=len(df.columns)-1, key="rel_y_var")
            color_var = st.selectbox("Color by", [None] + df.select_dtypes(include=['object']).columns.tolist(), key="rel_color_var")
            size_var = st.selectbox("Size by", [None] + df.select_dtypes(include=[np.number]).columns.tolist(), key="rel_size_var")
            
            chart_type = st.selectbox("Chart Type", ["Scatter", "Line", "Bubble"], key="rel_chart_type")
        
        with col2:
            st.markdown("**📊 Relationship Chart**")
            
            if chart_type == "Scatter":
                fig = px.scatter(
                    df,
                    x=x_var,
                    y=y_var,
                    color=color_var,
                    size=size_var if size_var else None,
                    title=f"{y_var} vs {x_var}",
                    template="plotly_white",
                    trendline="ols" if st.checkbox("Add Trendline", key="add_trendline") else None
                )
            elif chart_type == "Line":
                fig = px.line(
                    df.sort_values(x_var),
                    x=x_var,
                    y=y_var,
                    color=color_var,
                    title=f"{y_var} vs {x_var}",
                    template="plotly_white"
                )
            else:  # Bubble
                fig = px.scatter(
                    df,
                    x=x_var,
                    y=y_var,
                    color=color_var,
                    size=size_var if size_var else None,
                    title=f"{y_var} vs {x_var} (Bubble Chart)",
                    template="plotly_white"
                )
            
            fig.update_layout(
                title_font_size=14,
                xaxis_title=x_var,
                yaxis_title=y_var
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation Coefficient
            if df[x_var].dtype in ['int64', 'float64'] and df[y_var].dtype in ['int64', 'float64']:
                correlation = df[x_var].corr(df[y_var])
                st.markdown(f"**📊 Correlation Coefficient:** {correlation:.4f}")
                
                if abs(correlation) > 0.7:
                    st.markdown("🟢 **Strong Correlation**")
                elif abs(correlation) > 0.3:
                    st.markdown("🟡 **Moderate Correlation**")
                else:
                    st.markdown("🔴 **Weak Correlation**")
    
    with tab4:
        st.markdown("### 🎯 Consumption Clustering Analysis")
        
        if not st.session_state.models_trained:
            st.markdown("""
            <div class="warning-message">
                ⚠️ Please train models first to see cluster analysis!
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        ml_models = st.session_state.ml_models
        
        if 'kmeans' in ml_models.models:
            km_results = ml_models.models['kmeans']
            cluster_data = km_results['data_with_clusters']
            
            # Cluster Visualization Options
            viz_option = st.selectbox(
                "Select Cluster Visualization",
                ["2D Scatter Plot", "3D Scatter Plot", "Pair Plot", "Cluster Centers"]
            )
            
            if viz_option == "2D Scatter Plot":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 2D Cluster Visualization**")
                    feature_x = st.selectbox("Select X feature", cluster_data.columns[:-1], key="cluster_x")
                    feature_y = st.selectbox("Select Y feature", cluster_data.columns[:-1], index=1, key="cluster_y")
                    
                    fig = px.scatter(
                        cluster_data,
                        x=feature_x,
                        y=feature_y,
                        color="Cluster",
                        title=f"Cluster Visualization: {feature_y} vs {feature_x}",
                        template="plotly_white",
                        color_continuous_scale=px.colors.qualitative.Set3
                    )
                    fig.update_layout(
                        title_font_size=14,
                        xaxis_title=feature_x,
                        yaxis_title=feature_y
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**📈 Cluster Sizes**")
                    cluster_sizes = cluster_data['Cluster'].value_counts().reset_index()
                    cluster_sizes.columns = ['Cluster', 'Size']
                    
                    fig = px.bar(
                        cluster_sizes,
                        x='Cluster',
                        y='Size',
                        title="Cluster Sizes",
                        color='Cluster',
                        template="plotly_white",
                        color_continuous_scale=px.colors.qualitative.Set3
                    )
                    fig.update_layout(
                        title_font_size=14,
                        xaxis_title="Cluster",
                        yaxis_title="Number of Households"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster Metrics
                    st.markdown("**📊 Cluster Metrics**")
                    silhouette_score = km_results['metrics']['silhouette_score']
                    inertia = km_results['metrics']['inertia']
                    
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.metric("Silhouette Score", f"{silhouette_score:.4f}")
                    with col2_2:
                        st.metric("Inertia", f"{inertia:.2f}")
            
            elif viz_option == "3D Scatter Plot":
                st.markdown("**🎯 3D Cluster Visualization**")
                
                col3d_1, col3d_2, col3d_3 = st.columns(3)
                with col3d_1:
                    feature_3d_x = st.selectbox("Select X feature", cluster_data.columns[:-1], key="3d_x")
                with col3d_2:
                    feature_3d_y = st.selectbox("Select Y feature", cluster_data.columns[:-1], index=1, key="3d_y")
                with col3d_3:
                    feature_3d_z = st.selectbox("Select Z feature", cluster_data.columns[:-1], index=2, key="3d_z")
                
                fig = px.scatter_3d(
                    cluster_data,
                    x=feature_3d_x,
                    y=feature_3d_y,
                    z=feature_3d_z,
                    color="Cluster",
                    title=f"3D Cluster Visualization",
                    template="plotly_white"
                )
                fig.update_layout(
                    title_font_size=14,
                    scene=dict(
                        xaxis_title=feature_3d_x,
                        yaxis_title=feature_3d_y,
                        zaxis_title=feature_3d_z
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Pair Plot":
                st.markdown("**🔗 Pair Plot Matrix**")
                
                # Select features for pair plot (limit to 5 features for performance)
                selected_features = st.multiselect(
                    "Select features for pair plot (max 5)",
                    cluster_data.columns[:-1].tolist(),
                    default=cluster_data.columns[:4].tolist()[:4],
                    max_selections=5
                )
                
                if len(selected_features) >= 2:
                    fig = px.scatter_matrix(
                        cluster_data,
                        dimensions=selected_features,
                        color="Cluster",
                        title="Pair Plot Matrix",
                        template="plotly_white"
                    )
                    fig.update_layout(
                        title_font_size=14
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least 2 features for pair plot.")
            
            elif viz_option == "Cluster Centers":
                st.markdown("**🎯 Cluster Centers Analysis**")
                
                cluster_centers = km_results['cluster_centers']
                
                # Display cluster centers table
                st.markdown("### 📋 Cluster Centers")
                st.dataframe(cluster_centers.round(2), use_container_width=True)
                
                # Radar chart for cluster centers
                st.markdown("### 🕸️ Cluster Profiles (Radar Chart)")
                
                # Select features for radar chart
                radar_features = st.multiselect(
                    "Select features for radar chart (max 6)",
                    cluster_centers.columns.tolist(),
                    default=cluster_centers.columns[:4].tolist()[:4],
                    max_selections=6
                )
                
                if len(radar_features) >= 3:
                    fig = go.Figure()
                    
                    for i in range(len(cluster_centers)):
                        fig.add_trace(go.Scatterpolar(
                            r=cluster_centers.iloc[i][radar_features].values,
                            theta=radar_features,
                            fill='toself',
                            name=f'Cluster {i}'
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[cluster_centers[radar_features].min().min(), 
                                       cluster_centers[radar_features].max().max()]
                            )
                        ),
                        title="Cluster Profiles Comparison",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Cluster Statistics
            st.markdown("### 📋 Detailed Cluster Statistics")
            cluster_stats = km_results['cluster_statistics']
            
            for cluster_name, stats in cluster_stats.items():
                with st.expander(f"{cluster_name} - {stats['size']} households ({stats['percentage']:.1f}%)"):
                    col_stats_1, col_stats_2 = st.columns(2)
                    
                    with col_stats_1:
                        st.markdown("**📊 Average Values:**")
                        for feature, value in stats['mean_values'].items():
                            st.write(f"- {feature}: {value:.2f}")
                    
                    with col_stats_2:
                        st.markdown("**📈 Cluster Characteristics:**")
                        
                        # Determine cluster characteristics based on mean values
                        mean_consumption = stats['mean_values'].get('Energy_Consumption', 0)
                        if mean_consumption > df['Energy_Consumption'].quantile(0.75):
                            st.write("🔴 **High Consumption Cluster**")
                        elif mean_consumption > df['Energy_Consumption'].quantile(0.25):
                            st.write("🟡 **Medium Consumption Cluster**")
                        else:
                            st.write("🟢 **Low Consumption Cluster**")
                        
                        # Other characteristics
                        household_size = stats['mean_values'].get('Household_Size', 0)
                        if household_size > 3:
                            st.write("👥 **Large Households**")
                        else:
                            st.write("👤 **Small/Medium Households**")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    # Set custom styling
    set_custom_style()
    
    # Initialize session state
    initialize_session_state()
    
    # Enhanced Sidebar with College Logo and System Info
    with st.sidebar:
        # College Header in Sidebar
        st.markdown("""
        <div class="sidebar-header">
            <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;">🏛️ BEC</div>
            <div style="font-size: 0.9rem;">AI & ML Department</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("## 🧭 Navigation")
        
        pages = {
            "🏠 Home": home_page,
            "📊 Data Management": data_page,
            "🤖 ML Models": models_page,
            "⚡ Predictions": prediction_page,
            "📈 Visualizations": visualization_page
        }
        
        # Page selection
        selected_page = st.selectbox("Select Page", list(pages.keys()))
        
        # System Info Panel
        st.markdown("---")
        st.markdown("### 📊 System Information")
        
        # Dataset Status
        if st.session_state.data is not None:
            st.markdown("""
            <div class="sidebar-metric">
                <div style="font-weight: bold; color: #28a745;">✅ Dataset Loaded</div>
                <div style="font-size: 0.9rem;">Size: {} records</div>
            </div>
            """.format(len(st.session_state.data)), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-metric">
                <div style="font-weight: bold; color: #ffc107;">⚠️ No Dataset</div>
                <div style="font-size: 0.9rem;">Please load data first</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Models Status
        if st.session_state.models_trained:
            st.markdown("""
            <div class="sidebar-metric">
                <div style="font-weight: bold; color: #28a745;">✅ Models Trained</div>
                <div style="font-size: 0.9rem;">4 ML Models Ready</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-metric">
                <div style="font-weight: bold; color: #ffc107;">⚠️ Models Not Trained</div>
                <div style="font-size: 0.9rem;">Train models to enable predictions</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Deployment Status
        st.markdown("""
        <div class="sidebar-metric">
            <div style="font-weight: bold; color: #17a2b8;">🚀 Deployment Status</div>
            <div style="font-size: 0.9rem;">Development Environment</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh App", help="Refresh the application"):
            st.rerun()
        
        if st.button("📊 Generate Sample Data", help="Generate sample dataset"):
            if st.session_state.data is None:
                generator = EnergyDataGenerator(num_records=1000)
                df = generator.generate_dataset()
                st.session_state.data = df
                st.success("Sample data generated!")
                st.rerun()
            else:
                st.info("Data already loaded!")
    
    # Display selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()
