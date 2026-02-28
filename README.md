# Smart Energy Consumption Prediction & Optimization System

A comprehensive web-based machine learning system for analyzing and predicting household energy consumption patterns using Python and Streamlit.

## 🌟 Features

### 📊 Data Management
- **Synthetic Data Generator**: Generate realistic energy consumption datasets with 1000+ records
- **CSV Upload**: Import your own energy consumption data
- **Data Validation**: Automatic validation of required features

### 🤖 Machine Learning Models
- **Linear Regression**: Predict energy consumption with R² score metrics
- **Decision Tree Classifier**: Classify high usage (Yes/No) with accuracy metrics
- **KNN Classifier**: Predict efficiency categories (Low/Medium/High)
- **K-Means Clustering**: Group consumption patterns with silhouette score

### 📈 Visualizations
- **Correlation Heatmap**: Feature relationship analysis
- **Consumption Distribution**: Histograms and box plots
- **Feature Relationships**: Interactive scatter plots
- **Cluster Analysis**: Visual cluster representation

### ⚡ Prediction Interface
- **User Input Form**: Intuitive household information input
- **Real-time Predictions**: Instant results from all ML models
- **Optimization Recommendations**: Personalized energy-saving tips

## 🏗️ Project Structure

```
Smart_energy/
├── app.py                 # Main Streamlit application
├── data_generator.py      # Synthetic data generation
├── models.py             # ML models implementation
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project**
   ```bash
   cd Smart_energy
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📋 Dataset Features

The system uses the following features for energy consumption analysis:

| Feature | Type | Description |
|---------|------|-------------|
| Household_Size | Numerical | Number of people (1-6) |
| Appliance_Count | Numerical | Total appliances (5-25) |
| Average_Temperature | Numerical | Temperature in Celsius (-5 to 40°C) |
| Working_Hours | Numerical | Daily working hours (0-12) |
| Solar_Usage | Categorical | Solar panel installation (Yes/No) |
| Electricity_Tariff | Categorical | Tariff category (Low/Medium/High) |
| Day_Type | Categorical | Weekday or Weekend |
| Previous_Consumption | Numerical | Last month's consumption (kWh) |
| Energy_Consumption | Numerical | Target variable (kWh) |

## 🎯 Model Performance Metrics

### Linear Regression
- **R² Score**: Measures prediction accuracy
- **Mean Squared Error**: Prediction error metric
- **Cross-validation**: Model robustness testing

### Decision Tree Classifier
- **Accuracy**: Classification performance
- **Confusion Matrix**: Detailed classification results
- **Feature Importance**: Variable significance ranking

### KNN Classifier
- **Accuracy**: Multi-class classification performance
- **Classification Report**: Precision, Recall, F1-score

### K-Means Clustering
- **Silhouette Score**: Cluster quality metric
- **Inertia**: Within-cluster variance
- **Cluster Analysis**: Group characteristics

## 🖥️ Application Pages

### 🏠 Home Page
- Project overview and description
- Team details and contact information
- Quick start guide

### 📊 Data Management
- Generate synthetic datasets
- Upload CSV files
- Data preview and validation

### 🤖 ML Models
- Train all ML models
- View performance metrics
- Feature importance analysis

### ⚡ Predictions
- Interactive input form
- Real-time predictions
- Optimization recommendations

### 📈 Visualizations
- Correlation heatmaps
- Distribution plots
- Feature relationships
- Cluster analysis

## 🔧 Technical Implementation

### Data Preprocessing
- **Label Encoding**: Convert categorical variables
- **Feature Scaling**: StandardScaler for numerical features
- **Train-Test Split**: 80-20 split with stratification

### Model Training
- **Cross-validation**: 5-fold CV for robust evaluation
- **Hyperparameter Tuning**: Optimized model parameters
- **Performance Metrics**: Comprehensive evaluation

### Web Interface
- **Streamlit Framework**: Modern web application
- **Plotly Integration**: Interactive visualizations
- **Responsive Design**: Mobile-friendly layout

## 📊 Usage Example

1. **Generate Data**: Click "Generate Synthetic Data" with 1000 records
2. **Train Models**: Click "Train All Models" - takes ~30 seconds
3. **Make Predictions**: Fill in household information and get instant results
4. **Analyze**: Explore visualizations to understand patterns

## 🎨 UI Features

- **Professional Dashboard**: Clean, modern interface
- **Sidebar Navigation**: Easy page switching
- **Metrics Display**: Clear performance indicators
- **Interactive Charts**: Hover effects and zoom
- **Responsive Layout**: Works on all devices

## 🔄 Model Pipeline

```python
# Data Generation
generator = EnergyDataGenerator(num_records=1000)
df = generator.generate_dataset()

# Preprocessing
ml_models = EnergyMLModels()
X, y, df_processed = ml_models.preprocess_data(df)

# Model Training
lr_results = ml_models.train_linear_regression(X, y)
dt_results = ml_models.train_decision_tree(X, y)
knn_results = ml_models.train_knn(X, y)
km_results = ml_models.train_kmeans(X)

# Predictions
consumption = ml_models.predict_consumption(input_data)
high_usage = ml_models.predict_high_usage(input_data)
efficiency = ml_models.predict_efficiency_category(input_data)
```

## 🐛 Troubleshooting

### Common Issues

1. **Module Not Found Error**
   ```bash
   pip install -r requirements.txt
   ```

2. **Streamlit Not Running**
   ```bash
   streamlit run app.py
   ```

3. **Memory Issues**
   - Reduce dataset size below 5000 records
   - Close other applications

4. **Model Training Slow**
   - Reduce dataset size
   - Use fewer cross-validation folds

## 📈 Performance Optimization

- **Efficient Data Processing**: Vectorized operations
- **Model Caching**: Store trained models in session state
- **Lazy Loading**: Load components only when needed
- **Memory Management**: Clean up unused variables

## 🔮 Future Enhancements

- **Time Series Analysis**: Add temporal patterns
- **More ML Models**: Random Forest, XGBoost, Neural Networks
- **Real-time Data**: IoT sensor integration
- **Mobile App**: React Native application
- **Cloud Deployment**: AWS/Azure hosting

## 📞 Support

For technical support and questions:
- **Email**: energy-team@example.com
- **Website**: www.smartenergy.com
- **Documentation**: Check inline code comments

## 📄 License

This project is for educational and demonstration purposes. Please refer to the license file for usage terms.

---

**Built with ❤️ using Python, Streamlit, and Scikit-learn**
