"""
Smart Energy Consumption ML Models
Implements various ML algorithms for energy consumption prediction and optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score
)
import pickle
import warnings
warnings.filterwarnings('ignore')

class EnergyMLModels:
    """
    Comprehensive ML models for energy consumption analysis
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        self.target_encoder = None
        
    def preprocess_data(self, df, target_column='Energy_Consumption'):
        """
        Preprocess the dataset for ML models
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Target variable name
            
        Returns:
            tuple: (X_processed, y_processed, preprocessed_df)
        """
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Identify categorical and numerical columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from numerical columns if present
        if target_column in numerical_columns:
            numerical_columns.remove(target_column)
        
        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Prepare features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y, df_processed
    
    def create_classification_targets(self, y):
        """
        Create classification targets from regression target
        
        Args:
            y (pd.Series): Energy consumption values
            
        Returns:
            tuple: (high_usage_binary, efficiency_category)
        """
        # High Usage Classification (Yes/No) - above 75th percentile
        threshold = y.quantile(0.75)
        high_usage = (y > threshold).astype(int)  # 1 for Yes, 0 for No
        
        # Efficiency Category (Low/Medium/High)
        # Low: Bottom 33%, Medium: Middle 34%, High: Top 33%
        low_threshold = y.quantile(0.33)
        high_threshold = y.quantile(0.67)
        
        efficiency_category = []
        for val in y:
            if val <= low_threshold:
                efficiency_category.append('Low')
            elif val <= high_threshold:
                efficiency_category.append('Medium')
            else:
                efficiency_category.append('High')
        
        return high_usage, efficiency_category
    
    def train_linear_regression(self, X, y, test_size=0.2, random_state=42):
        """
        Train Linear Regression model for energy consumption prediction
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Test set proportion
            random_state (int): Random state
            
        Returns:
            dict: Model results and metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        results = {
            'model': model,
            'model_type': 'Linear Regression',
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            },
            'metrics': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            },
            'feature_importance': feature_importance,
            'data_split': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        }
        
        self.models['linear_regression'] = results
        return results
    
    def train_decision_tree(self, X, y, test_size=0.2, random_state=42):
        """
        Train Decision Tree Classifier for high usage prediction
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target (will be converted to binary)
            test_size (float): Test set proportion
            random_state (int): Random state
            
        Returns:
            dict: Model results and metrics
        """
        # Create binary target
        y_binary, _ = self.create_classification_targets(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )
        
        # Train model
        model = DecisionTreeClassifier(random_state=random_state, max_depth=10)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y_binary, cv=5, scoring='accuracy')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Classification Report
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results = {
            'model': model,
            'model_type': 'Decision Tree Classifier',
            'target_type': 'High Usage (Binary)',
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            },
            'metrics': {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': cm,
                'classification_report': class_report
            },
            'feature_importance': feature_importance,
            'data_split': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        }
        
        self.models['decision_tree'] = results
        return results
    
    def train_knn(self, X, y, test_size=0.2, random_state=42, n_neighbors=5):
        """
        Train KNN Classifier for efficiency category prediction
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target (will be converted to categories)
            test_size (float): Test set proportion
            random_state (int): Random state
            n_neighbors (int): Number of neighbors
            
        Returns:
            dict: Model results and metrics
        """
        # Create categorical target
        _, y_category = self.create_classification_targets(y)
        
        # Encode target categories
        if self.target_encoder is None:
            self.target_encoder = LabelEncoder()
        y_category_encoded = self.target_encoder.fit_transform(y_category)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_category_encoded, test_size=test_size, 
            random_state=random_state, stratify=y_category_encoded
        )
        
        # Train model
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y_category_encoded, cv=5, scoring='accuracy')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Classification Report
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        results = {
            'model': model,
            'model_type': 'KNN Classifier',
            'target_type': 'Efficiency Category',
            'target_classes': self.target_encoder.classes_.tolist(),
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            },
            'metrics': {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'confusion_matrix': cm,
                'classification_report': class_report
            },
            'data_split': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        }
        
        self.models['knn'] = results
        return results
    
    def train_kmeans(self, X, n_clusters=3, random_state=42):
        """
        Train K-Means for consumption clustering
        
        Args:
            X (pd.DataFrame): Features
            n_clusters (int): Number of clusters
            random_state (int): Random state
            
        Returns:
            dict: Clustering results and metrics
        """
        # Train model
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = model.fit_predict(X)
        
        # Metrics
        silhouette_avg = silhouette_score(X, cluster_labels)
        inertia = model.inertia_
        
        # Cluster centers
        cluster_centers = pd.DataFrame(
            model.cluster_centers_,
            columns=self.feature_names
        )
        
        # Add cluster labels to original data
        X_with_clusters = X.copy()
        X_with_clusters['Cluster'] = cluster_labels
        
        # Cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_data = X_with_clusters[X_with_clusters['Cluster'] == i]
            cluster_stats[f'Cluster_{i}'] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(X)) * 100,
                'mean_values': cluster_data[self.feature_names].mean().to_dict()
            }
        
        results = {
            'model': model,
            'model_type': 'K-Means Clustering',
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'metrics': {
                'silhouette_score': silhouette_avg,
                'inertia': inertia
            },
            'cluster_centers': cluster_centers,
            'cluster_statistics': cluster_stats,
            'data_with_clusters': X_with_clusters
        }
        
        self.models['kmeans'] = results
        return results
    
    def predict_consumption(self, input_data):
        """
        Predict energy consumption using trained linear regression model
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            
        Returns:
            float: Predicted energy consumption
        """
        if 'linear_regression' not in self.models:
            raise ValueError("Linear Regression model not trained. Call train_linear_regression first.")
        
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value
        
        # Scale features
        input_scaled = self.scaler.transform(input_df[self.feature_names])
        
        # Predict
        model = self.models['linear_regression']['model']
        prediction = model.predict(input_scaled)[0]
        
        return max(0, prediction)  # Ensure non-negative
    
    def predict_high_usage(self, input_data):
        """
        Predict high usage classification using trained decision tree model
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            
        Returns:
            tuple: (prediction, probability)
        """
        if 'decision_tree' not in self.models:
            raise ValueError("Decision Tree model not trained. Call train_decision_tree first.")
        
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Scale features
        input_scaled = self.scaler.transform(input_df[self.feature_names])
        
        # Predict
        model = self.models['decision_tree']['model']
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        result = "Yes" if prediction == 1 else "No"
        return result, probability
    
    def predict_efficiency_category(self, input_data):
        """
        Predict efficiency category using trained KNN model
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            
        Returns:
            tuple: (prediction, probability)
        """
        if 'knn' not in self.models:
            raise ValueError("KNN model not trained. Call train_knn first.")
        
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Scale features
        input_scaled = self.scaler.transform(input_df[self.feature_names])
        
        # Predict
        model = self.models['knn']['model']
        prediction_encoded = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Decode prediction
        prediction = self.target_encoder.inverse_transform([prediction_encoded])[0]
        
        return prediction, probability
    
    def save_models(self, filepath='energy_models.pkl'):
        """
        Save all trained models to file
        
        Args:
            filepath (str): Path to save models
        """
        model_data = {
            'models': self.models,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_encoder': self.target_encoder
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='energy_models.pkl'):
        """
        Load trained models from file
        
        Args:
            filepath (str): Path to load models from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.label_encoders = model_data['label_encoders']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_encoder = model_data['target_encoder']
        
        print(f"Models loaded from {filepath}")
    
    def get_model_summary(self):
        """
        Get summary of all trained models
        
        Returns:
            dict: Model summary
        """
        summary = {}
        
        for model_name, model_data in self.models.items():
            summary[model_name] = {
                'type': model_data['model_type'],
                'metrics': model_data['metrics']
            }
        
        return summary

def main():
    """
    Main function to test all models
    """
    from data_generator import EnergyDataGenerator
    
    print("Testing Energy ML Models...")
    
    # Generate sample data
    generator = EnergyDataGenerator(num_records=1000)
    df = generator.generate_dataset()
    
    # Initialize models
    ml_models = EnergyMLModels()
    
    # Preprocess data
    X, y, df_processed = ml_models.preprocess_data(df)
    
    # Train all models
    print("\nTraining Linear Regression...")
    lr_results = ml_models.train_linear_regression(X, y)
    print(f"Test R2 Score: {lr_results['metrics']['test_r2']:.4f}")
    
    print("\nTraining Decision Tree...")
    dt_results = ml_models.train_decision_tree(X, y)
    print(f"Test Accuracy: {dt_results['metrics']['test_accuracy']:.4f}")
    
    print("\nTraining KNN...")
    knn_results = ml_models.train_knn(X, y)
    print(f"Test Accuracy: {knn_results['metrics']['test_accuracy']:.4f}")
    
    print("\nTraining K-Means...")
    km_results = ml_models.train_kmeans(X)
    print(f"Silhouette Score: {km_results['metrics']['silhouette_score']:.4f}")
    
    # Test predictions
    sample_input = {
        'Household_Size': 3,
        'Appliance_Count': 12,
        'Average_Temperature': 22,
        'Working_Hours': 8,
        'Solar_Usage': 'Yes',
        'Electricity_Tariff': 'Medium',
        'Day_Type': 'Weekday',
        'Previous_Consumption': 25.5
    }
    
    print(f"\nTesting predictions with sample input: {sample_input}")
    
    consumption = ml_models.predict_consumption(sample_input)
    print(f"Predicted Consumption: {consumption:.2f} kWh")
    
    high_usage, prob = ml_models.predict_high_usage(sample_input)
    print(f"High Usage Prediction: {high_usage} (Probability: {prob[1]:.2f})")
    
    efficiency, prob = ml_models.predict_efficiency_category(sample_input)
    print(f"Efficiency Category: {efficiency} (Probability: {max(prob):.2f})")
    
    # Save models
    ml_models.save_models()
    
    print("\nAll models trained and saved successfully!")

if __name__ == "__main__":
    main()
