"""
Smart Energy Consumption Data Generator
Generates synthetic dataset for energy consumption prediction and optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class EnergyDataGenerator:
    """
    Generates synthetic energy consumption data with realistic patterns
    """
    
    def __init__(self, num_records=1000):
        self.num_records = num_records
        self.np_random = np.random.RandomState(42)
        
    def generate_dataset(self):
        """
        Generate complete synthetic dataset with all required features
        
        Returns:
            pd.DataFrame: Generated dataset
        """
        data = {}
        
        # Household Size (1-6 people)
        data['Household_Size'] = self.np_random.randint(1, 7, self.num_records)
        
        # Appliance Count (5-25 appliances)
        data['Appliance_Count'] = self.np_random.randint(5, 26, self.num_records)
        
        # Average Temperature (-5 to 40 Celsius)
        data['Average_Temperature'] = self.np_random.uniform(-5, 40, self.num_records)
        
        # Working Hours (0-12 hours)
        data['Working_Hours'] = self.np_random.randint(0, 13, self.num_records)
        
        # Solar Usage (Yes/No)
        solar_prob = 0.3  # 30% households have solar
        data['Solar_Usage'] = np.where(
            self.np_random.random(self.num_records) < solar_prob, 'Yes', 'No'
        )
        
        # Electricity Tariff (Low/Medium/High)
        tariff_choices = ['Low', 'Medium', 'High']
        tariff_probs = [0.3, 0.5, 0.2]
        data['Electricity_Tariff'] = self.np_random.choice(
            tariff_choices, self.num_records, p=tariff_probs
        )
        
        # Day Type (Weekday/Weekend)
        day_prob = 0.714  # 5/7 are weekdays
        data['Day_Type'] = np.where(
            self.np_random.random(self.num_records) < day_prob, 'Weekday', 'Weekend'
        )
        
        # Previous Consumption (base consumption in kWh)
        base_consumption = self._calculate_base_consumption(data)
        data['Previous_Consumption'] = base_consumption
        
        # Target: Energy Consumption (kWh)
        data['Energy_Consumption'] = self._calculate_energy_consumption(data)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add some realistic variations and noise
        df = self._add_realistic_variations(df)
        
        return df
    
    def _calculate_base_consumption(self, data):
        """
        Calculate base previous consumption based on household characteristics
        
        Args:
            data (dict): Dictionary with feature arrays
            
        Returns:
            np.array: Previous consumption values
        """
        # Base consumption formula considering household size and appliances
        base = (
            data['Household_Size'] * 2.5 +  # Base per person
            data['Appliance_Count'] * 0.8 +  # Base per appliance
            np.abs(data['Average_Temperature'] - 20) * 0.3  # Temperature deviation from optimal
        )
        
        # Add random variation
        noise = self.np_random.normal(0, 2, self.num_records)
        previous_consumption = base + noise
        
        # Ensure positive values
        previous_consumption = np.maximum(previous_consumption, 1)
        
        return np.round(previous_consumption, 2)
    
    def _calculate_energy_consumption(self, data):
        """
        Calculate target energy consumption based on all features
        
        Args:
            data (dict): Dictionary with feature arrays
            
        Returns:
            np.array: Energy consumption values
        """
        # Start with previous consumption as base
        consumption = data['Previous_Consumption'].copy()
        
        # Temperature effect (heating/cooling)
        temp_effect = np.where(
            data['Average_Temperature'] < 10,
            (10 - data['Average_Temperature']) * 0.5,  # Heating
            np.where(
                data['Average_Temperature'] > 25,
                (data['Average_Temperature'] - 25) * 0.4,  # Cooling
                0
            )
        )
        
        # Working hours effect
        work_effect = data['Working_Hours'] * 0.3
        
        # Solar usage reduction
        solar_reduction = np.where(
            data['Solar_Usage'] == 'Yes',
            -consumption * 0.15,  # 15% reduction with solar
            0
        )
        
        # Tariff effect (higher tariff might encourage conservation)
        tariff_effect = np.where(
            data['Electricity_Tariff'] == 'High',
            -consumption * 0.1,
            np.where(
                data['Electricity_Tariff'] == 'Low',
                consumption * 0.05,
                0
            )
        )
        
        # Day type effect
        day_effect = np.where(
            data['Day_Type'] == 'Weekend',
            consumption * 0.1,  # 10% more on weekends
            0
        )
        
        # Combine all effects
        consumption = (
            consumption + temp_effect + work_effect + 
            solar_reduction + tariff_effect + day_effect
        )
        
        # Add random noise
        noise = self.np_random.normal(0, 1.5, self.num_records)
        consumption = consumption + noise
        
        # Ensure positive values and reasonable range
        consumption = np.maximum(consumption, 0.5)
        consumption = np.minimum(consumption, 100)  # Cap at 100 kWh
        
        return np.round(consumption, 2)
    
    def _add_realistic_variations(self, df):
        """
        Add realistic variations and correlations to the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Modified dataframe with realistic variations
        """
        # Add correlation between household size and appliance count
        for i in range(len(df)):
            if df.loc[i, 'Household_Size'] >= 4 and df.loc[i, 'Appliance_Count'] < 10:
                df.loc[i, 'Appliance_Count'] = self.np_random.randint(10, 20)
            elif df.loc[i, 'Household_Size'] <= 2 and df.loc[i, 'Appliance_Count'] > 15:
                df.loc[i, 'Appliance_Count'] = self.np_random.randint(5, 15)
        
        # Add seasonal patterns to temperature
        seasonal_adjustment = np.sin(np.linspace(0, 4*np.pi, len(df))) * 5
        df['Average_Temperature'] = df['Average_Temperature'] + seasonal_adjustment
        df['Average_Temperature'] = np.round(df['Average_Temperature'], 1)
        
        return df
    
    def save_dataset(self, df, filename='energy_consumption_data.csv'):
        """
        Save dataset to CSV file
        
        Args:
            df (pd.DataFrame): Dataset to save
            filename (str): Output filename
        """
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
    
    def get_data_summary(self, df):
        """
        Get summary statistics of the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_records': len(df),
            'features': list(df.columns),
            'numerical_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'statistics': df.describe().to_dict()
        }
        
        return summary

def main():
    """
    Main function to generate and save dataset
    """
    print("Generating Smart Energy Consumption Dataset...")
    
    # Initialize generator
    generator = EnergyDataGenerator(num_records=1000)
    
    # Generate dataset
    df = generator.generate_dataset()
    
    # Save dataset
    generator.save_dataset(df)
    
    # Print summary
    summary = generator.get_data_summary(df)
    print(f"\nDataset Summary:")
    print(f"Total Records: {summary['total_records']}")
    print(f"Numerical Features: {summary['numerical_features']}")
    print(f"Categorical Features: {summary['categorical_features']}")
    
    # Display first few rows
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    df = main()
