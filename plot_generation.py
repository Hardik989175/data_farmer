# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBRegressor

# Set the name of the target variable to be predicted
TARGET = 'Target_Variable/Total Income'

def setup_directories():
    """
    Creates all the necessary directory folders to save the generated plots
    corresponding to each stage of the data science workflow.
    """
    print("Setting up plot directories")
    os.makedirs('plots/01_raw_data_analysis', exist_ok=True)
    os.makedirs('plots/02_data_cleaning', exist_ok=True)
    os.makedirs('plots/03_feature_engineering', exist_ok=True)
    os.makedirs('plots/04_outlier_handling', exist_ok=True)
    os.makedirs('plots/05_target_transformation', exist_ok=True)
    os.makedirs('plots/06_final_results', exist_ok=True)

def generate_raw_data_plots(raw_train_df):
    """
    Generates and saves visualizations for raw (uncleaned) training data:
    - Distribution of the target variable
    - Histograms of numerical features
    - Heatmap showing missing values
    - Correlation matrix among numerical variables
    """
    print("Generating plots for raw data analysis")
    
    # Plot and save the distribution of the target variable before transformation
    
    plt.figure(figsize=(10, 6))
    sns.histplot(raw_train_df[TARGET], kde=True)
    plt.title('Target Variable Distribution (Before Transformation)')
    plt.xlabel('Total Income')
    plt.ylabel('Frequency')
    plt.savefig('plots/01_raw_data_analysis/target_variable_distribution.png')
    plt.close()

    # Plot distributions for all numerical columns (except the target variable)
    
    numerical_features = raw_train_df.select_dtypes(include=np.number).columns.tolist()
    numerical_features.remove(TARGET)
    raw_train_df[numerical_features].hist(bins=20, figsize=(20, 15))
    plt.suptitle('Distribution of Numerical Features')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('plots/01_raw_data_analysis/numerical_features_distribution.png')
    plt.close()

    # Visualize the pattern of missing data across the dataset
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(raw_train_df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.savefig('plots/01_raw_data_analysis/missing_values_heatmap.png')
    plt.close()

    # Display a correlation heatmap to identify how numerical features relate to each other
    
    corr_df = raw_train_df.select_dtypes(include=np.number)
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr_df.corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('plots/01_raw_data_analysis/correlation_matrix.png')
    plt.close()

def generate_cleaning_plots(raw_train_df):
    """
    Visualizes the effect of data cleaning, specifically:
    - Cleans a temperature column that originally contains string values
    - Plots temperature before and after cleaning (as numeric values)
    """
    print("Generating plots for data cleaning")
    temp_col_original = 'K022-Ambient temperature (min & max)'
    
    if temp_col_original in raw_train_df.columns:
        # Extract the min temperature from a string like "24/37"
        df_temp_before = raw_train_df[[temp_col_original]].copy().dropna()
        df_temp_before['min_temp_before'] = pd.to_numeric(df_temp_before[temp_col_original].str.split('/', expand=True)[0], errors='coerce')
        
        # Plot min temperature before and after conversion
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        sns.histplot(df_temp_before['min_temp_before'], ax=axes[0], kde=True, bins=20)
        axes[0].set_title('Before: Min Temperature (from string)')
        sns.histplot(df_temp_before['min_temp_before'], ax=axes[1], kde=True, bins=20)
        axes[1].set_title('After: Min Temperature (as numeric)')
        plt.suptitle('Effect of Temperature Column Cleaning')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('plots/02_data_cleaning/temperature_cleaning_effect.png')
        plt.close()

def generate_feature_engineering_plots(raw_train_df):
    """
    Visualizes the impact of an engineered feature (interaction term)
    - Multiplies land area with population density to create a new feature
    - Plots how this engineered feature relates to income
    """
    print("Generating plots for feature engineering")
    if 'Total_Land_For_Agriculture' in raw_train_df.columns and 'L009-Population density' in raw_train_df.columns:
        df_fe = raw_train_df.copy()
        df_fe['land_x_density'] = df_fe['Total_Land_For_Agriculture'] * df_fe['L009-Population density']
        
        # Scatterplot to observe trend between new feature and target
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_fe, x='land_x_density', y=TARGET, alpha=0.5)
        plt.title('Engineered Feature: Land Area x Population Density vs. Income')
        plt.xlabel('Land Area * Population Density')
        plt.ylabel('Total Income')
        plt.savefig('plots/03_feature_engineering/land_x_density_visualization.png')
        plt.close()

def generate_outlier_plots(raw_train_df):
    """
    Demonstrates the effect of outlier handling (capping/extreme value clipping)
    - Caps the 'Total_Land_For_Agriculture' feature at 1st and 99th percentiles
    - Shows before and after boxplots
    """
    print("Generating plots for outlier handling")
    feature_to_cap = 'Total_Land_For_Agriculture'
    
    if feature_to_cap in raw_train_df.columns:
        q1 = raw_train_df[feature_to_cap].quantile(0.01)
        q3 = raw_train_df[feature_to_cap].quantile(0.99)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.boxplot(y=raw_train_df[feature_to_cap], ax=axes[0])
        axes[0].set_title(f'Before Capping: {feature_to_cap}')
        
        # Clip the outliers
        df_capped = raw_train_df.copy()
        df_capped[feature_to_cap] = np.clip(df_capped[feature_to_cap], q1, q3)
        
        sns.boxplot(y=df_capped[feature_to_cap], ax=axes[1])
        axes[1].set_title(f'After Capping (1% and 99%): {feature_to_cap}')
        plt.suptitle('Effect of Outlier Capping')
        plt.savefig('plots/04_outlier_handling/outlier_capping_effect.png')
        plt.close()

def generate_transformation_plots(raw_train_df):
    """
    Plots the impact of log transformation on the target variable:
    - Helps stabilize variance and normalize distribution for regression
    """
    print("Generating plots for target transformation")
    y_train_log = np.log1p(raw_train_df[TARGET])  # log(1 + y) to handle 0s safely

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(raw_train_df[TARGET], kde=True, ax=axes[0], bins=30)
    axes[0].set_title('Original Target Distribution')
    sns.histplot(y_train_log, kde=True, ax=axes[1], bins=30)
    axes[1].set_title('Log-Transformed Target Distribution')
    plt.suptitle('Effect of Log Transformation on Target Variable')
    plt.savefig('plots/05_target_transformation/log_transform_effect.png')
    plt.close()

def generate_final_results_plots(final_model, X_train, y_train):
    """
    Evaluates final model by visualizing:
    - Feature importance (top 20)
    - Actual vs. Predicted values on training data
    """
    print("Generating final results plots")
    
    # --- Feature Importance Plot ---
    print("Generating Feature Importance plot")
    importance_model = XGBRegressor(
        n_estimators=1500, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, gamma=1, random_state=42
    )
    importance_model.fit(X_train, y_train)
    
    # Get top features based on importance
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importance_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Bar plot of top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importances.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features in Final Model')
    plt.tight_layout()
    plt.savefig('plots/06_final_results/feature_importance.png')
    plt.close()

    # Actual vs Predicted Plot
    print("Generating Actual vs. Predicted plot")
    train_preds_log = final_model.predict(X_train)
    train_preds_original = np.expm1(train_preds_log)  # Convert back from log space
    y_train_original = np.expm1(y_train)
    
    results_df = pd.DataFrame({'Actual': y_train_original, 'Predicted': train_preds_original})
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=results_df, x='Actual', y='Predicted', alpha=0.5)
    plt.plot(
        [results_df['Actual'].min(), results_df['Actual'].max()],
        [results_df['Actual'].min(), results_df['Actual'].max()],
        '--r', linewidth=2
    )
    plt.title('Actual vs. Predicted Income on Full Training Set')
    plt.xlabel('Actual Income')
    plt.ylabel('Predicted Income')
    plt.tight_layout()
    plt.savefig('plots/06_final_results/actual_vs_predicted.png')
    plt.close()
