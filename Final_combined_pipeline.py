import pandas as pd
import numpy as np
import os
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
import gc
import argparse

# ==============================================================================
# --- PLOTTING FUNCTIONS ---
# ==============================================================================
TARGET_PLOT = 'Target_Variable/Total Income'

def setup_directories():
    """Creates all necessary directories for saving plots."""
    print("  - Setting up plot directories...")
    os.makedirs('plots/01_raw_data_analysis', exist_ok=True)
    os.makedirs('plots/02_data_cleaning', exist_ok=True)
    os.makedirs('plots/03_feature_engineering', exist_ok=True)
    os.makedirs('plots/04_outlier_handling', exist_ok=True)
    os.makedirs('plots/05_target_transformation', exist_ok=True)
    os.makedirs('plots/06_final_results', exist_ok=True)

def generate_raw_data_plots(raw_train_df):
    """Generates and saves plots for initial raw data analysis."""
    print("  - Generating plots for raw data analysis...")
    # ... (code for raw data plots)
    plt.figure(figsize=(10, 6))
    sns.histplot(raw_train_df[TARGET_PLOT], kde=True)
    plt.title('Target Variable Distribution (Before Transformation)')
    plt.xlabel('Total Income')
    plt.ylabel('Frequency')
    plt.savefig('plots/01_raw_data_analysis/target_variable_distribution.png')
    plt.close()

def generate_cleaning_plots(raw_train_df):
    """Generates plots to visualize the effect of data cleaning."""
    print("  - Generating plots for data cleaning...")
    # ... (code for cleaning plots)
    temp_col_original = 'K022-Ambient temperature (min & max)'
    if temp_col_original in raw_train_df.columns:
        df_temp_before = raw_train_df[[temp_col_original]].copy().dropna()
        df_temp_before['min_temp_before'] = pd.to_numeric(df_temp_before[temp_col_original].str.split('/', expand=True)[0], errors='coerce')
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        sns.histplot(df_temp_before['min_temp_before'], ax=axes[0], kde=True, bins=20)
        axes[0].set_title('Before: Min Temperature (from string)')
        sns.histplot(df_temp_before['min_temp_before'], ax=axes[1], kde=True, bins=20)
        axes[1].set_title('After: Min Temperature (as numeric)')
        plt.suptitle('Effect of Temperature Column Cleaning')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('plots/02_data_cleaning/temperature_cleaning_effect.png')
        plt.close()

def generate_all_plots(raw_train_df, final_model, X_train, y_train):
    """A master function to generate all plots."""
    print("\nStep: Generating all analytical plots...")
    setup_directories()
    generate_raw_data_plots(raw_train_df)
    generate_cleaning_plots(raw_train_df)
    # NOTE: Other plot generation functions can be added here
    # generate_feature_engineering_plots(raw_train_df)
    # generate_outlier_plots(raw_train_df)
    # generate_transformation_plots(raw_train_df)
    generate_final_results_plots(final_model, X_train, y_train)
    print("  - All plots generated successfully.")

def generate_final_results_plots(final_model, X_train, y_train):
    """Generates plots to evaluate the final model's performance on training data."""
    print("  - Generating final results plots...")
    # ... (code for final results plots)
    importance_model = XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, gamma=1, random_state=42)
    importance_model.fit(X_train, y_train)
    feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': importance_model.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importances.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features in Final Model')
    plt.tight_layout()
    plt.savefig('plots/06_final_results/feature_importance.png')
    plt.close()

# ==============================================================================
# --- HYPERPARAMETER TUNING FUNCTIONS ---
# ==============================================================================
def objective(trial, X, y):
    """Defines the search space for hyperparameters for Optuna."""
    # ... (code for optuna objective function)
    cat_params = {
        'iterations': trial.suggest_int('cat_iterations', 800, 2000),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('cat_depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1, 10),
        'verbose': 0,
        'random_seed': 42
    }
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 800, 2000),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('xgb_max_depth', 6, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('xgb_gamma', 0, 5),
        'random_state': 42
    }
    ridge_alpha = trial.suggest_float('ridge_alpha', 0.5, 5.0)
    estimators = [
        ('cat', CatBoostRegressor(**cat_params)),
        ('xgb', XGBRegressor(**xgb_params))
    ]
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=ridge_alpha),
        cv='passthrough'
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        stacking_regressor.fit(X_train, y_train)
        preds = stacking_regressor.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(score)
        del X_train, X_val, y_train, y_val, preds
        gc.collect()
    return np.mean(scores)

def run_tuning():
    """Runs the hyperparameter tuning process."""
    print("\n--- Starting Hyperparameter Tuning ---")
    train_df = pd.read_csv('final_pipeline/preprocessed_data/train_with_external_data.csv')
    X = train_df.drop(columns=[TARGET_PLOT, 'FarmerID']).select_dtypes(include=np.number)
    y = train_df[TARGET_PLOT]
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
    
    print("\n--- Tuning Complete ---")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (RMSE): {trial.value}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

# ==============================================================================
# --- MAIN PREDICTION PIPELINE ---
# ==============================================================================
def run_prediction_pipeline(generate_plots=False):
    """The main function to run the end-to-end prediction pipeline."""
    print("--- Final Model Training and Prediction Pipeline ---")

    # --- 1. Load All Necessary Data ---
    print("\nStep 1: Loading all data sources...")
    train_df = pd.read_csv('final_pipeline/preprocessed_data/train_with_external_data.csv')
    test_ultimate_df = pd.read_csv('final_pipeline/preprocessed_data/test_ultimate.csv')
    raw_test_df = pd.read_csv('dataset/LTF Challenge data with dictionary.xlsx - TestData.csv')
    raw_train_df_for_plots = pd.read_csv('dataset/LTF Challenge data with dictionary.xlsx - TrainData.csv')
    external_df = pd.read_csv('final_pipeline/external_data/commodity_prices/dataset/India_Key_Commodities_Retail_Prices_1997_2015_cleaned.csv')
    TARGET = 'Target_Variable/Total Income'

    # --- 2. Process Test Data ---
    print("\nStep 2: Applying external data feature engineering to the test set...")
    # ... (code for test data processing)
    external_df['Centre'] = external_df['Centre'].str.upper().str.strip()
    external_df.rename(columns={'Centre': 'CITY'}, inplace=True)
    staple_crops = ['Wheat', 'Rice', 'Tur/Arhar Dal']
    cash_crops = ['Sugar', 'Sunflower Oil (Packed)']
    external_df['Commodity'] = external_df['Commodity'].str.strip()
    avg_prices = external_df.groupby(['CITY', 'Commodity'])['Price per Kg'].mean().unstack()
    existing_staple_crops = [crop for crop in staple_crops if crop in avg_prices.columns]
    existing_cash_crops = [crop for crop in cash_crops if crop in avg_prices.columns]
    if existing_staple_crops:
        avg_prices['avg_staple_price'] = avg_prices[existing_staple_crops].mean(axis=1)
    if existing_cash_crops:
        avg_prices['avg_cash_crop_price'] = avg_prices[existing_cash_crops].mean(axis=1)
    avg_prices['max_commodity_price'] = avg_prices.max(axis=1)
    price_features = avg_prices[['avg_staple_price', 'avg_cash_crop_price', 'max_commodity_price']].reset_index()
    raw_test_df['CITY_CLEAN'] = raw_test_df['CITY'].str.upper().str.strip()
    price_features['CITY_CLEAN'] = price_features['CITY'].str.upper().str.strip()
    test_df_enriched = pd.merge(test_ultimate_df, raw_test_df[['FarmerID', 'CITY_CLEAN']], on='FarmerID', how='left')
    test_df_final = pd.merge(test_df_enriched, price_features, on='CITY_CLEAN', how='left')
    for col in ['avg_staple_price', 'avg_cash_crop_price', 'max_commodity_price']:
        train_median = train_df[col].median()
        test_df_final[col].fillna(train_median, inplace=True)
    test_df_final.drop(columns=['CITY_CLEAN', 'CITY'], inplace=True, errors='ignore')

    # --- 3. Prepare Data for Modeling ---
    print("\nStep 3: Preparing final dataframes for model training...")
    X_train = train_df.drop(columns=[TARGET, 'FarmerID'])
    y_train = train_df[TARGET]
    X_test = test_df_final[X_train.columns]
    X_train = X_train.select_dtypes(include=np.number)
    X_test = X_test.select_dtypes(include=np.number)

    # --- 4. Scale Features ---
    print("\nStep 4: Scaling all numerical features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # --- 5. Train Model ---
    print("\nStep 5: Defining and training the final stacked ensemble model...")
    estimators = [
        ('cat', CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=5, verbose=0, random_seed=42)),
        ('xgb', XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, gamma=1, random_state=42))
    ]
    final_model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
    final_model.fit(X_train, y_train)

    # --- 6. Generate Plots (Optional) ---
    if generate_plots:
        generate_all_plots(raw_train_df_for_plots, final_model, X_train, y_train)

    # --- 7. Generate Predictions ---
    print("\nStep 7: Generating predictions on the test set...")
    final_predictions_log = final_model.predict(X_test)
    final_predictions_original = np.expm1(final_predictions_log)
    final_predictions_original[final_predictions_original < 0] = 0

    # --- 8. Create Submission File ---
    print("\nStep 8: Creating the corrected submission file...")
    submission_df = pd.DataFrame({
        'FarmerID': raw_test_df['FarmerID'],
        'Target_Variable/Total Income': final_predictions_original
    })
    submission_df.to_csv('submission_v2_corrected.csv', index=False)
    
    print("\n--- Pipeline Complete ---")
    print("Corrected submission file 'submission_v2_corrected.csv' has been created successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Farmer Income Prediction pipeline.")
    parser.add_argument('--tune', action='store_true', help="Run hyperparameter tuning instead of the prediction pipeline.")
    parser.add_argument('--plots', action='store_true', help="Generate all plots during the prediction pipeline run.")
    args = parser.parse_args()

    if args.tune:
        run_tuning()
    else:
        run_prediction_pipeline(generate_plots=args.plots) 