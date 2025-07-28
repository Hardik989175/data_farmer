import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

print("--- Final Model Training and Prediction Pipeline (V2 - Corrected Scaling) ---")

# --- 1. Load All Necessary Data ---
print("Step 1: Loading all data sources...")
train_df = pd.read_csv('final_pipeline/preprocessed_data/train_with_external_data.csv')
test_ultimate_df = pd.read_csv('final_pipeline/preprocessed_data/test_ultimate.csv')
raw_test_df = pd.read_csv('dataset/LTF Challenge data with dictionary.xlsx - TestData.csv')
external_df = pd.read_csv('final_pipeline/external_data/commodity_prices/dataset/India_Key_Commodities_Retail_Prices_1997_2015_cleaned.csv')

TARGET = 'Target_Variable/Total Income'

# --- 2. Process Test Data to Match Training Data ---
print("Step 2: Applying external data feature engineering to the test set...")
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

# --- 3. Prepare Final Data for Modeling ---
print("Step 3: Preparing final dataframes for model training...")
X_train = train_df.drop(columns=[TARGET, 'FarmerID'])
y_train = train_df[TARGET]
X_test = test_df_final[X_train.columns]

# --- 4. CRITICAL FIX: Ensure only numeric columns before scaling ---
print("Step 4: Ensuring all columns are numeric before scaling...")
X_train = X_train.select_dtypes(include=np.number)
X_test = X_test.select_dtypes(include=np.number)


# --- 5. Scale all numerical features together ---
print("Step 5: Scaling all numerical features together...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to dataframes
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# --- 6. Define and Train the Final Stacked Model ---
print("Step 6: Defining and training the final stacked ensemble model...")
estimators = [
    ('cat', CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=5, verbose=0, random_seed=42)),
    ('xgb', XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, gamma=1, random_state=42))
]
final_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0)
)
final_model.fit(X_train, y_train)


# --- 7. Generate Predictions and Create Submission File ---
print("Step 7: Generating predictions on the test set...")
final_predictions_log = final_model.predict(X_test)
final_predictions_original = np.expm1(final_predictions_log)
final_predictions_original[final_predictions_original < 0] = 0

print("Step 8: Creating the corrected submission file...")
submission_df = pd.DataFrame({
    'FarmerID': raw_test_df['FarmerID'],
    'Target_Variable/Total Income': final_predictions_original
})
submission_df.to_csv('submission_v2_corrected.csv', index=False)

print("\n--- Pipeline Complete ---")
print("Corrected submission file 'submission_v2_corrected.csv' has been created successfully.") 