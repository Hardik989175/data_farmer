# Farmer Income Prediction: Project Pipeline and Analysis

This project aims to predict the total income of farmers based on a wide range of agricultural, economic, and demographic data. This document outlines the end-to-end pipeline used for data processing, model training, and prediction, as well as an explanation of the various analytical plots generated along the way.

---

## The Prediction Pipeline

The core logic is contained within the `generate_final_predictions_v2.py` script. It executes a series of steps to ensure data consistency, train a robust model, and generate the final predictions.

### Step 1: Data Loading
The pipeline begins by loading all necessary datasets:
- **Raw Training Data**: The original, unprocessed training data used for generating analytical plots.
- **Preprocessed Training Data**: Training data that has already been cleaned, feature-engineered, and includes external data.
- **Preprocessed Test Data**: The test set that has undergone the same preprocessing steps as the training set.
- **Raw Test Data**: The original test set, used primarily to retrieve `FarmerID` for the final submission.
- **External Commodity Data**: External data on commodity prices, which is merged to enrich the feature set.

### Step 2: Generating Analytical Plots
Before the main preprocessing and modeling, the script generates a series of plots to visualize the data at different stages. This is crucial for understanding the data and justifying the methods used. A detailed explanation of each plot is provided in the next section.

### Step 3: Final Test Data Processing
To ensure the test data matches the training data, the external commodity price features are merged with the test set. Any missing values that result from this merge are filled using the median value from the training data to prevent data leakage.

### Step 4: Preparing Data for Modeling
The data is split into features (`X_train`, `X_test`) and the target variable (`y_train`). The target variable is `Target_Variable/Total Income`.

### Step 5 & 6: Feature Scaling
All numerical features in both the training and test sets are scaled using `StandardScaler`. This standardizes features to have a mean of 0 and a standard deviation of 1, which is essential for the performance of many machine learning models, including the `Ridge` regression used in the final step of our model.

### Step 7: Final Model Training
A `StackingRegressor` is used as the final model. Stacking is an ensemble technique that combines the predictions of multiple base models to produce a final, more powerful prediction.
- **Base Models**: `CatBoostRegressor` and `XGBRegressor`, two powerful gradient boosting models.
- **Final Estimator**: `Ridge` regression, a linear model that takes the predictions of the base models as input and computes the final prediction.

### Step 8 & 9: Final Results and Submission
The trained model is used to:
1.  Generate final evaluation plots (`feature_importance.png` and `actual_vs_predicted.png`).
2.  Predict the target variable on the processed test set.
3.  Create the final `submission_v2_corrected.csv` file with the `FarmerID` and the predicted income.

---

## Explanation of Generated Plots

The plots are organized into directories corresponding to the different stages of the data analysis pipeline.

### `01_raw_data_analysis`
These plots provide a first look at the raw training data before any significant transformations.
- **`correlation_matrix.png`**: Shows the correlation between all numerical features. It helps identify multicollinearity and which features are most strongly correlated with the target variable.
- **`missing_values_heatmap.png`**: Provides a visual map of missing data across the entire dataset. This is critical for deciding on an imputation strategy.
- **`numerical_features_distribution.png`**: Displays histograms for every numerical feature, which is useful for understanding the scale, skewness, and underlying distribution of each variable.
- **`target_variable_distribution.png`**: Shows the distribution of the farmer's income. In this project, it's highly skewed, which justifies the use of a log transformation.

### `02_data_cleaning`
This section visualizes the impact of specific data cleaning steps.
- **`temperature_cleaning_effect.png`**: The original temperature data was provided as a string (e.g., "25/35"). This plot shows the distribution of the extracted minimum temperature before and after being cleaned and converted to a numeric type, demonstrating the successful parsing of the data.

### `03_feature_engineering`
Here, we visualize the relationship of newly created features with the target variable.
- **`land_x_density_visualization.png`**: This scatter plot shows an engineered interaction feature (Land Area * Population Density) against farmer income. This helps validate whether the new feature has a meaningful and predictive relationship with the target.

### `04_outlier_handling`
This plot demonstrates the effect of techniques used to mitigate the effect of extreme values.
- **`outlier_capping_effect.png`**: This uses a boxplot to show the distribution of the 'Total_Land_For_Agriculture' feature before and after capping its extreme values at the 1st and 99th percentiles. This prevents outliers from having an outsized influence on the model.

### `05_target_transformation`
This visualizes the transformation applied to the target variable to help the model perform better.
- **`log_transform_effect.png`**: This compares the histogram of the target variable before and after applying a log transformation. It clearly shows the transformation making the distribution more "normal," which helps linear-based models and improves the performance of many other models.

### `06_final_results`
These plots are related to the final trained model.
- **`actual_vs_predicted.png`**: This is a "goodness-of-fit" plot. It compares the model's predictions against the actual values *on the training set*. The closer the points lie to the diagonal red line, the better the model has learned the patterns in the training data.
- **`feature_importance.png`**: This bar chart displays the top 20 most influential features in the model. It helps explain what factors are most important for predicting a farmer's income and provides insight into the model's decision-making process. 