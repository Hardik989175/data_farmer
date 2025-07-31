# Import required libraries
import pandas as pd
import numpy as np
import optuna
import gc

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error

# Load the preprocessed training data
print("Loading preprocessed training data...")
train_df = pd.read_csv('final_pipeline/preprocessed_data/train_with_external_data.csv')

# Define the target column
TARGET = 'Target_Variable/Total Income'

# Drop non-numeric and ID columns from training data
X = train_df.drop(columns=[TARGET, 'FarmerID']).select_dtypes(include=np.number)
y = train_df[TARGET]

# Define the evaluation metric (Root Mean Squared Error)
# This function computes RMSE, which is suitable for regression problems
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Create a scorer object that tells sklearn to use RMSE during evaluation
log_rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Define Optuna's objective function for tuning
def objective(trial):
    """
    This function tells Optuna how to explore the hyperparameter space.
    It defines the range of values for each model parameter and evaluates
    performance using K-Fold cross-validation.
    """

    # Suggest hyperparameters for CatBoost
    cat_params = {
        'iterations': trial.suggest_int('cat_iterations', 800, 2000),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('cat_depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1, 10),
        'verbose': 0,  # suppress CatBoost's output
        'random_seed': 42
    }

    # Suggest hyperparameters for XGBoost
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 800, 2000),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('xgb_max_depth', 6, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('xgb_gamma', 0, 5),
        'random_state': 42
    }

    # Suggest hyperparameter for Ridge regression (meta-learner)
    
    ridge_alpha = trial.suggest_float('ridge_alpha', 0.5, 5.0)

    # --- Define the stacked ensemble model ---
    # Base learners: CatBoost and XGBoost
    # Final estimator: Ridge regression
    estimators = [
        ('cat', CatBoostRegressor(**cat_params)),
        ('xgb', XGBRegressor(**xgb_params))
    ]

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=ridge_alpha),
        cv='passthrough'  # Allows final estimator to use base learners' outputs
    )

    # Use 5-Fold Cross-Validation to evaluate the model
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X, y):
        # Split data into training and validation sets
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Train the stacked model on the training fold
        stacking_regressor.fit(X_train, y_train)

        # Predict on the validation fold
        preds = stacking_regressor.predict(X_val)

        # Calculate RMSE and store it
        score = rmse(y_val, preds)
        scores.append(score)

        # Free memory to avoid memory leaks during optimization
        del X_train, X_val, y_train, y_val, preds
        gc.collect()

    # Return the average RMSE across all folds
    return np.mean(scores)

# Start the Optuna optimization process
if __name__ == "__main__":
    print("Starting hyperparameter tuning with Optuna")

    # Create a new Optuna study
    # Goal: Minimize the average RMSE from cross-validation
    study = optuna.create_study(direction='minimize')

    # Run the optimization with a fixed number of trials
    study.optimize(objective, n_trials=50)  # You can increase n_trials for better tuning

    # Display results
    print("\nTuning Complete")
    print("Best trial found:")
    trial = study.best_trial
    print(f"  Value (Average RMSE): {trial.value:.4f}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Optional: Save the study object for future reference
    # import joblib
    # joblib.dump(study, 'stacking_model_tuning.pkl')

    print("\nTo use these hyperparameters, update them in the main training script.")
