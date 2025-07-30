import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
import gc

# --- Load Data ---
print("Loading preprocessed training data...")
train_df = pd.read_csv('final_pipeline/preprocessed_data/train_with_external_data.csv')

TARGET = 'Target_Variable/Total Income'
X = train_df.drop(columns=[TARGET, 'FarmerID']).select_dtypes(include=np.number)
y = train_df[TARGET]

# --- Define Scorer ---
# We use Root Mean Squared Error on the log-transformed values, which is common for this type of problem.
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

log_rmse_scorer = make_scorer(rmse, greater_is_better=False)

# --- Objective Function for Optuna ---
def objective(trial):
    """
    Defines the search space for hyperparameters and evaluates the model.
    """
    
    # -- CatBoost Hyperparameters --
    cat_params = {
        'iterations': trial.suggest_int('cat_iterations', 800, 2000),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('cat_depth', 6, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1, 10),
        'verbose': 0,
        'random_seed': 42
    }
    
    # -- XGBoost Hyperparameters --
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 800, 2000),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('xgb_max_depth', 6, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('xgb_gamma', 0, 5),
        'random_state': 42
    }

    # -- Final Estimator (Meta-Learner) Hyperparameters --
    ridge_alpha = trial.suggest_float('ridge_alpha', 0.5, 5.0)

    # -- Define the Stacking Model --
    estimators = [
        ('cat', CatBoostRegressor(**cat_params)),
        ('xgb', XGBRegressor(**xgb_params))
    ]
    
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=ridge_alpha),
        cv='passthrough' # Use the base models' internal CV if they have it, or just fit them.
    )

    # -- Cross-validation --
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        stacking_regressor.fit(X_train, y_train)
        preds = stacking_regressor.predict(X_val)
        
        # We are evaluating on the log-transformed values as per our target.
        score = rmse(y_val, preds)
        scores.append(score)
        
        # Clean up memory
        del X_train, X_val, y_train, y_val, preds
        gc.collect()

    return np.mean(scores)

# --- Run the Optimization ---
if __name__ == "__main__":
    print("Starting hyperparameter tuning with Optuna...")
    
    # Create a study object and specify the direction is to minimize the score (RMSE)
    study = optuna.create_study(direction='minimize')
    
    # Start the optimization
    study.optimize(objective, n_trials=50) # You can change n_trials to run more iterations
    
    print("\n--- Tuning Complete ---")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (RMSE): {trial.value}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # You can save the study results if you want
    # import joblib
    # joblib.dump(study, 'stacking_model_tuning.pkl')
    print("\nTo use these hyperparameters, update them in the main prediction script.") 