import pandas as pd
import numpy as np
import joblib
import os
from processing import preprocess_store_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'rossmann_xgboost_final.pkl')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'test.csv')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_6_week_forecast.csv')

def main():
    print("=== Rossmann Batch Prediction Pipeline ===")
    
    # load the trained XGBoost model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Did you run Phase 4?")
    
    print("Loading XGBoost model...")
    model = joblib.load(MODEL_PATH)
    
    # extract the exact column names the model expects
    expected_columns = model.feature_names_in_
    
    # load the raw test data
    print(f"Loading new data from {RAW_DATA_PATH}...")
    raw_df = pd.read_csv(RAW_DATA_PATH)
    
    # save the IDs if we need them for the final submission format
    if 'Id' in raw_df.columns:
        submission_ids = raw_df['Id']
        raw_df = raw_df.drop(columns=['Id'])
    else:
        submission_ids = raw_df.index
    
    # process the data using our custom module
    X_processed = preprocess_store_data(raw_df, training_columns=expected_columns)
    
    # generate predictions
    print("Generating forecasts...")
    log_predictions = model.predict(X_processed)
    
    # unsquash the log transform back to real currency
    real_predictions = np.expm1(log_predictions)
    
    print("Formatting output...")
    output_df = pd.DataFrame({
        'Id': submission_ids,
        'Forecasted_Sales': real_predictions
    })
    
    output_df.to_csv(OUTPUT_PATH, index=False)
    print("Success!")
    print(f"Forecasts saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()