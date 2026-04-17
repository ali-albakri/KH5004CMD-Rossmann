import pandas as pd
import numpy as np
import re

def preprocess_store_data(df, training_columns=None):
    df_clean = df.copy()
    
    # handle missing values
    df_clean = df_clean.ffill()
    
    # handle dates because XGBoost can't read raw date formats
    if 'date' in df_clean.columns:
        df_clean = df_clean.drop(columns=['date'])
        
    # drop target and customer columns if they accidentally sneaked in
    cols_to_drop = ['sales', 'customers', 'salespercustomer']
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], errors='ignore')

    # categorical encoding
    print("Encoding categorical variables...")
    df_encoded = pd.get_dummies(df_clean)
    
    # cleaning the column names so they exactly match the Phase 4 training data
    df_encoded.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in df_encoded.columns]
    
    # column alignment in the exact order
    if training_columns is not None:
        df_encoded = df_encoded.reindex(columns=training_columns, fill_value=0)
    
    print("Preprocessing complete.")
    print(f"Final shape: {df_encoded.shape}")
    return df_encoded