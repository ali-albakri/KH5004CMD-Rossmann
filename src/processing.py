import pandas as pd
import numpy as np
import re

def get_holiday_countdown(current_date):
    future_holidays = pd.to_datetime([     # Major German holidays for the 2015-2016 period
        '2015-10-03', '2015-10-31', '2015-11-01', '2015-12-25', 
        '2015-12-26', '2016-01-01', '2016-03-25', '2016-03-28'
    ])
    
    diffs = [(target - current_date).days for target in future_holidays if target >= current_date] # Only consider future holidays
    
    # Return the smallest difference, or 21 (our cap) if none found
    return min(diffs) if diffs else 21 # Cap at 21 days to prevent extreme values

def preprocess_store_data(df, training_columns=None):
    df_clean = df.copy()
    df_clean = df_clean.ffill() # Forward fill to handle missing values
    
    if 'Date' in df_clean.columns: # Convert to datetime if 'Date' column exists
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        
        df_clean['days_until_next_holiday'] = df_clean['Date'].apply(get_holiday_countdown) # Apply the function to create the new feature
        df_clean['days_until_next_holiday'] = df_clean['days_until_next_holiday'].clip(upper=21)
        
        df_clean['year'] = df_clean['Date'].dt.year
        df_clean['month'] = df_clean['Date'].dt.month
        df_clean['day'] = df_clean['Date'].dt.day
        df_clean['weekofyear'] = df_clean['Date'].dt.isocalendar().week.astype(int)
        
        df_clean = df_clean.drop(columns=['Date']) # Remove raw date as XGBoost can't read it
        
    # Drop columns that don't exist in the test set or are target variables
    cols_to_drop = ['sales', 'customers', 'salespercustomer']
    df_clean = df_clean.drop(columns=[c for c in cols_to_drop if c in df_clean.columns], errors='ignore')

    df_encoded = pd.get_dummies(df_clean) # One-hot encode categorical variables
    
    # Clean the column names (which fixes the LightGBM/XGBoost character error)
    df_encoded.columns = [re.sub(r'[^\w\s]', '', col).replace(' ', '_') for col in df_encoded.columns]
    
    if training_columns is not None: # Ensure the test set has the same columns as the training set, filling missing ones with 0
        df_encoded = df_encoded.reindex(columns=training_columns, fill_value=0)
    
    return df_encoded