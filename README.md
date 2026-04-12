# 📊 Rossmann Store Sales: 6-Week Forecasting & Driver Analysis (22.1)
![Static Badge](https://img.shields.io/badge/ROSSMANN%20Sales-%20DATA SCIENCE-red)

## 🎯 Project Overview
This project applies the end-to-end CRISP-DM lifecycle to build an automated machine learning pipeline. The primary objective is to forecast daily revenue for 1,115 Rossmann stores up to 6 weeks in advance, enabling highly accurate inventory and staffing optimization.

**Key Achievements:**
* **Forecasting Accuracy:** Developed a tuned XGBoost Regressor that achieved a Root Mean Square Percentage Error (RMSPE) of **13.3%**, successfully beating the <15% business safety threshold.
* **Business Insights:** Quantified the exact "marginal lift" of store promotions to validate marketing budgets and mapped out the revenue impact of competitor store proximity.
* **Deployment:** Transitioned experimental Jupyter notebooks into a modular, production-ready Python batch-processing script.

## 📂 Project Structure
* **`notebooks/`**: Contains the complete 6-phase CRISP-DM research environment (Data Prep, EDA, Modeling, and Evaluation).
* **`src/`**: Contains the production software factory.
  * `processing.py`: Modular data cleaning and feature engineering functions.
  * `predict.py`: The executable batch script for generating new forecasts.
* **`models/`**: Stores the serialized, pre-trained model (`rossmann_xgboost_final.pkl`).
* **`data/`**: (Local Only) Contains `raw/` input CSVs and the `processed/` final outputs.

## ⚙️ Environment Setup
1. Create and activate the environment:
`conda create -n rossmann_ds python=3.10`
`conda activate rossmann_ds`

2. Install dependencies:
`pip install pandas numpy scikit-learn xgboost joblib ipykernel`

## ⚙️ Pipeline Usage
1. Ensure your raw input data (e.g., test.csv) is located inside the data/raw/ directory.

2. Open your terminal at the root of the project directory and ensure that it is on `cmd` run the batch prediction script:
`python src/predict.py`
If you are facing issues use this following line:
`%USERPROFILE%\anaconda3\envs\rossmann_ds\python.exe src/predict.py`

3. The script will automatically clean the data, apply the model, and save the output to `data/processed/final_6_week_forecast.csv`