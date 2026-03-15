# Rossmann Store Sales - Retail Profitability & Sales Forecast (22.1)

## Project Overview
This project applies the CRISP-DM lifecycle to forecast daily sales for 1,115 Rossmann stores 6 weeks in advance. It also analyzes the "marginal lift" of promotions to determine net profitability.

## Project Structure
- `notebooks/`: Contains the 6-phase CRISP-DM execution notebooks.
- `data/`: (Local Only) Stores raw Rossmann CSVs and processed datasets.
- `src/`: Reusable Python scripts for modularity.

## Environment Setup
1. Create the environment: `conda create -n rossmann_ds python=3.10`
2. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn ipykernel`