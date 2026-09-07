# Evaluating Machine Learning Resilience to MNAR Data in Non-Profit CRM Systems

## Abstract & Background
Non-profit organizations increasingly rely on predictive Customer Relationship Management (CRM) models to identify high-net-worth donors[cite: 9]. However, these systems are highly vulnerable to "Missing Not At Random" (MNAR) data degradation, a statistical phenomenon occurring when wealthy individuals intentionally conceal financial metrics[cite: 9]. This research evaluates the predictive resilience of modern native-handling machine learning architectures (XGBoost and LightGBM) against traditional imputation pipelines (Logistic Regression and Random Forest paired with MICE) under severe MNAR conditions[cite: 9].

## Repository Structure
* **`data/raw/`**: Contains `sample_data.csv`. Full datasets must be downloaded locally and placed here.
* **`src/`**: Standalone Python scripts for data preprocessing, MNAR degradation, and model evaluation.

## Data Access & Setup
Due to file size constraints, the not all full datasets are not hosted in this repository. 
1. Download the **KDD Cup 1998** from https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html and **UCI Adult Census Income** datasets from data/raw[cite: 4, 7]. 
2. Install the required Python dependencies: `pip install pandas numpy scikit-learn xgboost lightgbm`[cite: 4].

## Usage
Run the Python pipelines in the `src/` directory to synthetically inject MNAR degradation into the baseline tabular datasets and evaluate the F1-Score decay of the four distinct models[cite: 7].
