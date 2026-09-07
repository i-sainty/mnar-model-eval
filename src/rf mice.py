import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime

# Specific imports for Random Forest and MICE
from sklearn.experimental import enable_iterative_imputer # Must be imported before IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

# ---------------------------------------------------------
# 1. Custom Functions (Degradation & Logging)
# ---------------------------------------------------------
def inject_pure_mnar_multi(X, target_columns=['capital-gain', 'education-num'], missing_rate=0.30):
    """Injects pure MNAR by masking the highest values in target columns."""
    X_corrupted = X.copy()
    for col in target_columns:
        sorted_col = X_corrupted[col].sort_values(ascending=False)
        num_to_drop = int(len(sorted_col) * missing_rate)
        indices_to_mask = sorted_col.head(num_to_drop).index
        X_corrupted.loc[indices_to_mask, col] = np.nan
    return X_corrupted

def log_experiment_results(y_true, y_pred, model_name, deg_type, features, missing_rate, imputer):
    """Logs the results to the master CSV tracker."""
    os.makedirs('../reports', exist_ok=True)
    log_file = '../reports/experiment_log.csv'
    file_exists = os.path.isfile(log_file)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, y_pred)
    
    row = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model': model_name,
        'Degradation_Type': deg_type,
        'Features_Masked': features,
        'Missing_Rate': missing_rate,
        'Imputation_Method': imputer,
        'F1_Score_Wealthy': round(f1[1], 4), 
        'Recall_Wealthy': round(recall[1], 4),
        'Overall_Accuracy': round(accuracy, 4)
    }
    
    with open(log_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader() 
        writer.writerow(row)

# ---------------------------------------------------------
# 2. Load and Prepare the Data
# ---------------------------------------------------------
print("Loading raw data...")
# Make sure 'census.csv' is in the same directory, or update this path
df = pd.read_csv('census.csv') 

# Binarize target variable
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
X = df.drop('income', axis=1)
y = df['income']

# Train/Test Split (Stratified to maintain class imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# 3. Build the MICE + Random Forest Preprocessor
# ---------------------------------------------------------
numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'education_level', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'native-country']

# Categorical Pipeline: Fill missing categories with the most frequent, then One-Hot Encode
rf_cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Full Preprocessor: Use MICE (IterativeImputer) for numericals instead of passing NaNs through
rf_preprocessor = ColumnTransformer(
    transformers=[
        ('num', IterativeImputer(max_iter=10, random_state=42), numerical_cols), 
        ('cat', rf_cat_pipeline, categorical_cols)
    ])

# ---------------------------------------------------------
# 4. Automated Experiment Loop
# ---------------------------------------------------------
rates_to_test = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99] 

print("\n" + "="*50)
print("🚀 STARTING RANDOM FOREST + MICE EXPERIMENT")
print("="*50)

for rate in rates_to_test:
    # 1. Inject MNAR degradation
    if rate == 0.0:
        X_train_mnar = X_train.copy()
    else:
        X_train_mnar = inject_pure_mnar_multi(
            X_train, 
            target_columns=['capital-gain', 'education-num'], 
            missing_rate=rate
        )
    
    # 2. Process data (MICE kicks in here to predict the missing numericals)
    print(f"Running MICE Imputation for {rate*100}% MNAR...")
    X_train_processed = rf_preprocessor.fit_transform(X_train_mnar)
    X_test_processed = rf_preprocessor.transform(X_test) 
    
    # 3. Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced', # Crucial for handling the minority class
        random_state=42,
        n_jobs=-1 # Uses all CPU cores for speed
    )
    rf_model.fit(X_train_processed, y_train)
    
    # 4. Predict and Evaluate
    y_pred_rf = rf_model.predict(X_test_processed)
    
    print(f"\n--- Quick Summary ({rate*100}% MNAR) ---")
    print(classification_report(y_test, y_pred_rf, target_names=['<=50K', '>50K']))
    
    # 5. Log Results
    log_experiment_results(
        y_true=y_test, 
        y_pred=y_pred_rf, 
        model_name="Random Forest (Balanced)", 
        deg_type="MNAR (Pure)", 
        features="capital-gain + education-num", 
        missing_rate=rate, 
        imputer="MICE (IterativeImputer)"
    )

print("\n🎉 RANDOM FOREST EXPERIMENTS COMPLETE! Check ../reports/experiment_log.csv")