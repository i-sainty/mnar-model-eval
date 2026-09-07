import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

# ---------------------------------------------------------
# 1. Custom Functions (Degradation & Logging)
# ---------------------------------------------------------
def inject_pure_mnar_multi(X, target_columns=['capital-gain', 'education-num'], missing_rate=0.30):
    X_corrupted = X.copy()
    for col in target_columns:
        sorted_col = X_corrupted[col].sort_values(ascending=False)
        num_to_drop = int(len(sorted_col) * missing_rate)
        indices_to_mask = sorted_col.head(num_to_drop).index
        X_corrupted.loc[indices_to_mask, col] = np.nan
    return X_corrupted

def log_experiment_results(y_true, y_pred, model_name, deg_type, features, missing_rate, imputer):
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
df = pd.read_csv('census.csv') 

df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate class weight for XGBoost
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ---------------------------------------------------------
# 3. Build the XGBoost Preprocessor (NO NUMERICAL IMPUTATION)
# ---------------------------------------------------------
numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'education_level', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'native-country']

xgb_cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

xgb_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols), # Pass raw NaNs straight to XGBoost!
        ('cat', xgb_cat_pipeline, categorical_cols)
    ])

# ---------------------------------------------------------
# 4. Automated Experiment Loop
# ---------------------------------------------------------
rates_to_test = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99] 

print("\n" + "="*50)
print("🚀 STARTING XGBOOST CASCADING FAILURE EXPERIMENT")
print("="*50)

for rate in rates_to_test:
    if rate == 0.0:
        X_train_mnar = X_train.copy()
    else:
        X_train_mnar = inject_pure_mnar_multi(
            X_train, 
            target_columns=['capital-gain', 'education-num'], 
            missing_rate=rate
        )
    
    # Process data (NaNs remain in the numerical columns)
    X_train_processed = xgb_preprocessor.fit_transform(X_train_mnar)
    X_test_processed = xgb_preprocessor.transform(X_test) 
    
    # Train XGBoost
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_weight, 
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train_processed, y_train)
    
    # Predict and Evaluate
    y_pred_xgb = xgb_model.predict(X_test_processed)
    
    print(f"\n--- Quick Summary ({rate*100}% MNAR) ---")
    print(classification_report(y_test, y_pred_xgb, target_names=['<=50K', '>50K']))
    
    # Log Results
    log_experiment_results(
        y_true=y_test, 
        y_pred=y_pred_xgb, 
        model_name="XGBoost", 
        deg_type="MNAR (Pure)", 
        features="capital-gain + education-num", 
        missing_rate=rate, 
        imputer="Native Handling"
    )

print("\n🎉 XGBOOST EXPERIMENTS COMPLETE! Check reports/experiment_log.csv")