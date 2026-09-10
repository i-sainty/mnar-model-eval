import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

# Define the Degradation Function
def inject_mnar(X, y, target_column='capital-gain', missing_rate=0.30):
    X_corrupted = X.copy()
    wealthy_indices = y[y == 1].index
    num_to_drop = int(len(wealthy_indices) * missing_rate)
    np.random.seed(42)
    indices_to_mask = np.random.choice(wealthy_indices, size=num_to_drop, replace=False)
    X_corrupted.loc[indices_to_mask, target_column] = np.nan
    print(f"MNAR Degradation (Rate: {missing_rate*100}%): {num_to_drop} values in '{target_column}' masked.")
    return X_corrupted

def inject_pure_mnar_multi(X, target_columns=['capital-gain', 'education-num'], missing_rate=0.90):
    """
    Injects Pure MNAR degradation across multiple columns independently.
    For each column, the probability of missingness depends exclusively 
    on the magnitude of the values within that specific column.
    """
    X_corrupted = X.copy()
    
    for col in target_columns:
        # 1. Sort the specific column from highest to lowest
        sorted_col = X_corrupted[col].sort_values(ascending=False)
        
        # 2. Calculate exactly how many rows to drop based on the threshold
        num_to_drop = int(len(sorted_col) * missing_rate)
        
        # 3. Get the index of the absolute highest values for THIS column
        indices_to_mask = sorted_col.head(num_to_drop).index
        
        # 4. Inject the missingness (NaN)
        X_corrupted.loc[indices_to_mask, col] = np.nan
        
        print(f"Pure MNAR (Rate: {missing_rate*100}%): Top {num_to_drop} highest values in '{col}' masked.")
        
    return X_corrupted

def log_experiment_results(y_true, y_pred, model_name, deg_type, features, missing_rate, imputer):
    """
    Automatically logs experiment results to a CSV file.
    """
    # Ensure the reports directory exists
    os.makedirs('../reports', exist_ok=True)
    log_file = '../reports/experiment_log.csv'
    
    file_exists = os.path.isfile(log_file)
    
    # Extract specific metrics for the '>50K' class (which is class 1)
    # precision_recall_fscore_support returns arrays for [class_0, class_1]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, y_pred)
    
    # Prepare the row of data
    row = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Model': model_name,
        'Degradation_Type': deg_type,
        'Features_Masked': features,
        'Missing_Rate': missing_rate,
        'Imputation_Method': imputer,
        'F1_Score_Wealthy': round(f1[1], 4), # Index 1 gets the >50K class
        'Recall_Wealthy': round(recall[1], 4),
        'Overall_Accuracy': round(accuracy, 4)
    }
    
    # Append the data to the CSV
    with open(log_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
    print(f"✅ Results successfully logged to {log_file}")

# Load the Raw Data
print("Loading raw data...")
df = pd.read_csv('census.csv')

# Prep target variable and split
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
X = df.drop('income', axis=1)
y = df['income']

# Recreate X_train as a DataFrame so our MNAR function can find the 'capital-gain' column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inject the missingness
print("Injecting MNAR missingness...")
#X_train_mnar_30 = inject_mnar(X_train, y_train, target_column='capital-gain', missing_rate=0.10)

# Inject Pure MNAR into both columns
X_train_mnar = inject_pure_mnar_multi(
    X_train, 
    target_columns=['capital-gain', 'education-num'], 
    missing_rate=0.20  # Change this to 0.60, 0.80, etc.
)


# Build the Imputation Pipeline
numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'education_level', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'native-country']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

imputing_preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
'''
# Process, Train, and Evaluate
print("Processing data through the imputation pipeline...")
X_train_imputed = imputing_preprocessor.fit_transform(X_train_mnar)
X_test_imputed = imputing_preprocessor.transform(X_test) 

print("Training model on 30% MNAR degraded data...")
# Using the Balanced Weights baseline you chose earlier!
degraded_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
degraded_model.fit(X_train_imputed, y_train)

y_pred_degraded = degraded_model.predict(X_test_imputed)

print("\n--- Model Performance (40% MNAR on Capital Gain) ---")
print(classification_report(y_test, y_pred_degraded))

#log test
log_experiment_results(
    y_true=y_test, 
    y_pred=y_pred_degraded, 
    model_name="LogReg (Balanced)", 
    deg_type="MNAR", 
    features="capital-gain + education-num", 
    missing_rate=0.20, 
    imputer="Median"
)
'''
# ---------------------------------------------------------
# 5. Automated Experiment Loop
# ---------------------------------------------------------

# Define the exact thresholds
rates_to_test = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]

for rate in rates_to_test:
    print(f"\n{'='*50}")
    print(f"🚀 RUNNING CASCADING FAILURE AT {rate*100}% MISSINGNESS")
    print(f"{'='*50}")
    
    # 1. Inject Pure MNAR into both columns at the current loop rate
    X_train_mnar = inject_pure_mnar_multi(
        X_train, 
        target_columns=['capital-gain', 'education-num'], 
        missing_rate=rate
    )
    
    # 2. Process data through the imputation pipeline
    X_train_imputed = imputing_preprocessor.fit_transform(X_train_mnar)
    X_test_imputed = imputing_preprocessor.transform(X_test) 
    
    # 3. Train the Model
    degraded_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    degraded_model.fit(X_train_imputed, y_train)
    
    # 4. Make Predictions
    y_pred_degraded = degraded_model.predict(X_test_imputed)
    
    # 5. Output quick visual feedback to the console
    print(f"\n--- Quick Summary ({rate*100}%) ---")
    print(classification_report(y_test, y_pred_degraded, target_names=['<=50K', '>50K']))
    
    # 6. Automatically log the current loop's results to your CSV
    log_experiment_results(
        y_true=y_test, 
        y_pred=y_pred_degraded, 
        model_name="LogReg (Balanced)", 
        deg_type="MNAR (Pure)", 
        features="capital-gain + education-num", 
        missing_rate=rate, 
        imputer="Median"
    )

print("\nALL Check reports/experiment_log.csv")
