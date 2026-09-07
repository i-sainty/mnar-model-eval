import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Load the Data
print("Loading processed data...")
X_train_processed = joblib.load('../data/processed/X_train_processed.joblib')
X_test_processed = joblib.load('../data/processed/X_test_processed.joblib')
y_train = joblib.load('../data/processed/y_train.joblib')
y_test = joblib.load('../data/processed/y_test.joblib')

# 2. Initialize the IMPROVED Model (Adding class_weight='balanced')
print("Training improved model (Balanced Weights)...")
weighted_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

# 3. Train on Processed Training Data
weighted_model.fit(X_train_processed, y_train)

# 4. Make Predictions
y_pred_weighted = weighted_model.predict(X_test_processed)

# 5. Output Results
print("\n--- Improved Model Performance (Balanced Weights) ---")
print(classification_report(y_test, y_pred_weighted))