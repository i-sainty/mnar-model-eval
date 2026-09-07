import joblib
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load the Data
print("Loading processed data...")
X_train_processed = joblib.load('../data/processed/X_train_processed.joblib')
X_test_processed = joblib.load('../data/processed/X_test_processed.joblib')
y_train = joblib.load('../data/processed/y_train.joblib')
y_test = joblib.load('../data/processed/y_test.joblib')

# 2. Apply SMOTE to the Training Data ONLY
print("Applying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# 3. Initialize the Model (Standard initialization)
print("Training model on SMOTE data...")
smote_model = LogisticRegression(max_iter=1000, random_state=42)

# 4. Train the Model on the RESAMPLED Data
smote_model.fit(X_train_resampled, y_train_resampled)

# 5. Make Predictions on the ORIGINAL Unseen Test Data
y_pred_smote = smote_model.predict(X_test_processed)

# 6. Output Results
print("\n--- Improved Model Performance (SMOTE) ---")
print(classification_report(y_test, y_pred_smote))