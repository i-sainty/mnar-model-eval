import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# 1. Load the text file (adjust the file path as needed)
print("Loading KDD98 raw data...")
df = pd.read_csv('cup98LRN.txt', sep=',', low_memory=False)

# 2. Define Target and Predictors
X = df.drop(columns=['TARGET_B', 'TARGET_D'])
y = df['TARGET_B']

# 3. Identify categorical and numerical columns (Update these based on your chosen features)
categorical_features = ['GENDER', 'HOMEOWNR', 'DOMAIN'] 
numerical_features = ['AGE', 'INCOME', 'WEALTH1', 'HIT']

X = X[categorical_features + numerical_features]

# 4. Create the Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Split and Transform
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 6. Save the processed data for your LogReg, RF, and XGBoost scripts
joblib.dump(X_train_processed, '../data/processed/kdd_X_train_clean.joblib')
joblib.dump(X_test_processed, '../data/processed/kdd_X_test_clean.joblib')
joblib.dump(y_train, '../data/processed/kdd_y_train.joblib')
joblib.dump(y_test, '../data/processed/kdd_y_test.joblib')

print("✅ Pre-processing complete. Data saved to data/processed/")
