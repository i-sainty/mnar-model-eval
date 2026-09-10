import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# 1. Load the Dataset
df = pd.read_csv('census.csv')

# 2. Encode the Target Variable
# Map '<=50K' to 0 and '>50K' to 1
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

# Separate features (X) and target (y)
X = df.drop('income', axis=1)
y = df['income']

# 3. Identify Column Types
numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'education_level', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'native-country']

# 4. Build the Pre-processing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# 5. Split the Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Fit and Transform the Data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

encoded_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(encoded_cat_cols)

print(f"Original Feature Count: {X.shape[1]}")
print(f"Processed Feature Count: {X_train_processed.shape[1]}")
print("Data processing complete. The dataset is now ready for the baseline model.")

import joblib
import os

os.makedirs('../data/processed', exist_ok=True)

joblib.dump(X_train_processed, '../data/processed/X_train_processed.joblib')
joblib.dump(X_test_processed, '../data/processed/X_test_processed.joblib')
joblib.dump(y_train, '../data/processed/y_train.joblib')
joblib.dump(y_test, '../data/processed/y_test.joblib')

print("Processed data successfully saved to 'data/processed/'")
