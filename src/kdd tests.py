import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# ==========================================
# 1. LOAD & SELECT FEATURES
# ==========================================
print("Loading KDD98 raw text data...")
df = pd.read_csv('cup98LRN.txt', sep=',', low_memory=False)

# Target & Feature separation
target_col = 'TARGET_B'
num_features = ['INCOME', 'WEALTH1', 'IC1', 'AVGGIFT', 'LASTGIFT', 'MAXRAMNT', 'NGIFTALL', 'AGE']
cat_features = ['GENDER']

# Clean up types and keep only the selected subset
df = df[[target_col] + num_features + cat_features].copy()
for col in num_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop initial NaNs from this baseline subset so we have a clean ground truth
df = df.dropna().reset_index(drop=True)

X = df[num_features + cat_features]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================================
# 2. PURE MNAR INJECTION FUNCTION
# ==========================================
def inject_pure_mnar(df_input, target_columns, missing_rate=0.30):
    """
    Masks the highest values within target_columns to simulate
    high-net-worth donor non-response (Pure MNAR).
    """
    df_degraded = df_input.copy()
    for col in target_columns:
        cutoff = df_degraded[col].quantile(1 - missing_rate)
        mask = df_degraded[col] >= cutoff
        df_degraded.loc[mask, col] = np.nan
    return df_degraded

# ==========================================
# 3. CSV LOGGING FUNCTION
# ==========================================
LOG_PATH = '../reports/kdd_experiment_log.csv'

def log_result(model_name, degradation, masked_cols, rate, imputation, f1_donor, recall_donor, acc):
    row = pd.DataFrame([{
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Dataset': 'KDD_Cup_1998',
        'Model': model_name,
        'Degradation_Type': degradation,
        'Features_Masked': masked_cols,
        'Missing_Rate': rate,
        'Imputation_Method': imputation,
        'F1_Score_Donor': round(f1_donor, 4),
        'Recall_Donor': round(recall_donor, 4),
        'Overall_Accuracy': round(acc, 4)
    }])
    if not os.path.exists(LOG_PATH):
        row.to_csv(LOG_PATH, index=False)
    else:
        row.to_csv(LOG_PATH, mode='a', header=False, index=False)

# ==========================================
# 4. PREPROCESSORS FOR MODELS
# ==========================================
# Scaled + Imputed pipeline for Linear/Bagged models (LogReg & RF)
mice_preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('mice', IterativeImputer(max_iter=5, random_state=42)),
        ('scaler', StandardScaler())
    ]), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# Passthrough preprocessor for Tree Native NA Handlers (XGBoost & LightGBM)
native_preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])


# ==========================================
# 5. EXPERIMENTAL BENCHMARKING LOOP
# ==========================================
missing_rates = [0.0, 0.10, 0.30, 0.60, 0.80, 0.99]
mnar_targets = ['INCOME', 'LASTGIFT']

# Pre-transform test sets
#X_test_mice = mice_preprocessor.fit(X_train).transform(X_test)
#X_test_native = native_preprocessor.fit(X_train).transform(X_test)

# Calculate scale_pos_weight for imbalance in XGBoost/LightGBM
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
pos_weight = neg_count / pos_count



print("\n🚀 Starting KDD98 MNAR Benchmark...")

from sklearn.metrics import precision_recall_curve
import numpy as np

# ==========================================
# 5. PRE-LOOP OPTIMIZATION (FIND THRESHOLD)
# ==========================================
print("\n⚙️ Calculating Optimal Decision Threshold on Baseline Data...")

# Temporarily transform baseline data to find the perfect mathematical threshold
X_tr_base = mice_preprocessor.fit_transform(X_train)
X_te_base = mice_preprocessor.transform(X_test)

# Train a baseline Random Forest to extract probabilities
rf_base = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_base.fit(X_tr_base, y_train)
rf_probs = rf_base.predict_proba(X_te_base)[:, 1]

# Calculate Precision and Recall across all possible thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, rf_probs)
f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"✅ Optimal Mathematical Threshold Found: {optimal_threshold:.4f}")

# ==========================================
# 6. EXPERIMENTAL BENCHMARKING LOOP
# ==========================================
missing_rates = [0.0, 0.10, 0.30, 0.60, 0.80, 0.99]
mnar_targets = ['INCOME', 'LASTGIFT']

# Calculate scale_pos_weight for imbalance in XGBoost/LightGBM
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
pos_weight = neg_count / pos_count

print("\n🚀 Starting KDD98 MNAR Benchmark...")

for rate in missing_rates:
    print(f"\n--- Running Degradation Rate: {int(rate * 100)}% ---")
    
    # Apply synthetic MNAR
    X_train_deg = inject_pure_mnar(X_train, mnar_targets, missing_rate=rate) if rate > 0 else X_train.copy()
    
    # Transform BOTH train and test inside the loop to prevent shape crashing
    X_tr_mice = mice_preprocessor.fit_transform(X_train_deg)
    X_te_mice = mice_preprocessor.transform(X_test)
    
    X_tr_nat = native_preprocessor.fit_transform(X_train_deg)
    X_te_nat = native_preprocessor.transform(X_test)
    
    # 1. Logistic Regression + MICE
    log_reg = LogisticRegression(class_weight='balanced', max_iter=1000)
    log_reg.fit(X_tr_mice, y_train)
    probs_lr = log_reg.predict_proba(X_te_mice)[:, 1]
    preds_lr = (probs_lr >= optimal_threshold).astype(int)
    log_result('LogReg (Balanced)', 'Pure MNAR', '+'.join(mnar_targets), rate, 'MICE',
               f1_score(y_test, preds_lr, pos_label=1), recall_score(y_test, preds_lr, pos_label=1), accuracy_score(y_test, preds_lr))

    # 2. Random Forest + MICE
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_tr_mice, y_train)
    probs_rf = rf.predict_proba(X_te_mice)[:, 1]
    preds_rf = (probs_rf >= optimal_threshold).astype(int)
    log_result('Random Forest (Balanced)', 'Pure MNAR', '+'.join(mnar_targets), rate, 'MICE',
               f1_score(y_test, preds_rf, pos_label=1), recall_score(y_test, preds_rf, pos_label=1), accuracy_score(y_test, preds_rf))

    # 3. XGBoost
    xgb_clf = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
    xgb_clf.fit(X_tr_nat, y_train)
    probs_xgb = xgb_clf.predict_proba(X_te_nat)[:, 1]
    preds_xgb = (probs_xgb >= optimal_threshold).astype(int)
    log_result('XGBoost', 'Pure MNAR', '+'.join(mnar_targets), rate, 'Native',
               f1_score(y_test, preds_xgb, pos_label=1), recall_score(y_test, preds_xgb, pos_label=1), accuracy_score(y_test, preds_xgb))

    # 4. LightGBM
    lgb_clf = lgb.LGBMClassifier(scale_pos_weight=pos_weight, random_state=42, verbose=-1)
    lgb_clf.fit(X_tr_nat, y_train)
    probs_lgb = lgb_clf.predict_proba(X_te_nat)[:, 1]
    preds_lgb = (probs_lgb >= optimal_threshold).astype(int)
    log_result('LightGBM', 'Pure MNAR', '+'.join(mnar_targets), rate, 'Native',
               f1_score(y_test, preds_lgb, pos_label=1), recall_score(y_test, preds_lgb, pos_label=1), accuracy_score(y_test, preds_lgb))

print(f"\n✅ All runs complete. Results appended to {LOG_PATH}")