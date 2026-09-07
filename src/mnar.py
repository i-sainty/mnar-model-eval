import pandas as pd
import numpy as np

def inject_mnar(X, y, target_column='capital-gain', missing_rate=0.30):
    """
    Injects Missing Not At Random (MNAR) degradation.
    Specifically targets individuals in the >50K bracket (y==1) 
    and masks their data in the target_column.
    """
    # Create a copy so we do not overwrite the original clean data
    X_corrupted = X.copy()
    
    # 1. Identify the index of the wealthy individuals (y == 1)
    wealthy_indices = y[y == 1].index
    
    # 2. Calculate exactly how many rows to drop based on the threshold
    num_to_drop = int(len(wealthy_indices) * missing_rate)
    
    # 3. Randomly select specific wealthy individuals to mask
    np.random.seed(42) # Set seed for perfect reproducibility in your dissertation
    indices_to_mask = np.random.choice(wealthy_indices, size=num_to_drop, replace=False)
    
    # 4. Inject the missingness (NaN) 
    X_corrupted.loc[indices_to_mask, target_column] = np.nan
    
    print(f"MNAR Degradation (Rate: {missing_rate*100}%): {num_to_drop} values in '{target_column}' masked.")
    
    return X_corrupted

# --- How to Test the Function ---
# Assuming you have your pre-processed X_train (DataFrame) and y_train from Step 1:
# X_train_mnar_30 = inject_mnar(X_train, y_train, target_column='capital-gain', missing_rate=0.30)