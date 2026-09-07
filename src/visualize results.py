import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the Experiment Log
log_path = '../reports/kdd_experiment_log.csv'

if not os.path.exists(log_path):
    print(f"Error: Could not find {log_path}. Make sure you've run your experiments.")
    exit()

df = pd.read_csv(log_path)

# 2. Clean and Filter the Data
# Ensure Missing_Rate is numeric
df['Missing_Rate'] = pd.to_numeric(df['Missing_Rate'], errors='coerce')

# Filter for the specific degradation type (Pure MNAR)
df_mnar = df[df['Degradation_Type'].str.contains("MNAR", na=False)]

# NEW FILTER: Only keep the runs where TWO categories were masked
#df_mnar = df_mnar[df_mnar['Features_Masked'] == 'capital-gain + education-num']

#Filter for KDD data
df_mnar = df[df['Features_Masked'] == 'INCOME+LASTGIFT']

if df_mnar.empty:
    print("Warning: No data found after filtering. Check your exact string names.")

# 3. Set up the Visualization Style
sns.set_theme(style="whitegrid")

# Define specific colors for each model to keep them consistent across graphs
model_colors = {
    'LogReg (Balanced)': '#1f77b4',         # Blue
    'Random Forest (Balanced)': '#2ca02c',  # Green 
    'XGBoost': '#ff7f0e',                   # Orange
    'LightGBM': '#d62728'                   # Red
}

# 4. Create the Plotting Function
def plot_metric(data, metric_col, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    
    # Create the line plot
    ax = sns.lineplot(
        data=data, 
        x='Missing_Rate', 
        y=metric_col, 
        hue='Model', 
        style='Model', 
        markers=True, 
        dashes=False,
        palette=model_colors,
        linewidth=2.5,
        markersize=8
    )
    
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Percentage of Data Missing (MNAR)", fontsize=12, labelpad=10)
    plt.ylabel(ylabel, fontsize=12, labelpad=10)
    
    # Format x-axis as percentages
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # Adjust legend
    plt.legend(title='Model Architecture', title_fontsize='11', fontsize='10', loc='lower left')
    
    # Save the figure
    os.makedirs('../reports/figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'../reports/figures/{filename}', dpi=300)
    print(f"✅ Saved figure: {filename}")
    plt.show()

# 5. Generate the Graphs
print("\nGenerating visual reports...\n" + "-"*30)

plot_metric(
    data=df_mnar, 
    metric_col='F1_Score_Donor', 
    title='Model Robustness: F1-Score vs. Missing Data (MNAR)', 
    ylabel='F1-Score (>50K Class)', 
    filename='kdd_mnar_f1_score_comparison.png'
)

plot_metric(
    data=df_mnar, 
    metric_col='Recall_Donor', 
    title='Model Sensitivity: Recall vs. Missing Data (MNAR)', 
    ylabel='Recall (>50K Class)', 
    filename='kdd_mnar_recall_comparison.png'
)

plot_metric(
    data=df_mnar, 
    metric_col='Overall_Accuracy', 
    title='Overall Accuracy vs. Missing Data (MNAR)', 
    ylabel='Accuracy', 
    filename='kdd_mnar_accuracy_comparison.png'
)

print("\n🎉 All visualizations complete! Check the ../reports/figures/ folder.")