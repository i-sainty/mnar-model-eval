import pandas as pd

print("Loading text file...")
df = pd.read_csv('cup98LRN.txt', sep=',', low_memory=False)

print("Exporting to Excel...")
df.to_excel('cup98LRN.xlsx', index=False)

print("Conversion complete")
