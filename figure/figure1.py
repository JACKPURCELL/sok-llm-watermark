import matplotlib.pyplot as plt
import pandas as pd

# Sample data
data = pd.read_csv('figure/eva-john23-hc3-dipper.csv')

# Creating a DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))
for t in df['type'].unique():
    subset = df[df['type'] == t]
    plt.plot(subset['fpr'], subset['tpr'], label=t)

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC-AUC Curve')
plt.legend()
plt.grid(True)
plt.savefig("figure/1.png")
