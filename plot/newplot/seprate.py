import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 18  # 设置全局字体大小为14
from collections import defaultdict

watermark_types = ["john23","xuandong23b","aiwei23","rohith23","xiaoniu23","aiwei23b","scott22"]
replace_dict = {
    "john23": "TGRL",
    "xuandong23b": "UG",
    "aiwei23": "UPV",
    "rohith23": "RDF",
    "xiaoniu23": "UB",
    "lean23": "CTWL",
    "scott22": "GO",
    "aiwei23b": "SIR",
}


colors = [
    'darkorange',
    'deepskyblue',
    'limegreen',
    'violet',
    'goldenrod',
    'lightpink',

    'teal'
]

color_dict = dict(zip(watermark_types, colors))

attacks = ["swap","translation","synonym-0.4", "copypaste-1-10","copypaste-3-10","copypaste-1-25","copypaste-3-25", "ContractionAttack", "ExpansionAttack",  "MisspellingAttack",   "dipper_l20_o0", "dipper_l40_o0", "LowercaseAttack", "TypoAttack"]
dataset=["hc3","c4"]
import csv
import matplotlib.pyplot as plt

# Read the data
data = {}
with open('/home/ljc/sok-llm-watermark/plot/newplot/tpr.data.csv', 'r') as f:
    reader = csv.reader(f)
    # next(reader)  # Skip the header

    for row in reader:
        attack = row[1]
        dataset = row[3]
        method = row[0]
        value = float(row[2])

        if attack not in data:
            data[attack] = {}
        if dataset not in data[attack]:
            data[attack][dataset] = {}
        data[attack][dataset][method] = value

# Define the attacks and datasets
attacks = ['synonym-0.4', 'MisspellingAttack', 'TypoAttack']
datasets = ['c4', 'hc3']
alphas = [1.0,0.5]
methods = list(data[attacks[0]][datasets[0]].keys())  # Get the methods from the data

attack_categories = {
    "Textual Integrity Attacks": ["synonym-0.4", "MisspellingAttack", "TypoAttack"],
    "Content Manipulation": ["copypaste-1-10", "copypaste-3-10","copypaste-1-25","copypaste-3-25"],
    "Linguistic Variation Attacks": ["ContractionAttack", "ExpansionAttack", "LowercaseAttack"],
    "Attack-Resistant Strategies": ["swap"],
    "Paraphrase Attack": ["dipper_l20_o0","dipper_l40_o0"]
}

# Plot the data for each category
# Plot the data for each category
for category, attacks in attack_categories.items():
    # Calculate the number of rows
    rows = (len(attacks) + 1) // 2

    # Create the subplots
    fig, axs = plt.subplots(rows, 2, figsize=(20, 10 * rows))

    # If there is only one row, make axs a 2D array
    if rows == 1:
        axs = np.array([axs])

    for i, attack in enumerate(attacks):
        for j, dataset in enumerate(datasets):
            values = [data[attack][dataset][method] for method in methods]
            colors = [color_dict.get(method, "black") for method in methods]
            axs[i // 2][i % 2].bar([x + j*0.4 for x in range(len(methods))], values, width=0.4, label=dataset, color=colors, alpha=alphas[j])

        axs[i // 2][i % 2].set_title(attack)
        axs[i // 2][i % 2].set_xticks(range(len(methods)))
        axs[i // 2][i % 2].set_xticklabels(methods, rotation=30, ha='right')
        axs[i // 2][i % 2].legend()

    plt.tight_layout()
    plt.savefig(f'./plot/newplot/output/{category.replace(" ", "_")}.pdf')  # Save the figure with the category name

    plt.close(fig)  # Close the figure to free up memory
# \subsubsection{Textual Integrity Attacks: Synonym Substitution, Misspelling, and Typographical Errors}
# \subsubsection{Content Manipulation: Copy-Paste Attacks and Textual Manipulation}
# \subsubsection{Linguistic Variation Attacks: Contraction, Expansion, and Lowercase Attacks}
# \subsubsection{Attack-Resistant Strategies: Swap Attacks}
# \subsubsection{Paraphrase Attack: Deep Paraphrasing (Dipper)}
