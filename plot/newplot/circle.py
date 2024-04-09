from typing_extensions import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
import csv
from operator import itemgetter
import matplotlib

plt.rcParams['font.size'] = 16
matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'
watermark_types = [
    ["rohith23"],
    ["john23", "aiwei23", "xiaoniu23", "aiwei23b", "scott22"],
    ["xuandong23b"],
    
    ["john23", "xuandong23b", "aiwei23","aiwei23b"],
    [  "xiaoniu23"],
    [ "rohith23", "scott22"] ,
    
    [ "john23",  "aiwei23b", "scott22","xuandong23b", "rohith23"],
    [ "xiaoniu23", ],1
    ["aiwei23"]
]
replace_dict = {
    "john23": "TGRL",
    "xuandong23b": "UG",
    "aiwei23": "UPV",
    "rohith23": "RDF",1
    "xiaoniu23": "UB",
    "lean23": "CTWL",
    "scott22": "GO",
    "aiwei23b": "SIR",
}

watermark_colors = {
    "rohith23": "orange",
    "xuandong23b": "deepskyblue",
    "john23": "limegreen",
    "aiwei23": "purple",
    "xiaoniu23": "magenta",
    "aiwei23b": "red",
    "scott22": "royalblue"
}


attack_categories = {
    "Textual Integrity Attacks": ["synonym-0.4", "MisspellingAttack", "TypoAttack","swap"],
    "Content Manipulation": ["copypaste-1-10", "copypaste-3-10","copypaste-1-25","copypaste-3-25"],
    "Linguistic Variation Attacks": ["ContractionAttack", "ExpansionAttack", "LowercaseAttack"],
    "Paraphrase Attack": ["dipper_l20_o0","dipper_l40_o0","dipper_l40_o20",   "dipper_l60_o20","translation"]
}
attack_replace_dict = {
    "synonym-0.4": "Syno",
    "MisspellingAttack": "Missp",
    "TypoAttack": "Typo",
    "swap": "Swap",
    "copypaste-1-10": "CP1-10",
    "copypaste-3-10": "CP3-10",
    "copypaste-1-25": "CP1-25",
    "copypaste-3-25": "CP3-25",
    "ContractionAttack": "Contra",
    "LowercaseAttack": "LowCase",
    "ExpansionAttack": "Expan",
    "dipper_l20_o0": "DP-20",
    "dipper_l40_o0": "DP-40",
    "dipper_l40_o20": "DP-40-20",
    "dipper_l60_o20": "DP-60-20",
    "translation": "Trans",
}
dataset = 'c4'
cmap = plt.cm.coolwarm


fig, axs = plt.subplots(len(watermark_types), len(attack_categories), figsize=(16, len(watermark_types) * 16 / len(attack_categories)), subplot_kw=dict(projection='polar')) 
plt.subplots_adjust(wspace=0.5, hspace=0.3)  # Adjust the space between subplots

for i, watermark_type in enumerate(watermark_types):
    for j, (attack_category, attacks) in enumerate(attack_categories.items()):
        
        dim_num = len(attacks)
        radians = np.linspace(0, 2 * np.pi, dim_num, endpoint=False)
        radians = np.concatenate((radians, [radians[0]]))

        data = DefaultDict(list)
        with open('/home/ljc/sok-llm-watermark/plot/newplot/output/output_c4_opt.txt', newline='') as fp:
            reader = csv.reader(fp, delimiter=',')
            for row in reader:
                method, attack, val = row[0], row[1], float(row[2])
                if (method in watermark_type) and (attack in attacks) :
                # if (method in watermark_type) and (attack in attacks) and (row[3] == dataset):
                    data[method].append((attack, val))
        for k, v in data.items():
            # color=cmap((i * len(watermark_type) + j) / (len(watermark_types) * len(attack_categories)))
            color = watermark_colors[k]
            v = np.array([x[1] for x in sorted(v, key=itemgetter(0))])
            v = np.concatenate((v, [v[0]]))

            axs[i, j].plot(radians, v, color=color, linewidth=2.5, label=replace_dict[k])
            axs[i, j].fill(radians, v, color=color, alpha=0.1)
        radar_labels = sorted(attacks)
        radar_labels = np.concatenate((radar_labels, [radar_labels[0]]))
        angles = radians * 180/np.pi
        # axs[i, j].set_thetagrids(angles, labels=radar_labels)
        axs[i, j].set_thetagrids(angles, labels=[attack_replace_dict[attack] for attack in radar_labels])
        axs[i, j].set_rticks([0.5, 0.9])  # Set radial ticks
        # axs[i, j].legend(loc='upper right')
        axs[i, j].legend(loc='upper right', bbox_to_anchor=(1.3, 1.4))
plt.tight_layout()
plt.savefig('./plot/newplot/output/tpr_circle.pdf')