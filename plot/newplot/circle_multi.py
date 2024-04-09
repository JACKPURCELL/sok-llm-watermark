from typing_extensions import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
import csv
from operator import itemgetter
import matplotlib
from matplotlib.lines import Line2D

plt.rcParams['font.size'] = 16
matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'
# watermark_types = [
#     ["rohith23", "xuandong23b"],
#     ["john23", "aiwei23", "xiaoniu23", "aiwei23b", "scott22"],
    
#     ["john23", "xuandong23b", "aiwei23","aiwei23b"],
#     [  "xiaoniu23"],
#     [ "rohith23", "scott22"] ,
    
#     [ "john23",  "aiwei23b", "scott22","xuandong23b", "rohith23"],
#     [ "xiaoniu23", "aiwei23"],
# ]
watermark_types = [
    ["rohith23", "xuandong23b","john23", "scott22"],
    [ "aiwei23", "xiaoniu23", "aiwei23b"],
    
]
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

watermark_colors = {
    "rohith23": "orange",
    "xuandong23b": "deepskyblue",
    "john23": "limegreen",
    "aiwei23": "purple",
    "xiaoniu23": "magenta",
    "aiwei23b": "red",
    "scott22": "royalblue"
}



attack_categories =  {
    "Contra": ["ContractionAttack",
               "ContractionAttack_multiple/ExpansionAttack",
    "ContractionAttack_multiple/LowercaseAttack",
    "ContractionAttack_multiple/MisspellingAttack",
    "ContractionAttack_multiple/swap",
    "ContractionAttack_multiple/TypoAttack",],
    
    "Swap": ["swap","swap_multiple/ContractionAttack",
    "swap_multiple/ExpansionAttack",
    "swap_multiple/LowercaseAttack",
    "swap_multiple/MisspellingAttack",
    "swap_multiple/TypoAttack",],
    
    "Syno": ["synonym-0.4","synonym-0.4_multiple/ContractionAttack",
    "synonym-0.4_multiple/ExpansionAttack",
    "synonym-0.4_multiple/LowercaseAttack",
    "synonym-0.4_multiple/MisspellingAttack",
    "synonym-0.4_multiple/swap",
    "synonym-0.4_multiple/TypoAttack"],
    
    "Typo":["TypoAttack","synonym-0.4_multiple/TypoAttack","ContractionAttack_multiple/TypoAttack","swap_multiple/TypoAttack"],
}

attack_replace_dict = {
 "ContractionAttack_multiple/ExpansionAttack": "+Expan",
    "ContractionAttack_multiple/LowercaseAttack": "+LowCase",
    "ContractionAttack_multiple/MisspellingAttack": "+Missp",
    "ContractionAttack_multiple/swap": "+Swap",
    "ContractionAttack_multiple/TypoAttack": "Contra+Typo",
    "swap_multiple/ContractionAttack": "+Contra",
    "swap_multiple/ExpansionAttack": "+Expan",
    "swap_multiple/LowercaseAttack": "+LowCase",
    "swap_multiple/MisspellingAttack": "+Missp",
    "swap_multiple/TypoAttack": "swap+Typo",
    "synonym-0.4_multiple/ContractionAttack": "+Contra",
    "synonym-0.4_multiple/ExpansionAttack": "+Expan",
    "synonym-0.4_multiple/LowercaseAttack": "+LowCase",
    "synonym-0.4_multiple/MisspellingAttack": "+Missp",
    "synonym-0.4_multiple/swap": "+Swap",
    "synonym-0.4_multiple/TypoAttack": "syno+Typo",
    # "TypoAttack": "Typo",
    # "synonym-0.4_multiple/TypoAttack": "+Syno",
    # "ContractionAttack_multiple/TypoAttack": "+Contra",
    # "swap_multiple/TypoAttack": "+Swap",
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
    "translation": "Trans",
}
dataset = 'c4'
cmap = plt.cm.coolwarm


fig, axs = plt.subplots(3, 3, figsize=(10, 11), subplot_kw=dict(projection='polar')) 
# plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the space between subplots
legend_elements = []
# plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.4, bottom=0.15, left=0.1, right=0.9)
plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.8, bottom=0.1, left=0.1, right=0.9)
for k, rp in replace_dict.items():
    legend_elements.append(Line2D([0], [0], color=watermark_colors[k], lw=2.5, label=rp))
    
for i, watermark_type in enumerate(watermark_types):
    for j, (attack_category, attacks) in enumerate(attack_categories.items()):
        
        dim_num = len(attacks)
        radians = np.linspace(0, 2 * np.pi, dim_num, endpoint=False)
        radians = np.concatenate((radians, [radians[0]]))

        data = DefaultDict(list)
        with open('./plot/newplot/output/output_c4_opt_multi.txt', newline='') as fp:
            reader = csv.reader(fp, delimiter=',')
            for row in reader:
                method, attack, val = row[0], row[1], float(row[2])
                if (method in watermark_type) and (attack in attacks) and (row[3] == dataset):
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
plt.savefig('./plot/newplot/output/tpr_circle_multi.pdf', dpi=300)