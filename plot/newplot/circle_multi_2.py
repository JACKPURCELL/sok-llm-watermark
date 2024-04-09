from typing_extensions import DefaultDict
import numpy as np
import matplotlib.pyplot as plt
import csv
from operator import itemgetter
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.lines import Line2D
plt.rcParams['font.size'] = 14
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
    ["rohith23", "xuandong23b","john23", "scott22","aiwei23", "xiaoniu23", "aiwei23b"],
    # [ "aiwei23", "xiaoniu23", "aiwei23b"],
    
]
replace_dict = {
    "john23": "TGRL",
    "xuandong23b": "UG",
    "aiwei23": "UPV",
    "rohith23": "RDF",
    "xiaoniu23": "UB",
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
    "Single": ["ContractionAttack","swap","synonym-0.4","TypoAttack","LowercaseAttack","MisspellingAttack"
               ],
    
    "Multi": [
"ContractionAttack_multiple/TypoAttack",
"ContractionAttack_multiple/swap",
"ContractionAttack_multiple/LowercaseAttack",
    "swap_multiple/LowercaseAttack",
    "swap_multiple/MisspellingAttack",
    "swap_multiple/TypoAttack",

    "synonym-0.4_multiple/swap",
    "synonym-0.4_multiple/LowercaseAttack",
    "synonym-0.4_multiple/TypoAttack",
    ],
    

    
    
}

attack_replace_dict = {
 "ContractionAttack_multiple/ExpansionAttack": "Contra+Expan",
    "ContractionAttack_multiple/LowercaseAttack": "Contra+LowCase",
    "ContractionAttack_multiple/MisspellingAttack": "Contra+Missp",
    "ContractionAttack_multiple/swap": "Contra+Swap",
    "ContractionAttack_multiple/TypoAttack": "Contra+Typo",
    "swap_multiple/ContractionAttack": "Swap+Contra",
    "swap_multiple/ExpansionAttack": "Swap+Expan",
    "swap_multiple/LowercaseAttack": "Swap+LowCase",
    "swap_multiple/MisspellingAttack": "Swap+Missp",
    "swap_multiple/TypoAttack": "Swap+Typo",
    "synonym-0.4_multiple/ContractionAttack": "Syno+Contra",
    "synonym-0.4_multiple/ExpansionAttack": "Syno+Expan",
    "synonym-0.4_multiple/LowercaseAttack": "Syno+LowCase",
    "synonym-0.4_multiple/MisspellingAttack": "Syno+Missp",
    "synonym-0.4_multiple/swap": "Syno+Swap",
    "synonym-0.4_multiple/TypoAttack": "Syno+Typo",
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


fig, axs = plt.subplots(1, 2, figsize=(8, 4.5), subplot_kw=dict(projection='polar')) 
# plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the space between subplots
legend_elements = []
# plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.4, bottom=0.15, left=0.1, right=0.9)
plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.8, bottom=0.1, left=0.1, right=0.9)
for k, rp in replace_dict.items():
    legend_elements.append(Line2D([0], [0], color=watermark_colors[k], lw=2.5, label=rp))
    
for i, watermark_type in enumerate(watermark_types):
    for j, (attack_category, attacks) in enumerate(attack_categories.items()):
        
        dim_num = len(attacks)
        if dim_num == 4:
            offset = 0 # 45 degrees in radians
        else:
            offset =  0
        # offset = 0
        radians = np.linspace(0 + offset, 2 * np.pi + offset, dim_num, endpoint=False)
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

            axs[j].plot(radians, v, color=color, linewidth=2.5, label=replace_dict[k])
            axs[j].fill(radians, v, color=color, alpha=0.1)
        radar_labels = sorted(attacks)
        radar_labels = np.concatenate((radar_labels, [radar_labels[0]]))
        angles = radians * 180/np.pi
        # axs[i, j].set_thetagrids(angles, labels=radar_labels)
        axs[j].set_thetagrids(angles, labels=[attack_replace_dict[attack] for attack in radar_labels])
        axs[j].set_rticks([0.5, 0.9])  # Set radial ticks
        # axs[i, j].legend(loc='upper right')
        # axs[j].legend(loc='upper right', bbox_to_anchor=(1.3, 1.4))

        
fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
plt.tight_layout(rect=[0, 0, 1, 0.93])  # 调整布局以避免重叠

# plt.tight_layout()
plt.savefig('./plot/newplot/output/tpr_circle_multi_2.pdf',bbox_inches='tight')