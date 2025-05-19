from matplotlib.lines import Line2D
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import csv
from operator import itemgetter
import matplotlib

plt.rcParams['font.size'] = 16
matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'

watermark_type = [
    "rohith23",
    "john23", "aiwei23", "xiaoniu23", "aiwei23b", "scott22",
    "xuandong23b",
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

attacks = ["dipper_l20_o0","dipper_l40_o0","dipper_l40_o20", "dipper_l60_o0", "dipper_l60_o20","dipper_l60_o40"]    

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
    "dipper_l60_o0": "DP-60",
    "dipper_l60_o40": "DP-60-40",
    "translation": "Trans",
}

fig, ax = plt.subplots(figsize=(7.5, 4), subplot_kw=dict(projection='polar'))

legend_elements = []

plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.8, bottom=0.1, left=0.05, right=1.0)


for k, rp in replace_dict.items():
    legend_elements.append(Line2D([0], [0], color=watermark_colors[k], lw=2.5, label=rp))

dim_num = len(attacks)
radians = np.linspace(0, 2 * np.pi, dim_num, endpoint=False)
radians = np.concatenate((radians, [radians[0]]))

data = defaultdict(list)
with open('./plot/newplot/output/output_c4_opt.txt', newline='') as fp:
    reader = csv.reader(fp, delimiter=',')
    for csv_row in reader:
        method, attack, val = csv_row[0], csv_row[1], float(csv_row[2])
        if (method in watermark_type) and (attack in attacks) :
            data[method].append((attack, val))

for k, v in data.items():
    color = watermark_colors[k]
    v = np.array([x[1] for x in sorted(v, key=itemgetter(0))])
    v = np.concatenate((v, [v[0]]))

    ax.plot(radians, v, color=color, linewidth=2.5, label=replace_dict[k])
    ax.fill(radians, v, color=color, alpha=0.1)

radar_labels = sorted(attacks)
radar_labels = np.concatenate((radar_labels, [radar_labels[0]]))
angles = radians * 180/np.pi
ax.set_thetagrids(angles, labels=[attack_replace_dict[attack] for attack in radar_labels])
ax.set_rticks([0.5, 0.9])

fig.legend(handles=legend_elements, loc='center right', ncol=1, bbox_to_anchor=(1.0, 0.5))

plt.tight_layout()
plt.savefig('./plot/newplot/output/dipper_circle.pdf', bbox_inches='tight')
print("./plot/newplot/output/dipper_circle.pdf")