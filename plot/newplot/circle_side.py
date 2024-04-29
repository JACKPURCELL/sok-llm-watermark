from matplotlib.lines import Line2D
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
    
    ["john23", "xuandong23b", "aiwei23",],
    [ "aiwei23b", "xiaoniu23"],
    [ "rohith23", "scott22"] ,
    
    [ "john23",  "aiwei23b", "scott22","xuandong23b", "rohith23"],
    [ "xiaoniu23", ],
    ["aiwei23"]
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


attack_categories = {
    "Textual Integrity Attacks": ["synonym-0.4", "MisspellingAttack", "TypoAttack","swap"],
    "Content Manipulation": ["copypaste-1-10", "copypaste-3-10","copypaste-1-25","copypaste-3-25"],
    "Linguistic Variation Attacks": ["ContractionAttack", "ExpansionAttack", "LowercaseAttack"],
    "Paraphrase Attack": ["dipper_l20_o0","dipper_l40_o0","translation"]
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


for j, (attack_category, attacks) in enumerate(attack_categories.items()):
    # fig, axs = plt.subplots(3, 3, figsize=(10, 11), subplot_kw=dict(projection='polar')) 
    fig, axs = plt.subplots(3, 3, figsize=(10, 11), subplot_kw=dict(projection='polar', aspect='equal'))
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the space between subplots
    legend_elements = []
    # plt.subplots_adjust(wspace=0.2, hspace=0.2, top=0.4, bottom=0.15, left=0.1, right=0.9)
    plt.subplots_adjust(wspace=0.3, hspace=0.3, top=0.8, bottom=0.1, left=0.1, right=0.9)
    for k, rp in replace_dict.items():
        legend_elements.append(Line2D([0], [0], color=watermark_colors[k], lw=2.5, label=rp))
    # 在循环开始前定义y坐标
    # y_coords = [0.6, 0.34, 0.0]
    # texts = ['Context Dependency', 'Generation Strategy', 'Detection Strategy']

    for i, watermark_type in enumerate(watermark_types):
        # 在每个图左侧添加文字
        # if i % 3 == 0:
        #     plt.text(-0.1, y_coords[i//3], texts[i//3], transform=plt.gcf().transFigure, rotation=90)

        # # 在第一和第二行之间，以及第二和第三行之间添加分隔线
        # if i % 3 == 2:
        #     line = Line2D([-0.1, 1.1], [y_coords[i//3] - 0.15, y_coords[i//3] - 0.15], color='black', linewidth=2, transform=plt.gcf().transFigure, figure=plt.gcf())
        #     plt.gcf().lines.extend([line])

        dim_num = len(attacks)
        if dim_num == 4:
            offset = np.pi / 4 # 45 degrees in radians
        else:
            offset =  np.pi / 3+np.pi / 6
        # offset = 0
        radians = np.linspace(0 + offset, 2 * np.pi + offset, dim_num, endpoint=False)
        radians = np.concatenate((radians, [radians[0]]))

        data = DefaultDict(list)
        with open('/home/ljc/sok-llm-watermark/plot/newplot/output/output_c4_llama.txt', newline='') as fp:
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

            axs[i//3, i%3].plot(radians, v, color=color, linewidth=2.5, label=replace_dict[k])
            axs[i//3, i%3].fill(radians, v, color=color, alpha=0.1)
            

        radar_labels = sorted(attacks)
        radar_labels = np.concatenate((radar_labels, [radar_labels[0]]))
        angles = radians * 180/np.pi
        # if 90 in angles:
        #     angles = [angle + 45 for angle in angles]
        # if 0 < 120 - angles[1] <0.01:
        #     angles = [angle + 30 for angle in angles]
        axs[i//3, i%3].set_thetagrids(angles, labels=[attack_replace_dict[attack] for attack in radar_labels])
        axs[i//3, i%3].set_rticks([0.5, 0.9])  # Set radial ticks
        axs[i//3, i%3].set_rlim([0, 1])
        # axs[i//3, i%3].legend(loc='upper right', bbox_to_anchor=(1.3, 1.4))

    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))

    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # 调整布局以避免重叠
    plt.savefig(f'./plot/newplot/output/tpr_circle_{attack_category}_llama.pdf',bbox_inches='tight')