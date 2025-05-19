# Re-import required packages and redefine data due to kernel reset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Reconstruct the dataframe
latex_data = {
    "TGRL": {
        "CLEAN": 0.993, "Contra": 0.944, "Expan": 0.944, "LowCase": 0.831,
        "Swap": 0.429, "Typo": 0.222, "Syno": 0.887, "Missp": 0.915,
        "CP1-10": 0.000, "CP3-10": 0.667, "CP1-25": 0.833, "CP3-25": 0.833,
        "DP-20": 0.667, "DP-40": 0.485, "Trans": 0.222
    },
    "UG": {
        "CLEAN": 0.993, "Contra": 0.857, "Expan": 0.833, "LowCase": 0.976,
        "Swap": 0.929, "Typo": 0.930, "Syno": 0.976, "Missp": 0.738,
        "CP1-10": 0.857, "CP3-10": 0.714, "CP1-25": 0.714, "CP3-25": 0.991,
        "DP-20": 0.976, "DP-40": 0.877, "Trans": 0.921
    }
}
df = pd.DataFrame.from_dict(latex_data, orient='index')

# Metadata
attacks = [
    "Contra", "Expan", "LowCase", "Swap", "Typo", "Syno", "Missp",
    "CP1-10", "CP3-10", "CP1-25", "CP3-25", "DP-20", "DP-40", "Trans"
]

attack_replace_dict = {
    "Contra": "Contraction", "Expan": "Expansion", "LowCase": "Lowercase",
    "Swap": "Swap", "Typo": "Typo", "Syno": "Synonym", "Missp": "Misspell",
    "CP1-10": "CP1-10", "CP3-10": "CP3-10", "CP1-25": "CP1-25", "CP3-25": "CP3-25",
    "DP-20": "DP-20", "DP-40": "DP-40", "Trans": "Translation"
}

watermark_colors = {
    "TGRL": "limegreen",
    "UG": "deepskyblue"
}

# Radar chart
dim_num = len(attacks)
radians = np.linspace(0, 2 * np.pi, dim_num, endpoint=False)
radians = np.concatenate((radians, [radians[0]]))

fig, ax = plt.subplots(figsize=(8, 4.5), subplot_kw=dict(projection='polar'))
legend_elements = []

for method, values in df.iterrows():
    v = [values[atk] for atk in attacks]
    v = np.concatenate((v, [v[0]]))  # closing the loop
    color = watermark_colors[method]
    ax.plot(radians, v, color=color, linewidth=2.5, label=method)
    ax.fill(radians, v, color=color, alpha=0.1)
    legend_elements.append(Line2D([0], [0], color=color, lw=2.5, label=method))

# Label setup
radar_labels = [attack_replace_dict[atk] for atk in attacks]
radar_labels.append(radar_labels[0])
angles = radians * 180 / np.pi
ax.set_thetagrids(angles, labels=radar_labels)
ax.set_rticks([0.5, 0.9])
ax.set_rlabel_position(120)
fig.legend(handles=legend_elements, loc='center right', ncol=1, bbox_to_anchor=(1.0, 0.5))

plt.tight_layout()
plt.savefig('plot/newplot/output/radar_TGRL_UG.pdf', bbox_inches='tight')
'plot/newplot/output/radar_TGRL_UG.pdf'
