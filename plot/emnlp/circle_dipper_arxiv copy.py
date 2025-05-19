# Re-import required packages and redefine data due to kernel reset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import pandas as pd
plt.rcParams['font.size'] = 16
matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'

# Reconstruct the dataframe
latex_data = {
    "TGRL": {
        "CLEAN": 0.993, "Contra": 0.944, "Expan": 0.944, "LowCase": 0.831,
        "Swap": 0.429, "Typo": 0.222, "Syno": 0.887, "Missp": 0.915,
        "CP1-10": 0.000, "CP3-10": 0.667, "CP1-25": 0.833, "CP3-25": 0.833,
        "DP-20": 0.667, "DP-40": 0.485,"DP-60": 0.287, "Trans": 0.222
    },
    "GO": {
        "CLEAN": 0.996, "Contra": 0.996, "Expan": 0.996, "LowCase": 0.994,
        "Swap": 0.982, "Typo": 0.956, "Syno": 0.986, "Missp": 0.996,
        "CP1-10": 0.864, "CP3-10": 0.887, "CP1-25": 0.946, "CP3-25": 0.982,
        "DP-20": 0.640, "DP-40": 0.560,"DP-60": 0.276, "Trans": 0.667
    }
}
df = pd.DataFrame.from_dict(latex_data, orient='index')

# Metadata
attacks = [
    "Contra", "Expan", "LowCase", "Swap", "Typo", "Syno", "Missp",
    "CP1-10", "CP3-10", "CP1-25", "CP3-25", "DP-20", "DP-40", "DP-60", "Trans"
]


watermark_colors = {
    "TGRL": "limegreen",
    "GO": "royalblue"
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
radar_labels = [atk for atk in attacks]
radar_labels.append(radar_labels[0])
angles = radians * 180 / np.pi
ax.set_thetagrids(angles, labels=radar_labels)
ax.set_ylim(0, 1.0)
ax.set_rticks([0.5, 0.9])
ax.set_rlabel_position(120)
fig.legend(handles=legend_elements, loc='center right', ncol=1, bbox_to_anchor=(1.0, 0.5))

plt.tight_layout()
plt.savefig('plot/emnlp/radar_TGRL_GO.pdf', bbox_inches='tight')
'plot/emnlp/radar_TGRL_GO.pdf'
