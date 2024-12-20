import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import matplotlib

# Set the color palette to 'coolwarm'
sb.set_theme(palette="coolwarm")
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
# Increase the size of the title
plt.rcParams['axes.titlesize'] = 20
# Increase the size of tick labels
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
# matplotlib.rcParams['font.family'] = 'optima'
# matplotlib.rcParams['font.weight'] = 'medium'
# Data initialization and normalization
data = {
    #g0s0,g1s0,g0s1,g1s1
"john23": [0.74, 0.21, 0.05, 0.01],
"rohith23": [0.42, 0.08, 0.42, 0.08],
"xiaoniu23": [0.49, 0.49, 0.01, 0.01],

"xuandong23b": [0.09, 0.53, 0.07, 0.3],
"aiwei23": [0.86, 0.01, 0.13, 0.0],
"aiwei23b": [0.98, 0.01, 0.01, 0.0],
"scott22": [0.89, 0.04, 0.06, 0.01],

}
for k, v in data.items():
    data[k] = (np.array(v) / np.sum(v)).reshape((2, 2)).T
replace_dict = {
    "john23": "TGRL",
    "xuandong23b": "UG",
    "aiwei23": "UPV",
    "rohith23": "RDF",
    "xiaoniu23": "UB",
    "scott22": "GO",
    "aiwei23b": "SIR",
}
# Create the figure and axes for the subplots
fig = plt.figure(figsize=(16, 9))  # Adjust figsize to match your example
plt.subplots_adjust(hspace=0.2, wspace=0.2)
# Plot the heatmaps
for i, (name, matrix) in enumerate(data.items()):
    # Determine position of the subplot
    if i < 3:  # First row
        ax = fig.add_subplot(2, 4, i + 1)  # +2 to center the first row
    else:  # Second row
        ax = fig.add_subplot(2, 4, i + 2)
    # Plot the heatmap
    sb.heatmap(matrix, annot=True, fmt=".2f", linewidths=3, ax=ax, cbar=False,
               square=True, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_title(replace_dict[name])
    # Remove x and y labels
    if i == 3:
        ax.set_xlabel('Specific')
    if i == 0 or i ==3:
        ax.set_ylabel('Generic')
    # Optionally, remove the tick labels as well if needed:
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
# Add a colorbar to the right of the heatmaps
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # x, y, width, height
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []  # Fake up the array of the scalar mappable to avoid errors
fig.colorbar(sm, cax=cbar_ax)
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rect to fit the colorbar

plt.savefig('/home/jkl6486/sok-llm-watermark/plot/newplot/output/adv_heatmap.pdf')






