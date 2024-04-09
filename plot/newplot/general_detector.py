import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from matplotlib.colors import to_rgb
plt.rcParams['font.size'] = 16
matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'
data = {}
data["john23"] = [52,205,39,704]  # g0s0, g1s0, g0s1, g1s1
data["lean23"] = [419,275,4,2]  # g0s0, g1s0, g0s1, g1s1
data["rohith23"] = [32,29,267,672]
data["xiaoniu23"] = [55,495,10,140]
data["xuandong23b"] = [40,69,23,868]
data["aiwei23"] = [129,563,1,7]
data["aiwei23b"] = [37,122,55,486]
data["scott22"] = [28,57,179,436]

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

for watermark_type in data:
    data[watermark_type][1], data[watermark_type][2] = data[watermark_type][2], data[watermark_type][1]

# Create figure and axes for a square divided into 4 equal parts
# fig, axs = plt.subplots(len(data), 1)
# fig, axs = plt.subplots(len(data), 1, figsize=(6, len(data)*6))  # Adjust the size of the figure
# rows = (len(data) + 1) // 2  # Calculate the number of rows needed
# fig, axs = plt.subplots(rows, 2, figsize=(12, rows*6))  # Adjust the size of the figure
cols = (len(data) + 1) // 2  # Calculate the number of columns needed
fig, axs = plt.subplots(2, cols, figsize=(cols*7, 15))  # Adjust the size of the figure

# Coordinates and size for equally sized squares
coords = [(0, 0.5), (0.5, 0.5), (0, 0), (0.5, 0)]
size = 0.5  # Half the dimension of the whole square

for idx, (watermark_type, values) in enumerate(data.items()):
    row = idx % 2
    col = idx // 2
    ax = axs[row, col]
    total = sum(values)
    percentages = [f"{x / total:.2%}" for x in values]
    percentages_value = [x / total for x in values]
    base_color = to_rgb('#ff6600')
    # Create and add each square patch with labels, using the predefined orange shades
    for i, coord in enumerate(coords):
        color = (*base_color, percentages_value[i])
        rect = patches.Rectangle(coord, size, size, linewidth=1, edgecolor='r', facecolor=color)
        ax.add_patch(rect)
        # Add text in the middle of the square
        ax.text(coord[0] + size/2, coord[1] + size/2, f'{percentages[i]}', ha="center", va="center", fontsize=16, weight='bold')

    # Set limits and aspect ratio to show a square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Adding X and Y axis labels
    ax.set_xlabel('Specific')
    ax.xaxis.set_label_position('top')  # Set X axis label position to top
    ax.xaxis.tick_top()  # Set X axis ticks to top
    ax.set_ylabel('General')

    # Setting the ticks on the X and Y axes to align with the center of the quadrants
    ax.set_xticks([0.25, 0.75])
    ax.set_yticks([0.25, 0.75])

    # Custom tick labels to show '0' and '1' at the desired positions
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['1', '0'])

    ax.set_title(replace_dict[watermark_type])
# plt.subplots_adjust(hspace=0.5) 
plt.subplots_adjust(hspace=1, wspace=1)  # Adjust the space between subplots
plt.tight_layout()
# plt.show()
plt.savefig('./plot/newplot/output/general_detector.pdf')