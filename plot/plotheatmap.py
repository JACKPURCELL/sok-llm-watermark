# -*- coding: utf-8 -*-
"""plotheatmap.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pVZ2zxMo0sC_yjs55FXV_zJwnS33D1B1
"""

import matplotlib.pyplot as mp
import seaborn as sb
import numpy as np

sb.color_palette("coolwarm", as_cmap=True)

# import file with data

data = {}
data["john23"] = [8,161,8,498]  # g0s0, g1s0, g0s1, g1s1
data["rohith23"] = [11,31,67,566]
data["xiaoniu23"] = [19,516,3,137]
data["xuandong23b"] = [21,59,4,591]
data["aiwei23"] = [43,626,1,5]
data["aiwei23b"] = [14,138,21,502]
data["scott22"] = [3,81,19,572]

for k, v in data.items():
    data[k] = (np.array(v)/(np.sum(v))).reshape((2, 2)).T

# cmap = mp.cm.coolwarm

replace_dict = {
    "john23": "TGRL",
    "xuandong23b": "UG",
    "aiwei23": "UPV",
    "rohith23": "RDF",
    "xiaoniu23": "UB",
    "scott22": "GO",
    "aiwei23b": "SIR",
}

# applying mask
# mask = np.triu(np.ones_like(data))

fig, axes = mp.subplots(2, 4)

sb.set(font_scale=0.5)

for k, name in enumerate(data):
    i, j = k // 4, k % 4
    ax = sb.heatmap(data[name], linewidths = 2, linecolor = 'w', square = True, annot = True, fmt = ".2f", vmin=0, vmax=1, cmap='coolwarm', ax=axes[i,j])
    ax.set(xlabel="Specific", ylabel="Generic", title = replace_dict[name])


mp.savefig(f'/home/ljc/sok-llm-watermark/plot/generic.pdf', transparent = True, format="pdf", bbox_inches="tight")

