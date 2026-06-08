#!/usr/bin/env uv
# /// script
# dependencies = [
#     "pandas",
#     "seaborn",
#     "matplotlib",
#     "SciencePlots"
# ]
# ///
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use("science")


def pt_to_inch(pt: float) -> float:
    return pt / 72.27


aspect = (1 + 5**0.5) / 2
width = 390  # pt
height = width / aspect
figsize = (pt_to_inch(width), pt_to_inch(height))

df = pd.read_csv("logs/evaluate/data_frame.csv")
df = df[df.detector != "oracle"]
df = df[df.detector != "best"]

fig, ax = plt.subplots(figsize=figsize)

sns.scatterplot(
    data=df,
    x="dd.wasserstein_distance",
    y="ocl.accuracy_seen_avg",
    hue="detector",
    style="strategy",
    ax=ax,
)
plt.xscale("log")
ax.legend(ncol=2)

plt.savefig("fig/dd_scatter.pdf", bbox_inches="tight")
