from cycler import cycler
from matplotlib.colors import ListedColormap

# The "science" then "nature" styles of SciencePlots, copied here to drop the package.
# Text is set with LaTeX, so it needs a TeX install.
PAPER_STYLE = {
    "axes.prop_cycle": cycler(
        color=["#0C5DA5", "#00B945", "#FF9500", "#FF2C00", "#845B97", "#474747", "#9e9e9e"]
    ),
    "figure.figsize": (3.3, 2.5),
    "xtick.direction": "in",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": True,
    "ytick.direction": "in",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": True,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "font.size": 7,
    "font.family": "sans-serif",
    "font.sans-serif": [
        "DejaVu Sans", "Arial", "Helvetica", "Lucida Grande", "Verdana", "Geneva",
        "Lucid", "Avant Garde", "sans-serif",
    ],
    "mathtext.fontset": "dejavusans",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{sfmath}",
}

atab10 = ListedColormap(
    [
        "#ff9ec4",
        "#cb04a9",
        "#5a243b",
        "#ce1513",
        "#ff9208",
        "#cbe82c",
        "#259320",
        "#16cde4",
        "#8f67f9",
        "#0229b0",
    ]
)

rename_methods = {"ADWIN_JOINT": "ADWIN*", "oracle": "Oracle"}

# SEED and STEPD runs use the one-sided variants. Show them under the plain names.
ONE_SIDED_LABELS = {"SEED_ONE_SIDED": "SEED", "STEPD_ONE_SIDED": "STEPD"}


# One column of the paper is 252 pt (3.5 in) wide. Plot at this size so text is not
# scaled down when the figure is included with width=\linewidth.
width = 252
scale = 1 / 72

FIGSIZE_43 = (width * scale, width * scale / (4 / 3))

# Wide plots with 1 to 3 panels (golden ratio)
FIGSIZE_SR = (width * scale, width * scale / 1.618)

# Tall plots with stacked panels
FIGSIZE_TALL = (width * scale, width * scale * 1.1)


# --- Colours (Okabe-Ito, safe for colour-blind readers) ---
# Every figure takes its colours from here so one meaning has one colour.
STRATEGY_PALETTE = {
    "FT": "#999999",
    "EWC": "#0072B2",
    "SI": "#56B4E9",
    "RWalk": "#009E73",
    "MAS": "#E69F00",
    "LWF": "#D55E00",
}

# Drift speed, from fast to slow (dark to light)
BOUNDARY_PALETTE = {
    "abrupt": "#7A3B63",
    "gradual": "#CC79A7",
    "slow": "#EBC3D8",
}

DETECTOR_PALETTE = {
    "oracle": "#000000",
    "ADWIN*": "#F0E442",
    "Best": "#999999",
}

# Error stream plots
TRUE_DRIFT_COLOR = "#009E73"
PRED_DRIFT_COLOR = "#D55E00"
DRIFT_WINDOW_COLOR = "#DDDDDD"

# No frame or ticks on the top and right
NO_TOP_RIGHT = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.top": False,
    "ytick.right": False,
}

# Text sizes in pt (the figure is 1:1 on paper, the caption is 8 pt), from large to small
FONT_TITLE, FONT_SUBTITLE, FONT_LABEL, FONT_TICK = 9, 8, 7, 6

# Colour of the red panel labels, such as (a), that the captions refer to
PANEL_LABEL_COLOR = "#D00000"

# --- Labels (sentence case) ---
LABEL_F1 = "F1 score"
LABEL_ACC = "Accuracy"
LABEL_NMDT = "NMDT"
