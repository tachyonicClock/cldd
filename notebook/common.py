from matplotlib.colors import ListedColormap

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


width = 390
scale = 1 / 72

FIGSIZE_43 = (width * scale, width * scale / (4 / 3))

# SILVER RATIO
FIGSIZE_SR = (width * scale, width * scale / (1 + 2**0.5))
