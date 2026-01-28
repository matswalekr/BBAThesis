import matplotlib.pyplot as plt
from cycler import cycler


def set_plot_style() -> None:

    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams["axes.prop_cycle"] = cycler(color=["#4C72B0", "#55A868", "#C44E52"])
