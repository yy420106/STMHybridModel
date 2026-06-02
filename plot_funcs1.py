import os
from datetime import datetime
from typing import Any, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import numpy as np
import seaborn as sns
from colorama import Fore, Style, init
from cycler import cycler
from matplotlib.colors import ListedColormap
from model import STMGeneModel
from util_funcs import calc_prot_fixed_points

__all__ = [
    "verbose_print",
    "format_save",
    "plot_hint_print",
    "plot_schematic_diagram1",
    "plot_schematic_diagram2",
    "plot_evolution_heatmap",
    "plot_cell_distribution_after_recov",
    "plot_bistability_map",
]

# global variables & settings
init(autoreset=True)  # colorama mode: autoreset

"""
Global Function Parameters Explanation
======================================>
figName : str
    Figure/Image name.
figSize : Sequence[float]
    Size of figure (unit: inch).
figFormat : str | Sequence[str]
    Export image format (png, jpg, tiff, pdf, ...).
dpi : int32
    Dots per inch.
save_dir : str
    Directory to save the exported figure.
quiet : bool
    Control flag.
<======================================
"""


def verbose_print(message: str, quiet: bool) -> None:
    """
    Auxilary function. Control the print behavior.

    Parameters
    ----------
    message : str
        Information to be printed.
    """

    if not quiet:
        print(message)


def format_save(
    figObj: plt.Figure,
    figName: str,
    figFormat: Union[str, Sequence[str]],
    save_dir: str,
    **kwargs: Any,
) -> None:
    """
    Auxilary function. Output files in different formats as required.

    Parameters
    ----------
    figObj : Figure
        Matplotlib.Figure object.
    **kwargs : Any
        Other acceptable keyword arguments of function `Matplotlib.Figure.savefig`.
    """

    if isinstance(figFormat, str):
        figObj.savefig(os.path.join(save_dir, f"{figName}.{figFormat}"), **kwargs)
    else:
        for fmt in figFormat:
            figObj.savefig(os.path.join(save_dir, f"{figName}.{fmt}"), **kwargs)


def plot_hint_print(func_name: str, quiet: bool, hint_type: str, **kwargs: Any) -> None:
    """
    Auxilary function. Print different types of hint information.

    Parameters
    ----------
    func_name : str
        Function name.
    hint_type : str
        Type of hint information.
    **kwargs : Any
        Other necessary keywords argument. See detailed definition.
    """

    if hint_type == "plot":
        verbose_print(
            message=f"\t{Fore.GREEN + Style.BRIGHT}<{func_name}>{Style.RESET_ALL} Plotting ...",
            quiet=quiet,
        )

    elif hint_type == "save":
        format: Union[str, Sequence[str]] = kwargs["format"]  # `format` must be provided
        verbose_print(
            message=f"\t{Fore.GREEN + Style.BRIGHT}<{func_name}>{Style.RESET_ALL} Saving {'/'.join(format)} ...",
            quiet=quiet,
        )

    elif hint_type == "end":
        time: Tuple[datetime, datetime] = kwargs["time"]  # `time` must be provided
        verbose_print(
            message=(
                f"\t{Fore.GREEN + Style.BRIGHT}<{func_name}>{Style.RESET_ALL} Done. "
                f"(Finished in {Fore.RED + Style.BRIGHT + str(time[1] - time[0]) + Style.RESET_ALL})"
            ),
            quiet=quiet,
        )

    else:
        raise ValueError("Unrecognizable hint type.")


def plot_schematic_diagram1(
    figName: Optional[str] = None,
    figSize: Optional[Sequence[float]] = None,
    figFormat: Union[str, Sequence[str], None] = None,
    dpi: Optional[int] = None,
    save_dir: Optional[str] = None,
    quiet: bool = False,
) -> None:
    """
    Base function for plotting model schematic diagram to illustrate protein dynamics.

    **NOTE** The data used in this plot are all self-designed, just in order to show the trend of
    mathematical change, therefore they do not represent any pratical significance.
    """

    p_start = datetime.now()

    # default set
    figName = "schematic_diagram1" if figName is None else figName
    figSize = (8, 8) if figSize is None else tuple(figSize)
    figFormat = ["pdf"] if figFormat is None else figFormat
    dpi = 800 if dpi is None else dpi
    save_dir = os.getcwd() if save_dir is None else save_dir

    # plot figure
    plot_hint_print("plot_schematic_diagram1", quiet, "plot")
    plt.rc("font", family="Arial")

    fig = plt.figure(num=figName, figsize=figSize)
    ax = axisartist.Subplot(fig, 111)
    ax = fig.add_axes(ax)

    # example data (self-designed, no practical significance)
    x = np.linspace(0, 26, 261)  # protein level
    y0 = 5 * x  # protein production
    yt = (
        125 * x**2 / (100 + x**2)
    )  # top limit of protein degradation, corresponding to the weakest methylation
    yb = (
        50 * x**2 / (100 + x**2)
    )  # bottom limit of protein degradation, corresponding to the strongest methylation

    # line-plot & add graded shadow
    BWR = plt.colormaps["bwr"](np.linspace(0.2, 0.8, 301))
    for idx, val in enumerate(np.linspace(50, 125, 301)):
        yi = val * x**2 / (100 + x**2)
        ax.plot(
            x, yi, linewidth=1, color=BWR[idx], alpha=0.3
        )  # color stands for the graded repression state

    ax.plot(x, yt, color="red", linewidth=2, label="Production (me0)")
    ax.plot(x, yb, color="blue", linewidth=2, label="Production (me3)")
    ax.plot(x, y0, color="limegreen", linewidth=2, label="Degredation")

    # add other graph elements
    ax.vlines(x=5, ymin=15, ymax=110, linestyles="dashed", colors="black", linewidth=1)
    ax.vlines(x=20, ymin=45, ymax=110, linestyles="dashed", colors="black", linewidth=1)

    ax.arrow(
        4,
        25,
        -3,
        0,
        head_width=2,
        head_length=1,
        linewidth=1,
        color="black",
        length_includes_head=True,
    )
    ax.arrow(
        6,
        100,
        13,
        0,
        head_width=2,
        head_length=1,
        linewidth=1,
        color="black",
        length_includes_head=True,
    )
    ax.arrow(
        25,
        100,
        -4,
        0,
        head_width=2,
        head_length=1,
        linewidth=1,
        color="black",
        length_includes_head=True,
    )
    ax.arrow(
        25,
        25,
        -12,
        0,
        head_width=2,
        head_length=1,
        linewidth=1,
        color="black",
        length_includes_head=True,
    )

    ax.text(x=0.3, y=28, s="differentiate", fontdict={"size": 13, "color": "red", "weight": "bold"})
    ax.text(x=10, y=103, s="self-sustain", fontdict={"size": 13, "color": "red", "weight": "bold"})
    ax.text(x=20.9, y=94, s="self-sustain", fontdict={"size": 13, "color": "red", "weight": "bold"})
    ax.text(x=17, y=28, s="differentiate", fontdict={"size": 13, "color": "blue", "weight": "bold"})
    ax.text(x=18.2, y=114, s="Stem Cell", fontdict={"size": 14, "color": "black", "weight": "bold"})

    ax.annotate(
        text="metastable\n  (critical)",
        xy=(5, 25),
        xytext=(8, 32),
        fontsize=14,
        color="dimgray",
        arrowprops={"width": 0.05, "headlength": 0.01, "color": "dimgray"},
    )
    ax.annotate(
        text="stable",
        xy=(20, 100),
        xytext=(21, 85),
        fontsize=14,
        color="dimgray",
        arrowprops={"width": 0.05, "headlength": 0.01, "color": "dimgray"},
    )

    ax.set_xlim([0, 27])
    ax.set_ylim([0, 135])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Protein Number")
    ax.set_ylabel("Propensity")
    ax.legend(frameon=False, loc="upper left", fontsize=15)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    for side in ["bottom", "left"]:
        ax.axis[side].set_axisline_style("-|>", size=1.5)  # arrow axis
        ax.axis[side].label.set_fontsize(16)
        ax.axis[side].label.set_weight("bold")
        ax.axis[side].label.set_pad(10)

    # save figure
    plot_hint_print("plot_schematic_diagram1", quiet, "save", format=figFormat)
    format_save(fig, figName, figFormat, save_dir, dpi=dpi, bbox_inches="tight")

    p_end = datetime.now()
    plot_hint_print("plot_schematic_diagram1", quiet, "end", time=(p_start, p_end))


def plot_schematic_diagram2(
    figName: Optional[str] = None,
    figSize: Optional[Sequence[float]] = None,
    figFormat: Union[str, Sequence[str], None] = None,
    dpi: Optional[int] = None,
    save_dir: Optional[str] = None,
    quiet: bool = False,
) -> None:
    """Base function for plotting model schematic diagram to illustrate branch theory."""

    p_start = datetime.now()

    # default set
    figName = "schematic_diagram2" if figName is None else figName
    figSize = (8, 8) if figSize is None else tuple(figSize)
    figFormat = ["pdf"] if figFormat is None else figFormat
    dpi = 800 if dpi is None else dpi
    save_dir = os.getcwd() if save_dir is None else save_dir

    # plot figure
    plot_hint_print("plot_schematic_diagram2", quiet, "plot")
    plt.rc("font", family="Arial")
    plt.rc("mathtext", fontset="stix")

    fig = plt.figure(num=figName, figsize=(8, 8))
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)

    # prepare data
    fixed_points_arr = np.full((70, 3), np.nan, np.float64)
    P_me23_arr = np.concatenate(
        (
            np.linspace(0, 0.33, 33, endpoint=False),
            np.linspace(0.33, 0.34, 20, endpoint=False),
            np.linspace(0.34, 0.5, 17),
        )
    )
    for i, Pme23 in enumerate(P_me23_arr):
        fixed_points = calc_prot_fixed_points(Pme23, *STMGeneModel().prot_fp_param)
        fixed_points_arr[i, : len(fixed_points)] = fixed_points

    ax.plot(P_me23_arr, fixed_points_arr[:, 0], linewidth=5, color="blue")
    ax.plot(P_me23_arr, fixed_points_arr[:, 1], linewidth=1, linestyle="--", color="limegreen")
    ax.plot(P_me23_arr, fixed_points_arr[:, 2], linewidth=5, color="red")
    vanish_idx = np.searchsorted(np.isnan(fixed_points_arr[:, 2]), True)
    if vanish_idx != 0:
        ax.vlines(
            P_me23_arr[vanish_idx - 1],
            fixed_points_arr[vanish_idx - 1, 0],
            fixed_points_arr[vanish_idx - 1, 2],
            linewidth=1,
            linestyles="--",
            color="darkmagenta",
        )

    ax.text(
        x=0.4,
        y=0.1,
        s="differentiated cell\nstable branch",
        fontdict={"size": 15, "color": "blue", "weight": "bold"},
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        x=0.6,
        y=0.91,
        s="stem cell\nstable branch",
        fontdict={"size": 15, "color": "red", "weight": "bold"},
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        x=0.2,
        y=0.33,
        s="critical line",
        fontdict={"size": 15, "color": "limegreen", "weight": "bold"},
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        x=0.74,
        y=0.4,
        s="branch line",
        fontdict={"size": 15, "color": "darkmagenta", "weight": "bold"},
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )

    ax.set_yscale("log")
    ax.set_ylim([10**-3, 10**3.3])
    ax.set_yticks(ticks=10 ** np.linspace(-3, 3, 7))
    ax.set_ylabel("Protein Number")
    ax.set_xlim([0, 0.53])
    ax.set_xticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel(r"Methylation Level ($P_{me2/me3}$)")
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    for side in ["bottom", "left"]:
        ax.axis[side].set_axisline_style("-|>", size=1.5)  # arrow axis
        ax.axis[side].label.set_fontsize(16)
        ax.axis[side].label.set_weight("bold")
        ax.axis[side].label.set_pad(10)
        ax.axis[side].major_ticklabels.set_fontsize(12)

    # save figure
    plot_hint_print("plot_schematic_diagram2", quiet, "save", format=figFormat)
    format_save(fig, figName, figFormat, save_dir, dpi=dpi, bbox_inches="tight")

    p_end = datetime.now()
    plot_hint_print("plot_schematic_diagram2", quiet, "end", time=(p_start, p_end))


def plot_evolution_heatmap(
    samples_meState: np.ndarray[Any, int],
    timeid_step: int,
    figName: Optional[str] = None,
    figSize: Optional[Sequence[float]] = None,
    figFormat: Union[str, Sequence[str], None] = None,
    dpi: Optional[int] = None,
    save_dir: Optional[str] = None,
    quiet: bool = False,
) -> None:
    """
    Base function for plotting representative model evolution heatmap.

    Parameters
    ----------
    samples_meState : NDArray[int32], shape (N, T, H)
        Gillespie simulation result of methylation state, with each dimension represents trial
        (N), sampling time (T) and histone (H), respectively.
    timeid_step : int32
        Step of index in time axis to plot.
    """

    p_start = datetime.now()

    # default set
    figName = "evolution_heatmap" if figName is None else figName
    figSize = (12, 3) if figSize is None else tuple(figSize)
    figFormat = ["pdf"] if figFormat is None else figFormat
    dpi = 800 if dpi is None else dpi
    save_dir = os.getcwd() if save_dir is None else save_dir

    # plot figure
    plot_hint_print("plot_evolution_heatmap", quiet, "plot")
    fig = plt.figure(figName, figSize)
    ax: plt.Axes = fig.add_axes([0.05, 0.1, 0.83, 0.8])
    cbar_ax: plt.Axes = fig.add_axes([0.9, 0.3, 0.025, 0.4])

    sns.heatmap(
        samples_meState[np.random.randint(samples_meState.shape[0]), ::timeid_step, :].T,
        cmap=ListedColormap([plt.colormaps.get_cmap("summer_r")(i / 3) for i in range(4)]),
        ax=ax,
        cbar_ax=cbar_ax,
        vmin=0,
        vmax=3,
    )

    ax.invert_yaxis()
    x2, y2 = ax.get_xlim()[1], ax.get_ylim()[1]
    ax.set_xlim(-x2 * 0.05, x2)
    ax.set_ylim(0, y2 * 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.annotate(
        "Time ",
        xy=(0.81 * x2, y2 * 1.07),
        xytext=(0.2 * x2, y2 * 1.07),
        arrowprops=dict(width=1, headwidth=5, color="black"),
        fontsize=16,
        font="Arial",
        weight="bold",
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.vlines(x=-x2 * 0.02, ymin=0.2 * y2, ymax=0.8 * y2, linewidth=2.5, color="black")
    ax.text(
        x=-x2 * 0.04,
        y=y2 * 0.5,
        s="Gene locus",
        fontsize=16,
        font="Arial",
        weight="bold",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
    )

    cbar_ax.set_title("H3K27", font="Arial", fontsize=14, weight="bold")
    cbar_ax.set_yticks(
        ticks=[3 / 8, 9 / 8, 15 / 8, 21 / 8],
        labels=["me0", "me1", "me2", "me3"],
        font="Arial",
        fontsize=13,
    )

    # save figure
    plot_hint_print("plot_evolution_heatmap", quiet, "save", format=figFormat)
    format_save(fig, figName, figFormat, save_dir, dpi=dpi, bbox_inches="tight")

    p_end = datetime.now()
    plot_hint_print("plot_evolution_heatmap", quiet, "end", time=(p_start, p_end))


def plot_cell_distribution_after_recov(
    arrest_days: Sequence[float],
    count_days: Sequence[float],
    diff_ratio: np.ndarray[Any, float],
    stem_ratio: np.ndarray[Any, float],
    figName: Optional[str] = None,
    figSize: Optional[Sequence[float]] = None,
    figFormat: Union[str, Sequence[str], None] = None,
    dpi: Optional[int] = None,
    save_dir: Optional[str] = None,
    quiet: bool = False,
) -> None:
    """
    Base function for plotting cell distribution after division recovery.

    Parameters
    ----------
    arrest_days: Sequence[float64]
        Cell dicision arrest length for testing.
    count_days: Sequence[float64]
        Number of days after divison recovery to collect statistical data.
    diff_ratio: NDArray[float64]
        Differentiated cell proportion data.
    stem_ratio: NDArray[float64]
        Stem-restored cell proportion data.
    """

    p_start = datetime.now()

    # default set
    figName = "cell_distribution_after_recov" if figName is None else figName
    figSize = (8, 8) if figSize is None else tuple(figSize)
    figFormat = ["pdf"] if figFormat is None else figFormat
    dpi = 800 if dpi is None else dpi
    save_dir = os.getcwd() if save_dir is None else save_dir

    # plot figure
    plot_hint_print("plot_cell_distribution_after_recov", quiet, "plot")
    plt.rc("font", family="Arial")
    fig = plt.figure(num=figName, figsize=figSize)
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)

    ax.scatter(
        np.tile(arrest_days, (len(count_days), 1)),
        diff_ratio,
        100,
        "darkred",
        marker="+",
        linewidths=2,
    )
    ax.scatter(
        np.tile(arrest_days, (len(count_days), 1)),
        stem_ratio,
        100,
        "darkblue",
        marker="+",
        linewidths=2,
    )

    diff_color = plt.get_cmap("autumn_r")(np.linspace(0.1, 1, len(count_days)))
    stem_color = plt.get_cmap("winter_r")(np.linspace(0.1, 1, len(count_days)))
    ax.set_prop_cycle(cycler(color=np.vstack(tup=(diff_color, stem_color))))

    def f(t):
        return int(t) if t == int(t) else t

    ax.plot(
        arrest_days,
        diff_ratio.T,
        linewidth=2.5,
        label=[f"differentiated (recov. {f(t)}d)" for t in count_days],
    )
    ax.plot(
        arrest_days,
        stem_ratio.T,
        linewidth=2.5,
        label=[f"pluripotency-restored (recov. {f(t)}d)" for t in count_days],
    )

    ax.set_xlim([arrest_days[0] - 1, arrest_days[-1] + 1])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xticks(ticks=arrest_days)
    ax.set_yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Cell Division Arrest (Day)")
    # legend = ax.legend(frameon=False, fontsize=12, loc="best")
    legend = ax.legend(frameon=False, fontsize=12, loc="center", bbox_to_anchor=(0.45, 0.9))
    legend.set_title(title="Cell Type", prop={"size": 13, "weight": "bold"})

    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    for side in ["bottom", "left"]:
        ax.axis[side].set_axisline_style("-|>", size=1.5)  # arrow axis
        ax.axis[side].label.set_fontsize(16)
        ax.axis[side].label.set_weight("bold")
        ax.axis[side].label.set_pad(10)
        ax.axis[side].major_ticklabels.set_fontsize(12)

    # save figure
    plot_hint_print("plot_cell_distribution_after_recov", quiet, "save", format=figFormat)
    format_save(fig, figName, figFormat, save_dir, dpi=dpi, bbox_inches="tight")

    p_end = datetime.now()
    plot_hint_print("plot_cell_distribution_after_recov", quiet, "end", time=(p_start, p_end))


def plot_bistability_map(
    bistabilityMap: np.ndarray[Any, float],
    axis0_ticks: np.ndarray[Any, float],
    axis1_ticks: np.ndarray[Any, float],
    T: float,
    scatter_points: Optional[Sequence[Tuple[float, float]]] = None,
    figName: Optional[str] = None,
    figSize: Optional[Sequence[float]] = None,
    figFormat: Union[str, Sequence[str], None] = None,
    dpi: Optional[int] = None,
    save_dir: Optional[str] = None,
    quiet: bool = False,
) -> None:
    """
    Base function for plotting k_me - Pdem bistabilityMap under different cell cycle.

    Parameters
    ----------
    bistabilityMap: NDArray[float64]
        2-D array to store bistability data of each (k_me, Pdem) set.
    axis0_ticks: NDArray[float64]
        1-D array that specify the vertical dimension ticks.
    axis1_ticks: NDArray[float64]
        1-D array that specify the horizontal dimension ticks.
    T: float64:
        Cell cyle.
    scatter_points: Sequence[Tuple[float, float]] (optional)
        Coordinates of points that to be scatter on plot.
    """

    p_start = datetime.now()

    # default set
    figName = "bistability_map" if figName is None else figName
    figSize = (8, 8) if figSize is None else tuple(figSize)
    figFormat = ["pdf"] if figFormat is None else figFormat
    dpi = 800 if dpi is None else dpi
    save_dir = os.getcwd() if save_dir is None else save_dir

    # plot figure
    plot_hint_print("plot_bistability_map", quiet, "plot")
    plt.rc("font", family="Arial")
    plt.rc("mathtext", fontset="stix")
    fig = plt.figure(num=figName, figsize=figSize)
    ax = fig.add_subplot(1, 1, 1)

    bmap = ax.imshow(bistabilityMap, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")  # heatmap

    # temperal variables for ticklabels
    axis0_log_len = np.log10(axis0_ticks[-1] / axis0_ticks[0])
    axis1_log_len = np.log10(axis1_ticks[-1] / axis1_ticks[0])

    axis0_ticklabels_min = np.ceil(np.log10(axis0_ticks[0]))
    axis0_ticklabels_max = np.floor(np.log10(axis0_ticks[-1]))
    axis0_ticklabels = np.arange(axis0_ticklabels_min, axis0_ticklabels_max + 1, dtype=np.int32)

    axis1_ticklabels_min = np.ceil(np.log10(axis1_ticks[0]))
    axis1_ticklabels_max = np.floor(np.log10(axis1_ticks[-1]))
    axis1_ticklabels = np.arange(axis1_ticklabels_min, axis1_ticklabels_max + 1, dtype=np.int32)

    x_len = axis1_ticks.size - 1
    y_len = axis0_ticks.size - 1

    if scatter_points is not None:
        for point in scatter_points:
            point_axis0, point_axis1 = point
            ax.scatter(
                x=np.log10(point_axis1 / axis1_ticks[0]) / axis1_log_len * x_len,
                y=np.log10(point_axis0 / axis0_ticks[0]) / axis0_log_len * y_len,
                s=150,
                marker="*",
                c="red",
            )

    ax.set_xticks(
        ticks=(axis1_ticklabels - np.log10(axis1_ticks[0])) / axis1_log_len * x_len,
        labels=[f"$10^{ ({int(k)}) }$" for k in axis1_ticklabels],
        fontsize=13,
    )
    ax.set_yticks(
        ticks=(axis0_ticklabels - np.log10(axis0_ticks[0])) / axis0_log_len * y_len,
        labels=[f"$10^{ ({int(k)}) }$" for k in axis0_ticklabels],
        fontsize=13,
    )
    ax.tick_params(width=2, length=5)
    ax.set_xlabel(
        r"Demethylation probability $P_{\mathrm{dem}}$",
        fontsize=16,
        weight="bold",
    )
    ax.set_ylabel(r"Methylation rate $k_{\mathrm{me}}$ (s$^{-1}$)", fontsize=16, weight="bold")
    ax.set_title(f"Cell Cycle: ${ ({round(T, 1)}) }$ h", fontsize=16, weight="bold")
    ax.set_aspect(1)

    cb = fig.colorbar(mappable=bmap, fraction=0.03, pad=0.03, shrink=18)
    cb.ax.set_title("$B$", fontsize=18, weight="bold")
    cb.ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.ax.tick_params(width=1, length=5, labelsize=11)

    # save figure
    plot_hint_print("plot_bistability_map", quiet, "save", format=figFormat)
    format_save(fig, figName, figFormat, save_dir, dpi=dpi, bbox_inches="tight")

    p_end = datetime.now()
    plot_hint_print("plot_bistability_map", quiet, "end", time=(p_start, p_end))
