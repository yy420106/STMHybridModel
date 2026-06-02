import math
import os
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style, init
from plot_funcs1 import verbose_print, format_save
from post_process_funcs import get_samples_gene_activity, get_samples_mek_ratio

__all__ = ["plot_chromMe", "plot_protQuant", "plot_geneAct", "plot_evolution_curves"]

# global variables & settings
init(autoreset=True)  # colorama mode: autoreset

# plot settings
# --------------------------------------------------------------------------------
ky = 0.05  # extension ratio of y-axis in both side
kxl = 0.005  # extension ratio of x-axis in left side
kxr = 0.045  # extension ratio of x-axis in right side

lw = 0.5  # line width for single trial curve
m_lw = 1.5  # line width for mean curve
alp_l = 0.5  # transparency of plot line
alp_bg = 0.2  # transparency of background

tk_w = 1.5  # width of tick
tk_l = 4  # length of tick

tk_ls = 12  # label size of tick-label
x_ls = 16  # label size of x-label
y_ls = 16  # label size of y-label
at_fs = 14  # font size of annotation in both title and side

xl_y = -0.2  # relative position of x-label of subplot in y-direction
tal_x = -0.03  # relative position of x-label of title annotation in x-direction
tal_y = 0.1  # relative position of x-label of title annotation in y-direction

sp_ax_l = 0.08  # subplot axis - left
sp_ax_r = 0.96  # subplot axis - right
sp_ax_b = 0.06  # subplot axis - bottom
sp_ax_t = 0.94  # subplot axis - top
yl_ax_l = 0.02  # y-labl axis - left
yl_ax_r = 0.06  # y-label axis - right
ta_ax_b = 0.95  # title annotation axis - bottom
ta_ax_t = 0.98  # title annotation axis - top

arr_w = 0.01  # arrow width
arr_x = 0.02  # realtive length of arrow in x-direction
arr_y = 0.12  # realtive length of arrow in y-direction

colorDict = {
    "me0": {"color": "darkorange", "m_color": "sienna"},
    "me1": {"color": "red", "m_color": "maroon"},
    "me2": {"color": "dodgerblue", "m_color": "midnightblue"},
    "me3": {"color": "mediumorchid", "m_color": "indigo"},
    "prot": {"color": "limegreen", "m_color": "darkgreen"},
    "gene": {"color": "dimgray"},
    "side": "lightgray",
    "title": {0: "orange", 1: "palegreen", 2: "violet", 3: "skyblue", 4: "salmon"},
    "arrow": {0: "red", 1: "blue", 2: "black"},
}  # colors in plot
# --------------------------------------------------------------------------------


def plot_chromMe(
    ax: plt.Axes,
    time_records: np.ndarray[Any, float],
    mek_ratio: np.ndarray[Any, float],
    time_unit_day: bool,
    color: str,
    annot: str,
    m_plot: bool = False,
    m_color: Optional[str] = None,
    bg_fill: bool = False,
    title_annot_timecut: Optional[Sequence[float]] = None,
    xticklabels: bool = True,
    xlabel: Optional[str] = None,
) -> None:
    """Module function of `plot_evolution_curves`, used for methylation modification section."""

    s_start, s_end = time_records[0], time_records[-1]

    if bg_fill:
        assert title_annot_timecut is not None
        for idx in range(len(title_annot_timecut) - 1):
            ax.fill_between(
                x=[title_annot_timecut[idx], title_annot_timecut[idx + 1]],
                y1=0,
                y2=1,
                facecolor=colorDict["title"].get(idx % len(colorDict["title"])),
                alpha=alp_bg,
            )
    ax.plot(time_records, mek_ratio.T, linewidth=lw, color=color, alpha=alp_l)
    if m_plot:
        assert m_color is not None
        ax.plot(time_records, mek_ratio.mean(axis=0), linewidth=m_lw, color=m_color)
    ax.set_ylim([0 - ky, 1 + ky])
    ax.set_xlim([s_start - kxl * (s_end - s_start), s_end + kxr * (s_end - s_start)])

    ax.set_yticks(ticks=[0.0, 0.5, 1.0])
    if time_unit_day:
        ax.set_xticks(ticks=5 * np.arange(math.ceil(s_start / 5), math.floor(s_end / 5) + 1))
    if not xticklabels:
        ax.set_xticklabels([])
    ax.tick_params(width=tk_w, length=tk_l, labelsize=tk_ls)
    if xlabel is not None:
        ax.set_xlabel(xlabel=xlabel, fontsize=x_ls, weight="bold")
        ax.xaxis.set_label_coords(x=(0.5 + kxl) / (1 + kxl + kxr), y=xl_y, transform=ax.transAxes)
    ax.fill_between(
        x=[s_end + kxl * (s_end - s_start), s_end + kxr * (s_end - s_start)],
        y1=0 - ky,
        y2=1 + ky,
        facecolor=colorDict["side"],
        alpha=alp_l,
    )
    ax.text(
        x=(1 + 2 * kxl + 0.5 * (kxr - kxl)) / (1 + kxl + kxr),
        y=0.5,
        s=annot,
        color="black",
        fontsize=at_fs,
        weight="bold",
        rotation=-90,
        verticalalignment="center",
        horizontalalignment="center",
        transform=ax.transAxes,
    )


def plot_protQuant(
    ax: plt.Axes,
    time_records: np.ndarray[Any, float],
    protQuant: np.ndarray[Any, float],
    time_unit_day: bool,
    color: str,
    annot: str,
    m_plot: bool = False,
    m_color: Optional[str] = None,
    bg_fill: bool = False,
    title_annot_timecut: Optional[Sequence[float]] = None,
    arrow_annot_pos: Optional[Sequence[float]] = None,
    xticklabels: bool = True,
    xlabel: Optional[str] = None,
    y_min_zero: bool = True,
) -> None:
    """Module function of `plot_evolution_curves`, used for protein section."""

    s_start, s_end = time_records[0], time_records[-1]
    y_min, y_max = 0.0 if y_min_zero else protQuant.min(), protQuant.max()

    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    else:
        sbar = 10 ** max(1, min(2, np.floor(np.log10(y_max))))  # scale bar: 10, 100
        y_min, y_max = np.floor(y_min / sbar) * sbar, np.ceil(y_max / sbar) * sbar

    if bg_fill:
        assert title_annot_timecut is not None
        for idx in range(len(title_annot_timecut) - 1):
            ax.fill_between(
                x=[title_annot_timecut[idx], title_annot_timecut[idx + 1]],
                y1=y_min,
                y2=y_max,
                facecolor=colorDict["title"].get(idx % len(colorDict["title"])),
                alpha=alp_bg,
            )
    ax.plot(time_records, protQuant.T, linewidth=lw, color=color, alpha=alp_l)
    if m_plot:
        assert m_color is not None
        ax.plot(time_records, protQuant.mean(axis=0), linewidth=m_lw, color=m_color)
    ax.set_ylim([y_min - ky * (y_max - y_min), y_max + ky * (y_max - y_min)])
    ax.set_xlim([s_start - kxl * (s_end - s_start), s_end + kxr * (s_end - s_start)])
    if arrow_annot_pos is not None:
        for idx in range(len(arrow_annot_pos)):
            ax.annotate(
                text="",
                xy=(s_end, arrow_annot_pos[idx]),
                xytext=(
                    s_end - arr_x * (s_end - s_start),
                    arrow_annot_pos[idx] + arr_y * (y_max - y_min),
                ),
                arrowprops={
                    "width": arr_w,
                    "color": colorDict["arrow"].get(idx % len(arrow_annot_pos)),
                },
            )

    ax.set_yticks(ticks=np.linspace(y_min, y_max, 3))
    if time_unit_day:
        ax.set_xticks(ticks=5 * np.arange(math.ceil(s_start / 5), math.floor(s_end / 5) + 1))
    if not xticklabels:
        ax.set_xticklabels([])
    ax.tick_params(width=tk_w, length=tk_l, labelsize=tk_ls)
    if xlabel is not None:
        ax.set_xlabel(xlabel=xlabel, fontsize=x_ls, weight="bold")
        ax.xaxis.set_label_coords(x=(0.5 + kxl) / (1 + kxl + kxr), y=xl_y, transform=ax.transAxes)
    ax.fill_between(
        x=[s_end + kxl * (s_end - s_start), s_end + kxr * (s_end - s_start)],
        y1=y_min - ky * (y_max - y_min),
        y2=y_max + ky * (y_max - y_min),
        facecolor=colorDict["side"],
        alpha=alp_l,
    )
    ax.text(
        x=(1 + 2 * kxl + 0.5 * (kxr - kxl)) / (1 + kxl + kxr),
        y=0.5,
        s=annot,
        color="black",
        fontsize=at_fs,
        weight="bold",
        rotation=-90,
        verticalalignment="center",
        horizontalalignment="center",
        transform=ax.transAxes,
    )


def plot_geneAct(
    ax: plt.Axes,
    time_records: np.ndarray[Any, float],
    geneActivity: np.ndarray[Any, float],
    bin_mid: np.ndarray[Any, float],
    bin_width: float,
    time_unit_day: bool,
    color: str,
    annot: str,
    bg_fill: bool = False,
    title_annot_timecut: Optional[Sequence[float]] = None,
    xticklabels: bool = True,
    xlabel: Optional[str] = None,
) -> None:
    """Module function of `plot_evolution_curves`, used for gene activity section."""

    s_start, s_end = time_records[0], time_records[-1]
    y_min, y_max = 0.0, np.ceil(geneActivity.max())
    if y_max == 0.0:
        y_min, y_max = -0.5, 0.5

    if bg_fill:
        assert title_annot_timecut is not None
        for idx in range(len(title_annot_timecut) - 1):
            ax.fill_between(
                x=[title_annot_timecut[idx], title_annot_timecut[idx + 1]],
                y1=y_min,
                y2=y_max,
                facecolor=colorDict["title"].get(idx % len(colorDict["title"])),
                alpha=alp_bg,
            )
    ax.bar(x=bin_mid, height=geneActivity, width=bin_width, color=color)
    ax.hlines(y=0.0, xmin=s_start, xmax=s_end, linewidth=m_lw, color=color)
    ax.set_ylim([y_min - ky * (y_max - y_min), y_max + ky * (y_max - y_min)])
    ax.set_xlim([s_start - kxl * (s_end - s_start), s_end + kxr * (s_end - s_start)])

    ax.set_yticks(ticks=np.linspace(y_min, y_max, 3))
    if time_unit_day:
        ax.set_xticks(ticks=5 * np.arange(math.ceil(s_start / 5), math.floor(s_end / 5) + 1))
    if not xticklabels:
        ax.set_xticklabels([])
    ax.tick_params(width=tk_w, length=tk_l, labelsize=tk_ls)
    if xlabel is not None:
        ax.set_xlabel(xlabel=xlabel, fontsize=x_ls, weight="bold")
        ax.xaxis.set_label_coords(x=(0.5 + kxl) / (1 + kxl + kxr), y=xl_y, transform=ax.transAxes)
    ax.fill_between(
        x=[s_end + kxl * (s_end - s_start), s_end + kxr * (s_end - s_start)],
        y1=y_min - ky * (y_max - y_min),
        y2=y_max + ky * (y_max - y_min),
        facecolor=colorDict["side"],
        alpha=alp_l,
    )
    ax.text(
        x=(1 + 2 * kxl + 0.5 * (kxr - kxl)) / (1 + kxl + kxr),
        y=0.5,
        s=annot,
        color="black",
        fontsize=at_fs,
        weight="bold",
        rotation=-90,
        verticalalignment="center",
        horizontalalignment="center",
        transform=ax.transAxes,
    )


def plot_evolution_curves(
    time_records: np.ndarray[Any, float],
    samples_protQuant: np.ndarray[Any, float],
    samples_meState: np.ndarray[Any, int],
    samples_transcrT: np.ndarray[Any, float],
    transcr_stat_bins: Optional[int] = None,
    custom_mod_show: Optional[Dict[str, bool]] = None,
    time_unit_day: bool = True,
    m_plot: bool = False,
    bg_fill: bool = False,
    title_annot_timecut: Optional[Sequence[float]] = None,
    title_annot_label: Optional[Sequence[str]] = None,
    arrow_annot_pos: Optional[Sequence[float]] = None,
    protQuant_min_zero: bool = True,
    figName: Optional[str] = None,
    figSize: Optional[Sequence[float]] = None,
    figFormat: Union[str, Sequence[str], None] = None,
    dpi: Optional[int] = None,
    save_dir: Optional[str] = None,
    quiet: bool = False,
) -> None:
    """
    Base function for plotting model evolution curves against time.

    Parameters
    ----------
    time_records : NDArray[float64], shape (T,)
        Array of time points (unit: hour) at which to monitor the model state.
    samples_protQuant : NDArray[float64], shape (N, T)
        Records of gene expression quantity during evolution.
    samples_meState : NDArray[int32], shape (N, T, H)
        Records of the methylation state of histone during evolution.
    samples_transcrT : NDArray[float64], shape (N, #)
        Array that store the time of every transcription event in each trial, empty spaces are
        filled with NaN. '#' indicates the length is nontrivial.
    transcr_stat_bins : int32 (optional)
        Number of bins in distribution histogram.
    custom_mod_show : Dict[str, bool] (optional)
        Customized strategy for module displacement. By default all elements (me0/1/2/3, prot,
        gene) are shown. You can provide a bool flag to enable/disable corresponding section.
        e.g. `custom_mod_show = dict(me0=False, prot=Flase)`
    time_unit_day : bool (default=True)
        Whether to convert time unit to Day instead of Hour.
    m_plot : bool (default=False)
        Whether to plot mean curve in methylation & protein section.
    bg_fill : bool (default=False)
        Whether to fill background.
    title_annot_timecut : Sequence[float64] (optional)
        Time-cut (boundary) for title annotation range.
    title_annot_label : Sequence[str] (optional)
        Label of title annotation, used together with parameter `title_annot_timecut`.
    arrow_annot_pos : Sequence[float] (optional)
        Position of arrow symbol at protein subplot.
    protQuant_min_zero : bool (default=True)
        Whether to set min value of y-axis in protein module to 0.
    """

    p_start = datetime.now()  # plot start

    # pre-process
    # module set
    mod_show = {"me0": True, "me1": True, "me2": True, "me3": True, "prot": True, "gene": True}
    if custom_mod_show is not None:
        assert set(custom_mod_show).issubset(set(mod_show)), "Invalid module names."
        mod_show.update(custom_mod_show)
    num_mod1 = mod_show["me0"] + mod_show["me1"] + mod_show["me2"] + mod_show["me3"]
    num_mod2 = mod_show["prot"] + mod_show["gene"]

    # transform time unit
    time_unit = "Hour"
    if time_unit_day:
        time_unit = "Day"
        time_records, samples_transcrT = time_records / 24.0, samples_transcrT / 24.0
        if title_annot_timecut is not None:
            title_annot_timecut = np.array(title_annot_timecut) / 24.0

    s_start, s_end = time_records[0], time_records[-1]
    ev_time = s_end - s_start

    # default set
    figName = "evolution_curves" if figName is None else figName
    figSize = (16, 9 * (num_mod1 + num_mod2) / 4) if figSize is None else tuple(figSize)
    figFormat = ["pdf"] if figFormat is None else figFormat
    dpi = 800 if dpi is None else dpi
    save_dir = os.getcwd() if save_dir is None else save_dir

    verbose_print(
        message=f"\t{Fore.GREEN + Style.BRIGHT}<plot_evolution_curves>{Style.RESET_ALL} Plotting ...",
        quiet=quiet,
    )

    # plot figure
    plt.rc("font", family="Arial")
    fig = plt.figure(figName, figSize)
    mod_idx = 0  # flag number

    # plot methylation modification level
    for j in range(4):
        if mod_show.get(f"me{j}"):
            plot_chromMe(
                ax=fig.add_subplot(num_mod1 + num_mod2, 1, mod_idx + 1),
                time_records=time_records,
                mek_ratio=get_samples_mek_ratio(samples_meState=samples_meState, k=j),
                time_unit_day=time_unit_day,
                color=colorDict[f"me{j}"].get("color"),
                annot=f"me{j}",
                m_plot=m_plot,
                m_color=colorDict[f"me{j}"].get("m_color") if m_plot else None,
                bg_fill=bg_fill,
                title_annot_timecut=title_annot_timecut,
                xticklabels=True if mod_idx + 1 == num_mod1 + num_mod2 else False,
                xlabel=f"Time ({time_unit})" if mod_idx + 1 == num_mod1 + num_mod2 else None,
            )
            mod_idx += 1

    # plot protein dynamics
    if mod_show.get("prot"):
        plot_protQuant(
            ax=fig.add_subplot(num_mod1 + num_mod2, 1, mod_idx + 1),
            time_records=time_records,
            protQuant=samples_protQuant,
            time_unit_day=time_unit_day,
            color=colorDict["prot"].get("color"),
            annot="protein",
            m_plot=m_plot,
            m_color=colorDict["prot"].get("m_color") if m_plot else None,
            bg_fill=bg_fill,
            title_annot_timecut=title_annot_timecut,
            arrow_annot_pos=arrow_annot_pos,
            xticklabels=True if mod_idx + 1 == num_mod1 + num_mod2 else False,
            xlabel=f"Time ({time_unit})" if mod_idx + 1 == num_mod1 + num_mod2 else None,
            y_min_zero=protQuant_min_zero,
        )
        mod_idx += 1

    # draw histogram of transcription
    if mod_show.get("gene"):
        assert transcr_stat_bins is not None
        # compute gene activity
        samples_geneActivity, bin_mid = get_samples_gene_activity(
            samples_transcrT=samples_transcrT, bins=transcr_stat_bins, range=(s_start, s_end)
        )

        plot_geneAct(
            ax=fig.add_subplot(num_mod1 + num_mod2, 1, mod_idx + 1),
            time_records=time_records,
            geneActivity=samples_geneActivity,
            bin_mid=bin_mid,
            bin_width=ev_time / transcr_stat_bins,
            time_unit_day=time_unit_day,
            color=colorDict["gene"].get("color"),
            annot="gene\nactivity",
            bg_fill=bg_fill,
            title_annot_timecut=title_annot_timecut,
            xticklabels=True if mod_idx + 1 == num_mod1 + num_mod2 else False,
            xlabel=f"Time ({time_unit})" if mod_idx + 1 == num_mod1 + num_mod2 else None,
        )
        mod_idx += 1

    # check
    assert mod_idx == num_mod1 + num_mod2

    # self-adaption space
    hsp = (
        (ta_ax_b - sp_ax_t)
        * (num_mod1 + num_mod2)
        / ((sp_ax_t - sp_ax_b) - (ta_ax_b - sp_ax_t) * (num_mod1 + num_mod2 - 1))
    )
    fig.subplots_adjust(left=sp_ax_l, right=sp_ax_r, bottom=sp_ax_b, top=sp_ax_t, hspace=hsp)

    # add combined ylabel
    yl_ax: plt.Axes = fig.add_axes([yl_ax_l, sp_ax_b, yl_ax_r - yl_ax_l, sp_ax_t - sp_ax_b])
    if num_mod1:
        yl_ax.text(
            x=0.5,
            y=((num_mod2 + num_mod1 / 2) + (num_mod2 + (num_mod1 - 1) / 2) * hsp)
            / ((num_mod1 + num_mod2) + (num_mod1 + num_mod2 - 1) * hsp),
            s="Histone Modification Level",
            fontsize=y_ls,
            weight="bold",
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            transform=yl_ax.transAxes,
        )
    if num_mod2:
        yl_ax.text(
            x=0.5,
            y=(num_mod2 / 2 + (num_mod2 - 1) / 2 * hsp)
            / ((num_mod1 + num_mod2) + (num_mod1 + num_mod2 - 1) * hsp),
            s="Gene Expression",
            fontsize=y_ls,
            weight="bold",
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            transform=yl_ax.transAxes,
        )
    yl_ax.axis("off")

    # add title annotation
    if title_annot_timecut is not None:
        assert title_annot_label is not None
        assert title_annot_timecut[0] == s_start and title_annot_timecut[-1] == s_end
        ta_ax: plt.Axes = fig.add_axes([sp_ax_l, ta_ax_b, sp_ax_r - sp_ax_l, ta_ax_t - ta_ax_b])
        ta_ax.set_xlim([s_start - kxl * ev_time, s_end + kxr * ev_time])
        ta_ax.set_ylim([0, 1])
        ta_ax.set_xticks([])
        ta_ax.set_yticks([])
        ta_ax.set_ylabel("Division", rotation=0, fontsize=at_fs, weight="bold")
        ta_ax.yaxis.set_label_coords(x=tal_x, y=tal_y, transform=ta_ax.transAxes)
        for idx in range(len(title_annot_timecut) - 1):
            ta_ax.fill_between(
                x=[title_annot_timecut[idx], title_annot_timecut[idx + 1]],
                y1=0,
                y2=1,
                facecolor=colorDict["title"].get(idx % len(colorDict["title"])),
                alpha=alp_bg,
            )
            ta_ax.text(
                x=(
                    (title_annot_timecut[idx + 1] - title_annot_timecut[idx]) / 2
                    + title_annot_timecut[idx]
                    - s_start
                    + kxl * ev_time
                )
                / ((1 + kxl + kxr) * ev_time),
                y=0.5,
                s=title_annot_label[idx],
                fontsize=at_fs,
                weight="bold",
                verticalalignment="center",
                horizontalalignment="center",
                transform=ta_ax.transAxes,
            )
        ta_ax.hlines(y=0.05, xmin=s_start, xmax=s_end, color="black", linewidth=m_lw)
        for side in ["left", "right", "bottom", "top"]:
            ta_ax.spines[side].set_linewidth(0)

    verbose_print(
        message=f"\t{Fore.GREEN + Style.BRIGHT}<plot_evolution_curves>{Style.RESET_ALL} Saving {'/'.join(figFormat)} ...",
        quiet=quiet,
    )

    # save figure
    format_save(fig, figName, figFormat, save_dir, dpi=dpi, bbox_inches="tight")

    p_end = datetime.now()  # plot end

    verbose_print(
        message=(
            f"\t{Fore.GREEN + Style.BRIGHT}<plot_evolution_curves>{Style.RESET_ALL} Done. "
            f"(Finished in {Fore.RED + Style.BRIGHT + str(p_end - p_start) + Style.RESET_ALL})"
        ),
        quiet=quiet,
    )
