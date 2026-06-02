#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import platform
import sys
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numba as nb
from colorama import Fore, Style, init
from commands import main_bimap, main_divarrest, main_dynmcyc, main_epistb, main_rescue, main_schmdg

# global variables & settings
os.chdir(path=sys.path[0])
nb.config.NUMBA_DEFAULT_NUM_THREADS = os.cpu_count()
init(autoreset=True)  # colorama mode: autoreset


def make_save_dir(cmd: str, temp: bool, cwd: Optional[str] = None) -> str:
    """
    Auxilary function. Make directory to save all possible output files.

    Parameters
    ----------
    cmd: str
        Callable command name.
    temp: bool
        If True, export the output files to a temp folder where they can overwrite the old oversion.
        Otherwise, all outputs will be stored in a separate folder (named by timestamp of the build).
    cwd: str (optional)
        Path to the current working directory.

    Returns
    -------
    save_dir: str
        Path to the output directory.
    """

    cwd = os.getcwd() if cwd is None else cwd
    output_dir = os.path.join(cwd, "outputs")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    save_dir = (
        os.path.join(output_dir, cmd, "temp")
        if temp
        else os.path.join(output_dir, cmd, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"The output files will be stored in directory: {Fore.BLUE + Style.BRIGHT + save_dir}")

    return save_dir


def write_command_log(log_dir: str) -> None:
    """
    Auxilary function. Write command log file.

    Parameters
    ----------
    log_dir: str
        Directory in which to write the log file.
    """

    with open(os.path.join(log_dir, "command.log"), "a") as log:
        if log_dir.endswith("temp"):
            log.write(
                f"# command ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
            )  # add a timestamp
        else:
            log.write("# command\n")  # omit timestamp this case
        log.write(f"# platform: {platform.platform()}\n")
        log.write(f"> {' '.join([sys.executable] + sys.argv)}\n\n")


def create_shared_parser() -> argparse.ArgumentParser:
    """Auxilary function. Create a shared CLI parameters parser."""

    parser = argparse.ArgumentParser(add_help=False)

    # options
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="run in quiet mode without showing any detailed information",
    )
    parser.add_argument(
        "-p", "--prefix", default=None, type=str, metavar="PREFIX", help="figure name prefix"
    )
    parser.add_argument(
        "-z", "--size", nargs=2, default=None, type=float, metavar=("W", "H"), help="figure size"
    )
    parser.add_argument(
        "-c",
        "--concise",
        action="store_true",
        help="concise mode with additional analysis disabled",
    )
    parser.add_argument(
        "-p2",
        "--prefix2",
        default=None,
        type=str,
        metavar="PREFIX2",
        help="additional figure name prefix",
    )
    parser.add_argument(
        "-z2",
        "--size2",
        nargs=2,
        default=None,
        type=float,
        metavar=("W2", "H2"),
        help="additional figure size",
    )
    parser.add_argument(
        "-f", "--format", nargs="+", default=None, type=str, metavar="FORMAT", help="image format"
    )
    parser.add_argument(
        "-r", "--dpi", default=None, type=int, metavar="DPI", help="image resolution"
    )
    parser.add_argument(
        "-t", "--temp", action="store_true", help="export files to command temp folder"
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="display all plotted figures before the program exits",
    )

    # global parameters
    parser_param = parser.add_argument_group(title="global parameters")
    parser_param.add_argument(
        "-ui",
        "--update_interval",
        default=1 / 60,
        type=float,
        metavar="UI",
        help="model update interval (unit: hour)",
    )
    parser_param.add_argument(
        "-hi",
        "--hist_interval",
        default=0.25,
        type=float,
        metavar="HI",
        help="histogram step interval (unit: hour)",
    )
    parser_param.add_argument(
        "-fr",
        "--fluct_ratio",
        default=0.5,
        type=float,
        metavar="FR",
        help="initial fluctuation ratio",
    )

    return parser


def add_subparser_schmdg(subparser: argparse._SubParsersAction) -> None:
    """Auxilary function. Add argument parser for subcommand `schmdg`."""

    subparser.add_parser(
        name="schmdg",
        help="plot schematic diagram of model mathematical principles",
        description="plot schematic diagram of model mathematical principles",
        parents=[create_shared_parser()],
    )


def add_subparser_epistb(subparser: argparse._SubParsersAction) -> None:
    """Auxilary function. Add argument parser for subcommand `epistb`."""

    subparser_epistb: argparse.ArgumentParser = subparser.add_parser(
        name="epistb",
        help="""
            plot the profiles of methylation level and STM expression change over multiple days in both stem 
            cells and differentiated cells to show epigenetic stability
        """,
        description="""
            plot the profiles of methylation level and STM expression change over multiple days in both stem 
            cells and differentiated cells to show epigenetic stability
        """,
        parents=[create_shared_parser()],
    )

    # command parameters
    subparser_epistb_param = subparser_epistb.add_argument_group(title="command parameters")
    subparser_epistb_param.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        metavar="NS",
        help="number of model samples",
        required=True,
    )
    subparser_epistb_param.add_argument(
        "-et",
        "--epi_tag",
        nargs="+",
        choices=[0, 1, 2, 3],
        type=int,
        metavar="ET",
        help="initial methylation state",
        required=True,
    )
    subparser_epistb_param.add_argument(
        "-md",
        "--monitor_days",
        type=float,
        metavar="MD",
        help="days of monitoring in simulation",
        required=True,
    )

    subparser_epistb_param.add_argument(
        "-ts",
        "--timeid_step",
        default=None,
        type=int,
        metavar="TS",
        help="step of time index in evolution heatmap plot (additional plot parameter)",
    )


def add_subparser_dynmcyc(subparser: argparse._SubParsersAction) -> None:
    """Auxilary function. Add argument parser for subcommand `dynmcyc`."""

    subparser_dynmcyc: argparse.ArgumentParser = subparser.add_parser(
        name="dynmcyc",
        help="plot the influence of dynamic (increasing/decreasing) cell cycle on stem cells",
        description="plot the influence of dynamic (increasing/decreasing) cell cycle on stem cells",
        parents=[create_shared_parser()],
    )

    # command parameters
    subparser_dynmcyc_param = subparser_dynmcyc.add_argument_group(title="command parameters")
    subparser_dynmcyc_param.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        metavar="NS",
        help="number of model samples",
        required=True,
    )
    subparser_dynmcyc_param.add_argument(
        "-df",
        "--dynamic_factor",
        nargs="+",
        type=float,
        metavar="DF",
        help="dynamic factor",
        required=True,
    )
    subparser_dynmcyc_param.add_argument(
        "-ec",
        "--equi_cycles",
        type=int,
        metavar="EC",
        help="cell cycle numbers in pre-equilibrium",
        required=True,
    )
    subparser_dynmcyc_param.add_argument(
        "-md",
        "--monitor_days",
        type=float,
        metavar="MD",
        help="days of monitoring in simulation",
        required=True,
    )


def add_subparser_divarrest(subparser: argparse._SubParsersAction) -> None:
    """Auxilary function. Add argument parser for subcommand `divarrest`."""

    subparser_divarrest: argparse.ArgumentParser = subparser.add_parser(
        name="divarrest",
        help="""
            plot transition curves of stem cell differentiation during different length of division arrest 
            & cell type distribution statistics after division recovery
        """,
        description="""
            plot transition curves of stem cell differentiation during different length of division arrest 
            & cell type distribution statistics after division recovery
        """,
        parents=[create_shared_parser()],
    )

    # command parameters
    subparser_divarrest_param = subparser_divarrest.add_argument_group(title="command parameters")
    subparser_divarrest_param.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        metavar="NS",
        help="number of model samples",
        required=True,
    )
    subparser_divarrest_param.add_argument(
        "-ec",
        "--equi_cycles",
        type=int,
        metavar="EC",
        help="cell cycle numbers in pre-equilibrium",
        required=True,
    )
    subparser_divarrest_param.add_argument(
        "-md",
        "--monitor_days",
        type=float,
        metavar="MD",
        help="days of monitoring in simulation",
        required=True,
    )
    subparser_divarrest_param.add_argument(
        "-ad",
        "--arrest_days",
        nargs="+",
        type=float,
        metavar="AD",
        help="days of cell division arrest",
        required=True,
    )
    subparser_divarrest_param.add_argument(
        "-ps",
        "--plot_step",
        default=1,
        type=int,
        metavar="PS",
        help="index step of sample that to be show in plot",
    )

    subparser_divarrest_param.add_argument(
        "-cd",
        "--count_days",
        nargs="+",
        default=None,
        type=float,
        metavar="CD",
        help="days to count cell type distribution after division restart (addditional plot parameter)",
    )


def add_subparser_rescue(subparser: argparse._SubParsersAction) -> None:
    """Auxilary function. Add argument parser for subcommand `rescue`."""

    subparser_rescue: argparse.ArgumentParser = subparser.add_parser(
        name="rescue",
        help="plot recovery curves of differentiated cells after divison arrest using different rescue stategies",
        description="plot recovery curves of differentiated cells after divison arrest using different rescue stategies",
        parents=[create_shared_parser()],
    )

    # command parameters
    subparser_rescue_param = subparser_rescue.add_argument_group(title="command parameters")
    subparser_rescue_param.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        metavar="NS",
        help="number of model samples",
        required=True,
    )
    subparser_rescue_param.add_argument(
        "-rs",
        "--rescue_strategy",
        nargs="+",
        choices=["M", "m", "A", "a", "S", "s"],
        type=str,
        metavar="RS",
        help="rescue strategy: M/m - remove methylation, A/a - add ATH1, S/s - add STM",
        required=True,
    )
    subparser_rescue_param.add_argument(
        "-ad",
        "--arrest_days",
        type=float,
        metavar="AD",
        help="days of cell division arrest",
        required=True,
    )
    subparser_rescue_param.add_argument(
        "-td",
        "--treat_days",
        type=float,
        metavar="TD",
        help="days of rescue treatment",
        required=True,
    )
    subparser_rescue_param.add_argument(
        "-rd",
        "--rest_days",
        default=0.0,
        type=float,
        metavar="RD",
        help="days of resting state before rescue",
    )
    subparser_rescue_param.add_argument(
        "-pr",
        "--prob_removal",
        default=0.85,
        type=float,
        metavar="PR",
        help="probability of methylation removal",
    )
    subparser_rescue_param.add_argument(
        "-ag",
        "--affn_growth",
        default=0.05,
        type=float,
        metavar="AG",
        help="growth ratio of transcrition factor affinity to gene locus",
    )
    subparser_rescue_param.add_argument(
        "-pe",
        "--prot_exogain",
        default=1.0,
        type=float,
        metavar="PE",
        help="exogenous gain of protein relative to stable level",
    )


def add_subparser_bimap(subparser: argparse._SubParsersAction) -> None:
    """Auxilary function. Add argument parser for subcommand `bimap`."""

    subparser_bimap: argparse.ArgumentParser = subparser.add_parser(
        name="bimap",
        help="""plot bistability heatmap in 2-D parameter k_me - Pdem space under different cell cycle conditions""",
        description="plot bistability heatmap in 2-D parameter k_me - Pdem space under different cell cycle conditions",
        parents=[create_shared_parser()],
    )

    # command parameters
    subparser_bimap_param = subparser_bimap.add_argument_group(title="command parameters")
    subparser_bimap_param.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        metavar="NS",
        help="number of model samples",
        required=True,
    )
    subparser_bimap_param.add_argument(
        "-cc", "--cell_cycle", nargs="+", type=float, metavar="CC", help="cell cycle", required=True
    )
    subparser_bimap_param.add_argument(
        "-nc",
        "--num_cycles",
        type=int,
        metavar="NC",
        help="number of cell cycles in simulation",
        required=True,
    )
    subparser_bimap_param.add_argument(
        "-mp", "--map_pixel", type=int, metavar="MP", help="pixels in both dimension", required=True
    )
    subparser_bimap_param.add_argument(
        "-kl",
        "--kme_logrange",
        nargs=2,
        default=[-6, -2],
        type=float,
        metavar=("KL1", "KL2"),
        help="log10 range of paramter k_me to be sampled",
    )
    subparser_bimap_param.add_argument(
        "-pl",
        "--pdem_logrange",
        nargs=2,
        default=[-4, 0],
        type=float,
        metavar=("PL1", "PL2"),
        help="log10 range of paramter Pdem to be sampled",
    )


def main():
    """Entrance of main program."""

    # create major parser
    parser = argparse.ArgumentParser(
        description="Control the details of model simulation and visulization, as well as figure plotting and saving.",
    )

    # subcommand parser
    subparser = parser.add_subparsers(dest="command", title="commands")

    # add subcommand
    add_subparser_schmdg(subparser)
    add_subparser_epistb(subparser)
    add_subparser_dynmcyc(subparser)
    add_subparser_divarrest(subparser)
    add_subparser_rescue(subparser)
    add_subparser_bimap(subparser)

    # parse arguments
    args = parser.parse_args()

    if args.command is not None:
        # make output directory
        save_dir = make_save_dir(cmd=args.command, temp=args.temp)
        # write log file
        write_command_log(log_dir=save_dir)

        if args.command == "schmdg":
            main_schmdg(
                quiet=args.quiet,
                prefix=args.prefix,
                size=args.size,
                format=args.format,
                dpi=args.dpi,
                save_dir=save_dir,
            )

        elif args.command == "epistb":
            # check input requirements
            if not args.concise:
                assert args.timeid_step is not None, (
                    "in non-concise mode the following arguments are required: -ts/--timeid_step"
                )

            main_epistb(
                num_samples=args.num_samples,
                epi_tag=args.epi_tag,
                monitor_days=args.monitor_days,
                update_interval=args.update_interval,
                concise=args.concise,
                timeid_step=args.timeid_step,
                quiet=args.quiet,
                prefix=args.prefix,
                size=args.size,
                prefix2=args.prefix2,
                size2=args.size2,
                format=args.format,
                dpi=args.dpi,
                save_dir=save_dir,
            )

        elif args.command == "dynmcyc":
            main_dynmcyc(
                num_samples=args.num_samples,
                dynamic_factor=args.dynamic_factor,
                equi_cycles=args.equi_cycles,
                monitor_days=args.monitor_days,
                update_interval=args.update_interval,
                hist_interval=args.hist_interval,
                fluc_ratio=args.fluct_ratio,
                quiet=args.quiet,
                prefix=args.prefix,
                size=args.size,
                format=args.format,
                dpi=args.dpi,
                save_dir=save_dir,
            )

        elif args.command == "divarrest":
            # check input requirement
            if not args.concise:
                assert args.count_days is not None, (
                    "in non-concise mode the following arguments are required: -cd/--count_days"
                )

            main_divarrest(
                num_samples=args.num_samples,
                equi_cycles=args.equi_cycles,
                monitor_days=args.monitor_days,
                arrest_days=args.arrest_days,
                plot_step=args.plot_step,
                update_interval=args.update_interval,
                hist_interval=args.hist_interval,
                fluc_ratio=args.fluct_ratio,
                concise=args.concise,
                count_days=args.count_days,
                quiet=args.quiet,
                prefix=args.prefix,
                size=args.size,
                prefix2=args.prefix2,
                size2=args.size2,
                format=args.format,
                dpi=args.dpi,
                save_dir=save_dir,
            )

        elif args.command == "rescue":
            # process input
            args.rescue_strategy = list(set(s.upper() for s in args.rescue_strategy))

            main_rescue(
                num_samples=args.num_samples,
                rescue_strategy=args.rescue_strategy,
                arrest_days=args.arrest_days,
                treat_days=args.treat_days,
                rest_days=args.rest_days,
                prob_removal=args.prob_removal,
                affn_growth=args.affn_growth,
                prot_exogain=args.prot_exogain,
                update_interval=args.update_interval,
                hist_interval=args.hist_interval,
                fluc_ratio=args.fluct_ratio,
                quiet=args.quiet,
                prefix=args.prefix,
                size=args.size,
                format=args.format,
                dpi=args.dpi,
                save_dir=save_dir,
            )

        elif args.command == "bimap":
            # process input
            args.kme_logrange.sort()
            args.pdem_logrange.sort()

            main_bimap(
                num_samples=args.num_samples,
                T=args.cell_cycle,
                num_cycles=args.num_cycles,
                map_pixel=args.map_pixel,
                kme_logrange=args.kme_logrange,
                pdem_logrange=args.pdem_logrange,
                update_interval=args.update_interval,
                quiet=args.quiet,
                prefix=args.prefix,
                size=args.size,
                format=args.format,
                dpi=args.dpi,
                save_dir=save_dir,
            )

        else:
            print("Unrecognizable subcommand encountered.")
            parser.print_help()

        if args.display:
            plt.show()  # show figure

    else:
        parser.print_usage()


if __name__ == "__main__":
    main()
