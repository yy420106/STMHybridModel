from typing import Any

import numba as nb
import numba_progress as nbp
import numpy as np
from util_funcs import calc_beta, calc_E, calc_P_me23, calc_theta

__all__ = [  # noqa: RUF022
    "gillespie_get_propensities",
    "gillespie_draw",
    "gillespie_ssa",
    "gillespie_ssa_parallel",
]  # JIT-complie


@nb.njit(
    signature_or_function=nb.types.Tuple(
        types=(nb.float64[:], nb.float64[:], nb.float64, nb.float64)
    )(nb.float64, nb.int32[:], nb.types.UniTuple(dtype=nb.float64, count=27)),
    cache=True,
)
def gillespie_get_propensities(
    protQuant: float, meState: np.ndarray[Any, int], all_param: tuple[float, ...]
) -> tuple[np.ndarray[Any, float], np.ndarray[Any, float], float, float]:
    """
    Compute propensities of H3 methylation, H3 demethylation, gene transcription and protein
    degradation based on the current state.

    Parameters
    ----------
    protQuant : float64
        Protein quantity.
    meState : NAarray[int32], shape (H,)
        Methylation state of chromatin H3 histone at target gene locus.
    all_param : tuple[float64, ...]
        A 27-element tuple specify the model parameters value, in the order of `T`, `T_0`, `alpha`,
        `theta_max`, `sigma`, `K_d`, `f_min`, `f_max`, `f_lim`, `P_t`, `gamma_transcr`, `n_ppt`,
        `kappa`, `beta_max`, `b`, `e_distal`, `rho`, `k_me`, `P_dem`, `P_ex`, `k_me01`, `k_me12`,
        `k_me23`, `gamma_me01`, `gamma_me12`, `gamma_me23`, `gamma_dem`.

    Returns
    -------
    mePropensity : NDArray[float64], shape (H,)
        Methylation propensity for each H3 histone.
    demPropensity : NDArray[float64], shape (H,)
        Demethylation propensity for each H3 histone.
    exprPropensity : float64
        Gene transcription propensity.
    pdegPropensity : float64
        Protein degradation propensity.
    """

    # Unpack parameters
    (
        T,
        T_0,
        alpha,
        theta_max,
        sigma,
        K_d,
        f_min,
        f_max,
        f_lim,
        P_t,
        gamma_transcr,
        _,  # n_ppt
        kappa,
        beta_max,
        b,
        e_distal,
        rho,
        _,  # k_me
        _,  # P_dem
        _,  # P_ex
        k_me01,
        k_me12,
        k_me23,
        gamma_me01,
        gamma_me12,
        gamma_me23,
        gamma_dem,
    ) = all_param

    # intermediate variable
    P_me23 = calc_P_me23(meState)  # inhibtory methylation ratio
    E = calc_E(meState, rho, e_distal)  # neighbor enhancement
    theta = calc_theta(
        protQuant, theta_max, sigma, K_d
    )  # transcriptional efficiencyactor-dependent activation level
    beta = calc_beta(T, T_0, beta_max, b=b)  # effective PRC2 activity

    # compute propensities
    mePropensity = beta * (
        (gamma_me01 + k_me01 * E) * (meState == 0)
        + (gamma_me12 + k_me12 * E) * (meState == 1)
        + (gamma_me23 + k_me23 * E) * (meState == 2)
    )
    demPropensity = gamma_dem * (meState > 0)
    exprPropensity = (
        min(alpha * theta * (f_max - min(P_me23 / P_t, 1) * (f_max - f_min)), f_lim) + gamma_transcr
    )
    pdegPropensity = protQuant * kappa

    return mePropensity, demPropensity, exprPropensity, pdegPropensity


@nb.njit(
    signature_or_function=nb.types.Tuple(types=(nb.float64, nb.int32, nb.int32))(
        nb.float64, nb.int32[:], nb.types.UniTuple(dtype=nb.float64, count=27)
    ),
    cache=True,
)
def gillespie_draw(
    protQuant: float, meState: np.ndarray[Any, int], all_param: tuple[float, ...]
) -> tuple[float, int, int]:
    """
    Draws a event and the time it took to do that event in a poisson process.

    Parameters
    ----------
    protQuant : float64
        Protein quantity.
    meState : NAarray[int32], shape (H,)
        Methylation state of chromatin H3 histone at target gene locus.
    all_param : tuple[float64, ...]
        A 27-element tuple specify the model parameters value, in the order of `T`, `T_0`, `alpha`,
        `theta_max`, `sigma`, `K_d`, `f_min`, `f_max`, `f_lim`, `P_t`, `gamma_transcr`, `n_ppt`,
        `kappa`, `beta_max`, `b`, `e_distal`, `rho`, `k_me`, `P_dem`, `P_ex`, `k_me01`, `k_me12`,
        `k_me23`, `gamma_me01`, `gamma_me12`, `gamma_me23`, `gamma_dem`.

    Returns
    -------
    deltaT : float64
        Time interval (unit: sec) for next comming event.
    event_class : int32 (in 0, 1, 2, 3)
        Next event marks, with 0, 1, 2, 3 represents H3 methylation, H3 demethylation, gene
        transcription and protein degradation, respectively.
    histone_idx : int32 (in 0, 1, 2, ... , H-1)
        Histone index in which next event to be occured. Note that this value is useful only if
        methylation or demethylation occurs, otherwise it is set to H3 histone number (H) and meaningless.
    """

    # compute propensity distribution
    mePropensity, demPropensity, exprPropensity, pdegPropensity = gillespie_get_propensities(
        protQuant, meState, all_param
    )
    props = np.append(
        arr=np.concatenate((mePropensity, demPropensity)), values=[exprPropensity, pdegPropensity]
    )
    props_sum = props.sum()

    # compute next time
    # Principles
    # The interval time distribution of the Poisson process is exponential
    # X1 ~ Exp(k1), X2 ~ Exp(k2), then min(X1, X2) ~ Exp(k1 + k2)
    deltaT = np.random.exponential(scale=1.0 / props_sum)

    # draw event from this distribution
    q = np.random.rand() * props_sum
    idx = 0
    p_sum = 0.0
    while p_sum <= q:
        p_sum += props[idx]
        idx += 1

    # classify event
    if idx <= 2 * meState.size:
        event_class, histone_idx = (idx - 1) // meState.size, (idx - 1) % meState.size
    else:
        event_class, histone_idx = 3 + idx - props.size, meState.size

    return deltaT, event_class, histone_idx


@nb.njit(boundscheck=True, nogil=True, cache=True)
def gillespie_ssa(
    protQuant0: float,
    meState0: np.ndarray[Any, int],
    time_to_next_repl0: float,
    all_param: tuple[float, ...],
    time_records: np.ndarray[Any, float],
    sizeFactor: float = 1.1,
) -> tuple[np.ndarray[Any, float], np.ndarray[Any, int], np.ndarray[Any, float]]:
    """
    Gillespie stochastic simulation algorithm (SSA).

    Parameters
    ----------
    protQuant0 : float64
        Initial protein quantity at the start of simulation.
    meState0 : NDArray[int32], shape (H,)
        Initial methylation state of each H3 histone at the start of simulation.
    time_to_next_repl0 : float64
        Time interval (unit: hour) between simulation initiation and the 1st DNA replication after
        that. Note that if it set to 0.0 (in most case), then `protQuant0` and `meState0` refers
        to the model state at the very end of last cell cycle, and a cell division will do
        immediately.
    all_param : tuple[float64, ...]
        A 27-element tuple specify the model parameters value, in the order of `T`, `T_0`, `alpha`,
        `theta_max`, `sigma`, `K_d`, `f_min`, `f_max`, `f_lim`, `P_t`, `gamma_transcr`, `n_ppt`,
        `kappa`, `beta_max`, `b`, `e_distal`, `rho`, `k_me`, `P_dem`, `P_ex`, `k_me01`, `k_me12`,
        `k_me23`, `gamma_me01`, `gamma_me12`, `gamma_me23`, `gamma_dem`.
    time_records : NDArray[float64], shape (T,)
        Array of time points (unit: hour) at which to monitor the model state.
    sizeFactor : float64 (default=1.1)
        Factor that controls the initial array size of return `transcrT_records`. See **NOTE**.

    Returns
    -------
    protQuant_records : NDArray[float64], shape (T,)
        1-D array, with entry t is the protein quantity at time_records[t]
    meState_records : NDArray[int32], shape (T, H)
        2-D array, with entry (t, h) is the methylation state of histone with index h at time_records[t]
    transcrT_records : NDArray[float64], shape (#,)
        1-D array with enough length that store the time of every transcription event, empty spaces
        are filled with NaN. "#" indicates the length is nontrivial, which correlates with function
        parameter `sizeFactor`, `time_records` and model parameter `f_lim`.

    **NOTE** By default, Numba will not do bounds checking after compilation. However, out of
    bounds accesses sometimes can produce garbage results, segfaults or crash. In this function,
    we add bounds check to JIT, which may slightly decrease the performance.
    """

    # extract some key parameters
    T = all_param[0]
    f_lim = all_param[8]
    n_ppt = all_param[11]
    P_dem = all_param[18]
    P_ex = all_param[19]

    # initialize output
    protQuant_records = np.empty(shape=(time_records.size,), dtype=np.float64)
    protQuant_records[0] = protQuant0

    meState_records = np.empty(shape=(time_records.size, meState0.size), dtype=np.int32)
    meState_records[0, :] = meState0

    transcrT_records = np.full(
        shape=(int(sizeFactor * (time_records[-1] - time_records[0]) * 3600 * f_lim),),
        fill_value=np.nan,
        dtype=np.float64,
    )  # initialize a huge array to store transcription time; increase 'sizeFactor' value if IndexError is raised
    transcr_idx = 0

    # temporary variables
    curr_protQuant = protQuant0  # current gene expression
    curr_meState = meState0.copy()  # current methylation state
    curr_time = time_records[0]  # current time [hour]
    next_time_records_idx = (
        1  # next time index at which gene expression and methylation state need to be recorded
    )
    next_repl_time = time_records[0] + time_to_next_repl0  # next DNA replication time [hour]

    # evolution loop
    while next_time_records_idx < time_records.size:
        while curr_time < time_records[next_time_records_idx]:
            # draw the event and interval time
            deltaT, event_class, histone_idx = gillespie_draw(
                protQuant=curr_protQuant, meState=curr_meState, all_param=all_param
            )
            delta_time = deltaT / 3600  # convert unit from second to hour

            # save current model state before evolution
            prev_protQuant = curr_protQuant
            prev_meState = curr_meState.copy()

            if curr_time + delta_time < next_repl_time:
                curr_time += delta_time  # update time

                # update current methylation state
                if event_class == 0:
                    curr_meState[histone_idx] += 1  # H3 methylation
                elif event_class == 1:
                    curr_meState[histone_idx] -= 1  # H3 demethylation
                elif event_class == 2:
                    curr_protQuant += n_ppt  # gene expression

                    # transcription-coupled demethylation
                    curr_meState[
                        (np.random.rand(curr_meState.size) < P_dem) & (curr_meState > 0)
                    ] -= 1

                    # transcription-coupled histone exchange
                    # histones with indices 2i and 2i+1 belong to the same nucleosome
                    nucleosome_idx = np.argwhere(
                        np.random.rand(curr_meState.size // 2) > (1 - P_ex) ** 2
                    ).ravel()
                    curr_meState[2 * nucleosome_idx] = 0
                    curr_meState[2 * nucleosome_idx + 1] = 0

                    # record transcription time
                    transcrT_records[transcr_idx] = curr_time
                    transcr_idx += 1
                else:
                    curr_protQuant = max(curr_protQuant - 1, 0.0)  # protein degradation
            else:
                # reset time when meeting cell cycle
                curr_time = next_repl_time

                # nucleosomes reassemble after DNA-replication
                nucleosome_idx = np.argwhere(np.random.rand(curr_meState.size // 2) > 0.5).ravel()
                curr_meState[2 * nucleosome_idx] = 0
                curr_meState[2 * nucleosome_idx + 1] = 0

                # update next replication time
                next_repl_time += T

        # update methylation state from next recording time to current time
        temp_idx = np.searchsorted(
            time_records > curr_time, True
        )  # the first time index after current time
        protQuant_records[next_time_records_idx:temp_idx] = prev_protQuant
        meState_records[next_time_records_idx:temp_idx] = prev_meState

        # update next recording time index
        next_time_records_idx = temp_idx

    return protQuant_records, meState_records, transcrT_records


@nb.njit(parallel=True, nogil=True, cache=True)
def gillespie_ssa_parallel(
    protQuant0: np.ndarray[Any, float],
    meState0: np.ndarray[Any, int],
    time_to_next_repl0: np.ndarray[Any, float],
    all_param: tuple[float, ...],
    time_records: np.ndarray[Any, float],
    p_bar: nbp.ProgressBar | None = None,
    sizeFactor: float = 1.1,
) -> tuple[np.ndarray[Any, float], np.ndarray[Any, int], np.ndarray[Any, float]]:
    """
    Multiple samples parallel version of function `gillespie_ssa.gillespie_ssa`, each sample is
    an independent parallel computing branch. This function provide a compatible API for class
    `GeneChromModel`.

    Parameters
    ----------
    protQuant0 : NDArray[floa64], shape (N,)
        Initial gene expression quantity of each trial at the start of simulation, its length is
        the number of total trials (N).
    meState0 : NDArray[int32], shape (N, H)
        Initial methylation state of chromatin H3 histone of each trial at the start of simulation.
        Axis 0 represents trial, axis 1 represents histone.
    time_to_next_repl0 : NDArray[float64], shape (N,)
        Time interval (unit: hour) between the simulation initiation time and the 1st DNA replication
        after that of each trial.
    all_param : tuple[float64, ...]
        A 27-element tuple specify the model parameters value, in the order of `T`, `T_0`, `alpha`,
        `theta_max`, `sigma`, `K_d`, `f_min`, `f_max`, `f_lim`, `P_t`, `gamma_transcr`, `n_ppt`,
        `kappa`, `beta_max`, `b`, `e_distal`, `rho`, `k_me`, `P_dem`, `P_ex`, `k_me01`, `k_me12`,
        `k_me23`, `gamma_me01`, `gamma_me12`, `gamma_me23`, `gamma_dem`.
    time_records : NDArray[float64], shape (T,)
        Array of time points (unit: hour) at which to monitor the model state.
    p_bar : ProgressBar (optional)
        A numba implementation object of tqdm to show the progress.
    sizeFactor : float64 (default=1.1)
        Factor that controls the initial array size of return `transcrT_records`. See **NOTE**.

    Returns
    -------
    samples_protQuant : NDArray[float64], shape (N, T)
        2-D array, with entry (n, t) is the gene expression quantity at time_records[t] in trial[n].
    samples_meState : NDArray[int32], shape (N, T, H)
        3-D array, with entry (n, t, h) is the methylation state of histone with index h at
        time_records[t] in trial[n].
    samples_transcrT : NDArray[float64], shape (N, #)
        2-D array with enough length that store the time of every transcription event in each trial,
        empty spaces are filled with NaN. '#' indicates the length is nontrivial, see function
        'gillespie_ssa' for more details.
    """

    # check
    assert protQuant0.size == meState0.shape[0] and meState0.shape[0] == time_to_next_repl0.size
    N, H = meState0.shape

    # extract some key parameters
    f_lim = all_param[8]

    # initialize output
    samples_protQuant = np.empty(shape=(N, time_records.size), dtype=np.float64)
    samples_meState = np.empty(shape=(N, time_records.size, H), dtype=np.int32)
    samples_transcrT = np.empty(
        shape=(N, int(sizeFactor * (time_records[-1] - time_records[0]) * 3600 * f_lim)),
        dtype=np.float64,
    )

    # parallel loop
    for n in nb.prange(N):
        samples_protQuant[n], samples_meState[n], samples_transcrT[n] = gillespie_ssa(
            protQuant0=protQuant0[n],
            meState0=meState0[n],
            time_to_next_repl0=time_to_next_repl0[n],
            all_param=all_param,
            time_records=time_records,
            sizeFactor=sizeFactor,
        )
        if p_bar is not None:
            p_bar.update(1)  # update

    return samples_protQuant, samples_meState, samples_transcrT
