from typing import Any, Tuple, Union

import numpy as np

__all__ = [
    "get_samples_mek_ratio",
    "get_samples_gene_activity",
    "get_samples_gene_P_onoff",
    "get_samples_bistability",
    "get_samples_prot_1stUpDnTime",
    "get_samples_epi_1stTransTime",
]


def get_samples_mek_ratio(
    samples_meState: np.ndarray[Any, int], k: Union[int, Tuple[int, ...]]
) -> np.ndarray[Any, float]:
    """
    Compute percentage of specific methylation type over the whole gene locus of each sample.
    This fucntion asumes 3-D input with multiple samples.

    Parameters
    ----------
    samples_meState : NDArray[int32], shape (N, T, H)
        Gillespie simulation result of methylation state, with each dimension represents trial
        (N), sampling time (T) and histone (H), respectively.
    k : int32 | Tuple[int32, ...]
        k-th methylation to be computed. Acceptable value: 0, 1, 2, 3.

    Returns
    -------
    samples_mek_ratio : NDArray[float64], shape (N, T)
        Sum of k-th methylation ratio profile for each sample.
    """

    samples_mek_ratio = (
        np.sum(np.isin(element=samples_meState, test_elements=k), axis=2) / samples_meState.shape[2]
    )

    return samples_mek_ratio


def get_samples_gene_activity(
    samples_transcrT: np.ndarray[Any, float], bins: int, range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get histogram information of gene transcription. This function assumes 2-D input with
    multiple samples.

    Parameters
    ----------
    samples_transcrT : NDArray[float64], shape (N, #)
        Transcription time for each sample in simulation, with extra spaces filled with NaN.
    bins : int32
        Number of bins in distribution histogram
    range : Tuple[float, float]
        Time range.

    Returns
    -------
    samples_gene_activity : NDArray[float64], shape (`bins`,)
        Gene activity in each time bins, defined as the average transcription number during that
        interval.
    bin_mid : NDArray[float64], shape (`bins`,)
        Mid-points for each bins, used in plotting.
    """

    hist, bin_edges = np.histogram(samples_transcrT, bins=bins, range=range)

    samples_gene_activity = hist / samples_transcrT.shape[0]
    bin_mid = (bin_edges[:-1] + bin_edges[1:]) / 2  # mid-point

    return samples_gene_activity, bin_mid


def get_samples_gene_P_onoff(
    samples_meState: np.ndarray[Any, int], P_t: float
) -> Tuple[np.ndarray[Any, float], np.ndarray[Any, float]]:
    """
    Compute possibility of gene in activated or silent state. Gene is defined active when the
    ratio of inhibitory methylation (Pme23) exceeds `3*Pt/4`, and silent when Pme23 under `Pt/4`.
    This function assumes 3-D input with multiple samples.

    **NOTE** According to the computational strategy, to get a confident result, the sampling times
    are recommanded to be evenly distributed throughout the simulation.

    Parameters
    ----------
    samples_meState : NDArray[int32], shape (N, T, H)
        Gillespie simulation result of methylation state, with each dimension represents trial
        (N), sampling time (T) and histone (H), respectively.
    P_t : float
        Model free parameter. Threshold ratio of me2/me3 to reach strongest repression plateau.

    Returns
    -------
    samples_P_on : NDArray[float64], shape (N,)
        Probability of each sample to be in silent state.
    samples_P_off : NDArray[float64], shape (N,)
        Probability of each sample to be in active state.
    """

    # get me2/me3 ratio
    samples_me23_ratio = get_samples_mek_ratio(samples_meState=samples_meState, k=(2, 3))

    # next average over different time points
    samples_P_on = np.sum(samples_me23_ratio < P_t / 4, axis=-1) / samples_me23_ratio.shape[-1]
    samples_P_off = np.sum(samples_me23_ratio > 3 * P_t / 4, axis=-1) / samples_me23_ratio.shape[-1]

    return samples_P_on, samples_P_off


def get_samples_bistability(
    samples_meState: np.ndarray[Any, int],
    time_records: np.ndarray[Any, float],
    time_to_first_repl: float,
    T: float,
    P_t: float,
) -> float:
    """
    Calculate bistability measure Bi (`4*Pon*Poff`) of gene in the simulaion. For stability, only
    calculate time points that in the last hour before every replication

    Parameters
    ----------
    samples_meState : NDArray, shape (N, T, H)
        Gillespie simulation result of methylation state, with each dimension represents trial
        (N), sampling time (T) and histone (H), respectively.
    time_records : NDArray, shape (T,)
        Array of points in the time (unit: hour) for which to record the methylation state.
    time_to_first_repl : float64
        Time interval (unit: hour) between `time_records[0]` and the first DNA replication time
        after that.
    T : float64
        Model free parameter. Cell cycle.
    P_t : float64
        Model free parameter. Threshold ratio of me2/me3 to reach strongest repression plateau.

    Returns
    -------
    samples_bistability : float64
        A value (0-1) to quatify the bistability, the closer to 1, the stronger bistable.
    """

    # the 1st replication time in sampling
    first_repl_time = time_records[0] + time_to_first_repl
    # compute next replication time for each time point
    next_repl_time = first_repl_time + np.ceil((time_records - first_repl_time) / T) * T
    time_to_next_repl = next_repl_time - time_records  # hour

    # extract valid time points
    valid_samples = samples_meState[:, (time_to_next_repl >= 0.0) & (time_to_next_repl < 1.0), :]

    # compute bistability measure
    samples_P_on, samples_P_off = get_samples_gene_P_onoff(valid_samples, P_t)
    samples_bistability = 4 * np.mean(samples_P_on) * np.mean(samples_P_off)

    return samples_bistability


def get_samples_prot_1stUpDnTime(
    samples_protQuant: np.ndarray[Any, float],
    time_records: np.ndarray[Any, float],
    threshold: float,
    updn: bool,
    start_idx: int = 0,
    time_unit_day: bool = True,
) -> np.ndarray[Any, float]:
    """
    Compute the up/down time for protein molecule to reach a threshold level in each sample.

    Parameters
    ----------
    samples_protQuant : NDArray[float64], shape (N, T)
        Protein quantity in simulation.
    time_records : NDArray[float64], shape (T,)
        Array of points in the time (unit: hour) for which to record the model state.
    theshold : float64
        Target value of protein number that to be examine.
    updn : bool
        Specify the direction of dynamic change towards `threshold` for statistics. If True, in
        growth (up) trend, otherwise in decline (down) trend.
    start_idx : int32 (default=0)
        The start index of time in `time_records`. It controls where to start observation.
    time_unit_day : bool (default=True)
        Whether to use day as time unit, otherwise use hour.

    Returns
    -------
    samples_prot_1stUpDnTime : NDArray[float64], shape (N,)
        An array containing the up/down time of each sample, invalid samples are represented
        with NaN.
    """

    assert time_records.size == samples_protQuant.shape[1]

    def search_updnTime(protQuant: np.ndarray[Any, float]) -> float:
        """Find the 1st time in gene expression array (1-D) to be greater/lesser than the threshold."""
        if updn:
            # up trend, find the 1st time for protein level to be greater than the threshold
            try:
                tmp_idx = start_idx + next(
                    idx for idx, val in enumerate(protQuant[start_idx:]) if val >= threshold
                )
                updnTime = time_records[tmp_idx] - time_records[start_idx]
            except StopIteration:
                updnTime = np.nan  # invalid sample
        else:
            # down trend, similar process as above
            try:
                tmp_idx = start_idx + next(
                    idx for idx, val in enumerate(protQuant[start_idx:]) if val <= threshold
                )
                updnTime = time_records[tmp_idx] - time_records[start_idx]
            except StopIteration:
                updnTime = np.nan
        return updnTime

    # vectorize
    samples_prot_1stUpDnTime = np.apply_along_axis(
        func1d=search_updnTime, axis=1, arr=samples_protQuant
    )
    if time_unit_day:
        samples_prot_1stUpDnTime /= 24.0

    return samples_prot_1stUpDnTime


def get_samples_epi_1stTransTime(
    samples_meState: np.ndarray[Any, float],
    time_records: np.ndarray[Any, float],
    P_t: float,
    on2off: bool,
    start_idx: int = 0,
    time_unit_day: bool = True,
) -> float:
    """
    Compute the epigenetic transition time of gene in each sample.

    Parameters
    ----------
    samples_meState : NDArray[int32], shape (N, T, H)
        Gillespie simulation result of methylation state, with each dimension represents trial
        (N), sampling time (T) and histone (H), respectively.
    time_records : NDArray[float64], shape (T,)
        Array of points in the time (unit: hour) for which to record the model state.
    P_t : float
        Model free parameter. Threshold ratio of me2/me3 to reach strongest repression plateau.
    on2off : bool
        Specify the gene state alteration. If True, from ON to OFF state; if False, vice versa.
    start_idx : int32 (default=0)
        The start index of time in `time_records`. It controls where to start observation.
    time_unit_day : bool (default=True)
        Whether to use day as time unit, otherwise use hour.

    Returns
    -------
    samples_epi_1stTransTime : NDArray[float64], shape (N,)
        Epigenetic transition time of each sample, invalid samples are filled with NaN.
    """

    assert time_records.size == samples_meState.shape[1]

    # get me2/me3 ratio
    samples_me23_ratio = get_samples_mek_ratio(samples_meState=samples_meState, k=(2, 3))

    def search_transTime(me23_ratio: np.ndarray[Any, float]) -> float:
        """Find the 1st time in me2/me3 ratio array (1-D) to be greater/lesser than corresponding threshold."""
        if on2off:
            # silence, find the 1st time for me2/me3 level to be greater than 3*Pt/4
            try:
                tmp_idx = start_idx + next(
                    idx for idx, val in enumerate(me23_ratio[start_idx:]) if val >= 3 * P_t / 4
                )
                transTime = time_records[tmp_idx] - time_records[start_idx]
            except StopIteration:
                transTime = np.nan  # invalid sample
        else:
            # activation, find the 1st time for me2/me3 level to be lesser than Pt/4
            try:
                tmp_idx = start_idx + next(
                    idx for idx, val in enumerate(me23_ratio[start_idx:]) if val <= P_t / 4
                )
                transTime = time_records[tmp_idx] - time_records[start_idx]
            except StopIteration:
                transTime = np.nan

        return transTime

    # vectorize
    samples_epi_1stTransTime = np.apply_along_axis(
        func1d=search_transTime, axis=1, arr=samples_me23_ratio
    )
    if time_unit_day:
        samples_epi_1stTransTime /= 24.0

    return samples_epi_1stTransTime
