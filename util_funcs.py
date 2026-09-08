import math
from typing import Any

import numba as nb
import numpy as np
import sympy as sp

__all__ = [  # noqa: RUF022
    "calc_P_me23",
    "calc_E",
    "calc_beta",
    "calc_theta",
    "calc_time_to_next_repl_after_ev",
    "calc_time_to_next_repl_after_ccc",
    "calc_prot_fixed_points",
]
# JIT compile: calc_P_me23, calc_E, calc_alpha, calc_omega, calc_theta
# these functions will be used in JIT-compiled Gillespie Stochastic Sampling Algorithm (SSA)


@nb.njit(signature_or_function=nb.float64(nb.int32[:]), cache=True)
def calc_P_me23(meState: np.ndarray[Any, int]) -> float:
    """
    Compute the percentage of repressive methylation marks (me2/me3) over target gene locus.

    Parameters
    ----------
    meState : NDArray[int32], shape (H,)
        Methylation state of chromatin H3 histone at target gene locus.

    Returns
    -------
    P_me23 : flotat64
       Ratio of repressive methylation modification.
    """

    P_me23 = np.sum(meState >= 2) / meState.size

    return P_me23


@nb.njit(signature_or_function=nb.float64[:](nb.int32[:], nb.float64, nb.float64), cache=True)
def calc_E(meState: np.ndarray[Any, int], rho: float, e_distal: float) -> np.ndarray[Any, float]:
    """
    Compute enhancement gain by neighbors and distal interaction for methylation propensities.

    Parameters
    ----------
    meState : NDArray[int32], shape (H,)
        Methylation state of chromatin H3 histone at target gene locus.
    rho : float64
        Model free parameter. Activation capacity of PRC2 by 2-methylation relative to 3-methylation.
    e_distal : float64
        Model free parameter. Distal gain.

    Returns
    -------
    E : NDArray[float64], shape (H,)
        Methylation enhancement gain by adjacent neighbors of each histone.
    """

    # initialize output
    E = np.empty(shape=(meState.size,), dtype=np.float64)

    for idx in range(meState.size):
        # histones with indices 2i and 2i+1 belong to the same nucleosome, neighbor histones are
        # defined within +1/-1 nucleosome
        if idx % 2 == 0:
            neighbor_meState = np.concatenate(
                (meState[max(idx - 2, 0) : idx], meState[idx + 1 : min(idx + 4, meState.size)])
            )
        else:
            neighbor_meState = np.concatenate(
                (meState[max(idx - 3, 0) : idx], meState[idx + 1 : min(idx + 3, meState.size)])
            )
        E[idx] = rho * np.sum(neighbor_meState == 2) + np.sum(neighbor_meState == 3) + e_distal

    return E


@nb.njit(
    signature_or_function=nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), cache=True
)
def calc_beta(T: float, T_0: float, beta_max: float, b: float) -> float:
    """
    Compute the effective PRC2 activity.

    Parameters
    ----------
    T : float64
        Model free parameter. Cell cycle.
    T_0 : float64
        Model free parameter. Normalized cell cycle.
    beta_max : float64
        Model free parameter. Maximum effective PRC2 activity
    b : float64
        Model free parameter. Exponential factor of PRC2 activity

    Returns
    -------
    beta : float64
        Effective PRC2 activity.
    """

    beta = beta_max * (1 - math.exp(-b * T / T_0))
    # beta = beta_max * (1 - math.exp(-b))  # without cell cycle-dependency
    
    return beta


@nb.njit(
    signature_or_function=nb.float64(nb.float64, nb.float64, nb.float64, nb.float64), cache=True
)
def calc_theta(protQuant: float, theta_max: float, sigma: float, K_d: float) -> float:
    """
    Compute cofactor (ATH1)-dependent gene self-activation level based on Hill equation.

    Parameters
    ----------
    protQuant : float64
        Protein quantity.
    theta_max : float64
        Model free parameter. Maximum self-activation level.
    sigma : float64
        Model free parameter. Hill coefficient.
    K_d : float64
        Model free parameter. Apparent dissociation constant.

    Returns
    -------
    theta : float64
        Cofactor-dependent gene self-activation level.
    """

    theta = theta_max * math.pow(protQuant, sigma) / (K_d + math.pow(protQuant, sigma))

    return theta


def calc_time_to_next_repl_after_ev(
    time_to_next_repl: np.ndarray[Any, float], T: float, ev_time: float
) -> np.ndarray[Any, float]:
    """
    Calculate the time to next DNA replication after model evolution with constant cell cycle.

    Parameters
    ----------
    time_to_next_repl : NDArray[float64], shape (N,)
        Time to next DNA replication in each trial before model evolution.
    T : float64
        Cell cycle of the model.
    ev_time : float64
        Evolution time (unit: hour).

    Returns
    -------
    new_time_to_next_repl : NDAarray[float64], shape (N,)
        Time to next DNA replication after evolution.
    """

    if math.isinf(T):
        new_time_to_next_repl = np.full(
            shape=time_to_next_repl.shape, fill_value=math.inf, dtype=np.float64
        )
    else:
        assert np.all(time_to_next_repl < T)
        new_time_to_next_repl = (
            time_to_next_repl + np.ceil((ev_time - time_to_next_repl) / T) * T - ev_time
        )

    return new_time_to_next_repl


def calc_time_to_next_repl_after_ccc(
    old_time_to_next_repl: np.ndarray[Any, float], old_T: float, new_T: float
) -> np.ndarray[Any, float]:
    """
    Calculate the time to next DNA replication after cell cycle changes based on equal-scale
    transformation. This means that, if a cell has 1 hour to divide, and then cell cycle doubles,
    the transformed cell will divide 2 hours later.

    Parameters
    ----------
    old_time_to_next_repl : NDArray[float64], shape (N,)
        Time to next DNA replication before cell cycle changes.
    ole_T : float64
        Old cell cycle.
    new_T : float64
        New cell cycle.

    Returns
    -------
    new_time_to_next_repl : NDArray[float64], shape (N,)
        Time to next DNA replication after cell cycle changes.

    **NOTE** If cell cycle is infinite, it represents that cell stops dividing. If a cell restart
    division from quiescence, `new_time_to_next_repl` is set to 0. If `old_time_to_next_repl` is
    0.0, the cell will immediately do a DNA replication and divide the next moment before new
    cycle is applied.
    """

    if math.isinf(old_T):
        if math.isinf(new_T):
            new_time_to_next_repl = np.full(shape=old_time_to_next_repl.shape, fill_value=math.inf)
        else:
            new_time_to_next_repl = np.full(shape=old_time_to_next_repl.shape, fill_value=0.0)
    else:
        assert np.all(old_time_to_next_repl < old_T)

        # equal-scale transform
        # special treatment for NaN
        # if `old_time_to_next_repl` is 0.0 and `new_cell_cycle` is infinite, set `new_time_to_next_repl`  to 0.0
        transform_func = np.vectorize(
            pyfunc=lambda x, y, z: 0.0 if x == 0 and z == math.inf else x / y * z
        )
        new_time_to_next_repl = transform_func(old_time_to_next_repl, old_T, new_T)

    return new_time_to_next_repl


def calc_prot_fixed_points(
    P_me23: float,
    alpha: float,
    theta_max: float,
    sigma: float,
    K_d: float,
    f_min: float,
    f_max: float,
    f_lim: float,
    P_t: float,
    gamma_transcr: float,
    n_ppt: float,
    kappa: float,
    p0_array: np.ndarray | None = None,
    solver_prec: int = 32,
    verify_tol: float = 3e-16,
    ndigits: float = 8,
) -> np.ndarray[Any, float]:
    """
    Compute the fixed number of protein molecules for the production-degradation system to be stable or
    metastable (critical) at different chromatin methylation state.

    Parameters
    ----------
    P_me23 : float64
        Repressive methylation (me2/me3) modification level of STM.
    alpha : float64
        Model free parameter. Trans-acting activation level of STM.
    theta_max : float64
        Model free parameter. Maximum effective activation level of local ATH1 protein.
    sigma : float64
        Model free parameter. Hill coefficient.
    Kd : float64
        Model free parameter. Dissociation constant.
    f_min : float64
        Model free parameter. Minimum transcription initiation rate.
    f_max : float64
        Model free parameter. Maximum transcription initiation rate.
    f_lim : float64
        Model free parameter. Upper bound of transcription initiation rate with activation.
    P_t : float64
        Model free parameter. Thereshold value of maximum repression.
    gamma_transcr : float64
        Model free parameter. Random transcription rate
    n_ppt : float64
        Model free parameter. Average protein number to be translated per transcript.
    kappa : float64
        Model free parameter. Protein degradation rate.
    p0_array : NDArray[float64], shape (#,) (optional)
        Initial points (prediction root) to be used in sympy numerical solving.
    solver_prec : int32 (default=32)
        Precision (decimal places) of sympy solver.
    verify_tol : float64 (default=3e-16)
        Tolerance of deviation from 0 in verification of equation roots.
    ndigits : int32 (default=8)
        Decimal precision of roots.

    Returns
    -------
    prot_crtitical_points : NDArray[float64], shape (#,)
        Array of possible crtical points (ascending order).
    """

    assert 0 <= P_me23 <= 1, "Acceptable range of 'P_me23' is from 0 to 1."

    # define protein dynamic function
    p = sp.Symbol("p")  # protein number
    f_lin = f_max - min(P_me23 / P_t, 1) * (f_max - f_min)  # linear piecewise
    prot_dynamic_func = (
        n_ppt * (alpha * theta_max * p**sigma / (K_d + p**sigma) * f_lin + gamma_transcr)
        - kappa * p
    )
    lim = alpha * theta_max * f_lin  # limit non-random transcription rate

    # define numerical solver
    def eq_solver(p0: float) -> float:
        """Get real and positive root for protein dynamic equation from a given initial point (p0)."""
        try:
            # by default, sympy solver will do verification while solving to only get reasonable root
            # however, that functions which are very steep near the root, the verification of the solution
            # may fail therefore, disable the built-in verification, use manually check instead
            root = complex(
                sp.nsolve(prot_dynamic_func, p, p0, verify=False, prec=solver_prec).evalf(chop=True)
            )
            # manually check, keep positive real root only
            if root.imag == 0 and root.real >= 0:
                root = root.real
                if abs(prot_dynamic_func.subs(p, root)) > verify_tol:
                    root = math.nan  # mistake
            else:
                root = math.nan  # imaginary root or real but negative root
        except ValueError:
            root = math.nan  # if no root at all

        return root

    p0_array = np.concatenate(([0], 10 ** np.arange(-4, 4, 0.1))) if p0_array is None else p0_array

    # solve equation
    roots = np.vectorize(pyfunc=eq_solver, otypes=[np.float64])(p0_array)  # vectorize solver
    valid_roots = np.unique(np.round(roots[~np.isnan(roots)], decimals=ndigits))

    # check bound
    lim_root = n_ppt * (f_lim + gamma_transcr) / kappa  # limit root (at limit transcription rate)
    if lim <= f_lim:
        # always below limit
        prot_critical_points = np.sort(valid_roots)
    else:
        # reach limit
        split_point = math.pow(
            K_d * (f_lim / lim) / (1 - f_lim / lim), 1 / sigma
        )  # where reach limit
        prot_critical_points = np.sort(valid_roots[valid_roots <= split_point])
        if lim_root > split_point:
            prot_critical_points = np.append(prot_critical_points, round(lim_root, ndigits=ndigits))

    return prot_critical_points
