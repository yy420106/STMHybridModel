import math
from typing import Any, List, Optional, Tuple, Union, Self

import numba_progress as nbp
import numpy as np
from gillespie_ssa import gillespie_get_propensities, gillespie_ssa_parallel
from parameters import Parameters
from util_funcs import (
    calc_P_me23,
    calc_prot_fixed_points,
    calc_time_to_next_repl_after_ccc,
    calc_time_to_next_repl_after_ev,
)

__all__ = ["STMGeneModel"]


class STMGeneModel(Parameters):
    """A gene expression-chromatin modification-cell division coupled model at STM gene locus."""

    __L = 3482  # gene length (bp)

    def __init__(
        self,
        N: int = 1,
        protQuant: Union[float, np.ndarray[Any, float], None] = None,
        meState: Optional[np.ndarray[Any, int]] = None,
        meState_fastBuild: Union[int, Tuple[int, ...]] = -1,
        time_to_next_repl: Union[float, np.ndarray[Any, float], None] = None,
        time: float = 0.0,
        **kwargs: float,
    ) -> None:
        """
        Constructor of class `STMGeneModel`.

        Parameters
        ----------
        N : int32 (default=1)
            The number of sub-models.
        protQuant : float64 | NDArray[float64], shape (N,) (optional)
            Protein quantity in each sub-model. If a single float number is passed, it will be
            broadcasted to all sub-models. By default, it will be set randomly.
        meState : NDArray[int32], shape (H,) or (N, H) (optional)
            Methylation state of H3 histone at target gene locus in each sub-model. If a 1-D array
            is passed, it will be broadcasted to all sub-models. Note that this parameter has a
            higher priority than `meState_fastBuild`.
        meState_fastBuild : int32 | Tuple[int32, ...] (default=-1)
            Fast building method of chromatin methylation state. It should be a integer or a tuple
            of integer represents the building method for each sub-model, in the latter case, the
            tuple length must be N. Only -1, 0, 1, 2 or 3 is valid, where -1 use random setting
            and other k refers to uniformly k-methylation. If a single integer is provided, it will
            be broadcasted to all sub-models.
        time_to_next_repl : float64 | NDArray[float64], shape (N,) (optional)
            Time interval (unit: hour) to next DNA replication (cell division) in each sub-model.
            By default, they are set randomly among all possibilities. Note that, 0.0 is also
            acceptable when cell cycle is infinte (division stop), in this case `self.protQuant`
            and `self.meState` refers to the model state at the very end of the last presumed cell
            cycle. If a single float is provided, it will be broadcasted to all sub-models.
        time : float64 (default=0.0)
            Timestamp (unit: hour) that represents the objective time.
        **kwargs : float64
            Keyword arguments that specify the free parameters. See class `Parameters` for details.

        Attributes
        ----------
        self.param_dict : Dict[str, float64] (inherited from class `Parameters`)
        self.H : int32
        self.N : int32
        self.protQuant : NDArray[float64], shape (N,)
        self.meState : NDArray[int32], shape (N, H)
        self.time_to_next_repl : NDArray[float], shape (N,)
        self.time : float64
        """

        assert N > 0, "The number of sub-models must be greater than 0."

        # initialize class attributes
        super(STMGeneModel, self).__init__(**kwargs)  # build attributes 'param_dict'

        self.H = 2 * math.ceil(self.__L / 200)  # number of H3 histones; a nucleosome ~200bp
        self.N = N

        # update attributes 'protQuant'
        if protQuant is None:
            # randomly set
            self.protQuant = np.random.randint(low=0, high=100 * self.H, size=(self.N,)).astype(
                np.float64
            )
        elif isinstance(protQuant, np.ndarray):
            assert protQuant.shape == (self.N,), "The shape of 'protQuant' is not compatible."
            assert np.all(protQuant >= 0), "Negative value is not acceptable for 'protQuant'."
            self.protQuant = protQuant.astype(np.float64)
        else:
            assert protQuant >= 0, "Negative value is not acceptable for 'protQuant'."
            self.protQuant = np.full(shape=(self.N,), fill_value=protQuant, dtype=np.float64)

        # update attributes 'meState'
        if meState is None:
            if isinstance(meState_fastBuild, tuple):
                assert len(meState_fastBuild) == self.N, (
                    "The length of 'meState_fastBuild' is not compatible."
                )
                # initialize
                self.meState = np.empty(shape=(self.N, self.H), dtype=np.int32)
                for idx in range(self.N):
                    if meState_fastBuild[idx] == -1:
                        self.meState[idx] = np.random.randint(low=0, high=4, size=(self.H,))
                    elif meState_fastBuild[idx] in [0, 1, 2, 3]:
                        self.meState[idx] = np.full(
                            shape=(self.H,), fill_value=meState_fastBuild[idx], dtype=np.int32
                        )
                    else:
                        raise ValueError("Invalid option of 'meDtate_fastBuild' is provided.")
            else:
                if meState_fastBuild == -1:
                    # randomly set
                    self.meState = np.random.randint(low=0, high=4, size=(self.N, self.H))
                elif meState_fastBuild in [0, 1, 2, 3]:
                    self.meState = np.full(
                        shape=(self.N, self.H), fill_value=meState_fastBuild, dtype=np.int32
                    )
                else:
                    raise ValueError("Invalid option of 'meDtate_fastBuild' is provided.")
        else:
            assert 1 <= meState.ndim <= 2, "Only 1-D or 2-D array is acceptable for 'meState'."
            assert np.isin(meState, [0, 1, 2, 3]).all(), (
                "Invalid methylation state in 'meState' is provided."
            )
            if meState.ndim == 2:
                assert meState.shape == (self.N, self.H), (
                    "The shape of 'meState' is not compatible."
                )
                self.meState = meState.astype(np.int32)
            else:
                assert meState.shape == (self.H,), "The shape of 'meState' is not compatible."
                self.meState = np.tile(meState.astype(np.int32), reps=(self.N, 1))

        # update attributes 'time_to_next_repl'
        if time_to_next_repl is None:
            # randomly set
            if self.param_dict["T"] == np.inf:
                self.time_to_next_repl = np.random.choice((0.0, np.inf), size=N, replace=True)
            else:
                self.time_to_next_repl = np.random.uniform(0.0, self.param_dict["T"], size=N)
        else:
            if isinstance(time_to_next_repl, np.ndarray):
                assert time_to_next_repl.shape == (self.N,), (
                    "The shape of 'time_to_next_repl' is not compatible."
                )
                self.time_to_next_repl = time_to_next_repl.astype(np.float64)
            else:
                self.time_to_next_repl = np.full(
                    shape=(self.N,), fill_value=time_to_next_repl, dtype=np.float64
                )

            # check time compatibility
            if self.param_dict["T"] == np.inf:
                assert np.all(
                    (self.time_to_next_repl == np.inf) | (self.time_to_next_repl == 0.0)
                ), "Logical problem: 'time_ro_next_repl' and 'T' is not compatible."
            else:
                assert (
                    0 <= self.time_to_next_repl.min()
                    and self.time_to_next_repl.max() < self.param_dict["T"]
                ), "Logical problem: 'time_ro_next_repl' and 'T' is not compatible."

        self.time = time

    @property
    def all_param(self) -> Tuple[float, ...]:
        return super().get_param(param_names="all")

    @property
    def prot_fp_param(self) -> Tuple[float, ...]:
        return super().get_param(param_names="prot_fp")

    def set_free_param(self, **kwargs: float) -> None:
        """
        Set values of model free parameters.

        Parameters
        ----------
        **kwargs: float64
            Keyword arguments that specify the free model parameters to their new values.

        **NOTE** This function rewrites the father class method and makes some extensions. If
        free parameter `T` is to be changed, then an additional modification of model attribute
        `self.time_to_next_repl` will be done accordingly.
        """

        try:
            # check if cell cycle is changed
            new_T = kwargs["T"]
            old_T = self.param_dict["T"]
            self.time_to_next_repl = calc_time_to_next_repl_after_ccc(
                old_time_to_next_repl=self.time_to_next_repl, old_T=old_T, new_T=new_T
            )  # update time to next replication
        except KeyError:
            pass

        super().set_free_param(**kwargs)  # update free parameters

    def get_propensities(
        self,
    ) -> List[Tuple[np.ndarray[Any, float], np.ndarray[Any, float], float, float]]:
        """
        Compute current propensities of 4 possible event: H3 methylation, H3 demethylation, gene
        transcription & protein degreadtion for each sample.

        Return a list containing N tuples, where each tuple represents a sample and composed of
        4 elements.
        """
        return [
            gillespie_get_propensities(self.protQuant[i], self.meState[i], self.all_param)
            for i in range(self.N)
        ]

    def get_fixed_points(self) -> List[np.ndarray[Any, float]]:
        """
        Compute the fixed number of protein molecules for each sample. Return a list containing N
        arrays.
        """
        return [
            calc_prot_fixed_points(calc_P_me23(self.meState[i]), *self.prot_fp_param)
            for i in range(self.N)
        ]

    def evolve(
        self,
        ev_time: float,
        time_step: float,
        p_bar: Optional[nbp.ProgressBar] = None,
        sizeFactor: float = 1.1,
    ) -> Tuple[
        np.ndarray[Any, float],
        np.ndarray[Any, float],
        np.ndarray[Any, int],
        np.ndarray[Any, float],
    ]:
        """
        Update model over time and record the intermediate state during evolution.

        Parameters
        ----------
        ev_time : float64
            Evolution time (unit: hour). Non-negative (include 0) is acceptable.
        time_step : float64
            Record time step.
        p_bar : ProgressBar (optional)
            A numba implementation object of tqdm to show the progress.
        sizeFactor : float64 (default=1.1)
            Factor that controls the initial array size of `samples_transcrT`. See **NOTE**.

        Returns
        -------
        time_records : NDArray[float64], shape (T,)
            Array of time points (unit: hour) at which to monitor the model state.
        samples_protQuant : NDArray[float64], shape (N, T)
            Records of gene expression quantity (protein level) during evolution.
        samples_meState : NDArray[int32], shape (N, T, H)
            Records of the methylation state of histone during evolution.
        samples_transcrT : NDArray[float64], shape (N, #)
            Array that store the time of every transcription event in each trial, empty spaces are filled with
            NaN. "#" indicates the length is non-constant.

        **NOTE** If out-of-bounds (index error) occurs, consider increasing parameter `sizeFactor`.
        Usually the default value (1.1) is enough.
        """

        # set record time points
        if ev_time > 0:
            n = ev_time / time_step
            time_records = np.linspace(self.time, self.time + int(n) * time_step, int(n) + 1)
            if n != int(n):
                time_records = np.append(time_records, self.time + ev_time)  # add the end time
        elif ev_time == 0:
            time_records = np.array([self.time])
        else:
            raise ValueError("'ev_time' must be non-negative.")

        # evolution & sampling
        samples_protQuant, samples_meState, samples_transcrT = gillespie_ssa_parallel(
            self.protQuant,
            self.meState,
            self.time_to_next_repl,
            self.all_param,
            time_records,
            p_bar,
            sizeFactor,
        )

        # update model state to the final time
        self.protQuant = samples_protQuant[:, -1]
        self.meState = samples_meState[:, -1, :]
        self.time_to_next_repl = calc_time_to_next_repl_after_ev(
            self.time_to_next_repl, self.param_dict["T"], ev_time
        )
        self.time += ev_time

        return time_records, samples_protQuant, samples_meState, samples_transcrT

    def extract_sub_model(self, indices_or_condition: np.ndarray[Any, int | bool]) -> Self:
        """Extract sub-model from a existed `STMGeneModel` object."""
        return self.__class__.__extract_sub_model(
            model=self, indices_or_condition=indices_or_condition
        )

    @classmethod
    def __extract_sub_model(
        cls, model: Self, indices_or_condition: np.ndarray[Any, int | bool]
    ) -> Self:
        """
        Extract sub-model from a existed 'STMGeneModel' object. (private method)

        Parameters
        ----------
        model : STMGeneModel
            The father model to be extracted.
        indices_or_condition : NDArray[int | bool], shape (#,)
            Indices (integer array) of sub-model or a filter condition (bool array) that specify
            the samples to be extracted.

        Returns
        -------
        sub_model : STMGeneModel
            The sub-model that contains those single sample which satisfy the condition or with
            the queried indices.
        """

        assert isinstance(indices_or_condition, np.ndarray), (
            "'indices_or_condition' must be a array."
        )

        # save parameters
        kwargs = {key: model.param_dict[key] for key in model.default_free_param_dict().keys()}

        if indices_or_condition.dtype == int:
            # indices
            sub_model = cls(
                N=len(indices_or_condition),
                protQuant=model.protQuant.take(indices=indices_or_condition, axis=0),
                meState=model.meState.take(indices=indices_or_condition, axis=0),
                time_to_next_repl=model.time_to_next_repl.take(
                    indices=indices_or_condition, axis=0
                ),
                time=model.time,
                **kwargs,
            )
        elif indices_or_condition.dtype == bool:
            # condition
            sub_model = cls(
                N=indices_or_condition.sum(),
                protQuant=model.protQuant[indices_or_condition],
                meState=model.meState[indices_or_condition],
                time_to_next_repl=model.time_to_next_repl[indices_or_condition],
                time=model.time,
                **kwargs,
            )
        else:
            raise TypeError(
                "Only 'int' or 'bool' is acceptable data type for array 'indices_or_condition'."
            )

        return sub_model
