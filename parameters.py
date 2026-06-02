import copy
import math
from collections import OrderedDict
from operator import itemgetter
from typing import Tuple, Union

__all__ = ["load_default_free_param_dict", "update_depend_param", "Parameters"]

"""
Parameters Explanation
======================>
NOTE: Parameters with prefix * represents dependent parameters in this model. Their quantitative 
relations to other free parameters are shown after the in-line arrows.

T : Cell cyle [hour]
T_0 : Normalized cell cycle [hour]

alpha : Trans-acting activation level of STM
theta_max : Maximum self-activation level of STM
sigma : Hill coefficient
K_d : Apparent dissociation constant of ATH1-STM complex

f_min : Minimum transcription initiation rate [1/sec]
f_max : Maximum transcription initiation rate [1/sec]
f_lim : Upper bound of transcription initiation rate [1/sec]
P_t : Threshold proportion of H3K27me3/me2 marks at which maximal repression reaches
gamma_transcr : Random transcription rate [1/sec]
n_ppt : Average protein number to be translated per transcript

kappa : Protein degradation rate [1/sec]

beta_max : Maximum effective PRC2 activity
b : Exponential factor of PRC2 activity
e_distal : Distal gain contribution
rho : Activation level of PRC2 by 2-methylation relative to 3-methylation
k_me : Reference PRC2-mediated methylation rate [1/sec]
*k_me01 : PRC2-mediated 1-methylation rate [1/sec]  ==>  9 * k_me
*k_me12 : PRC2-mediated 2-methylation rate [1/sec]  ==>  6 * k_me
*k_me23 : PRC2-mediated 3-methylation rate [1/sec]  ==>  k_me
*gamma_me01 : Random 1-methylation rate by noise [1/sec]  ==>  k_me01 / 20
*gamma_me12 : Random 2-methylation rate by noise [1/sec]  ==>  k_me12 / 20
*gamma_me23 : Random 3-methylation rate by noise [1/sec]  ==>  k_me23 / 20

P_dem : Transcription-coupled demethylation probability
P_ex : Transcription-coupled histone exchange probability
*gamma_dem : integrated random demethylation rate [1/sec]  ==>  f_min * P_dem

Total: 27
<======================
"""


def load_default_free_param_dict() -> OrderedDict[str, float]:
    """
    Load free parameters with their default values.

    Returns
    -------
    free_param_dict : Dict[str, float]
        Dictionary that maps model free parameters to their default values.
    """

    free_param_dict = OrderedDict()

    # cell division
    free_param_dict["T"] = 22.0
    free_param_dict["T_0"] = 22.0
    # gene activation
    free_param_dict["alpha"] = 1.0
    free_param_dict["theta_max"] = 1.0
    free_param_dict["sigma"] = 2.0
    free_param_dict["K_d"] = 180.0
    # gene transcription & translation
    free_param_dict["f_min"] = 1e-4
    free_param_dict["f_max"] = 5e-3  # 4e-3
    free_param_dict["f_lim"] = 1 / 60
    free_param_dict["P_t"] = 1 / 3
    free_param_dict["gamma_transcr"] = 1e-8
    free_param_dict["n_ppt"] = 1.0
    # protein degradation
    free_param_dict["kappa"] = 5e-6  # 4e-6
    # histone methylation
    free_param_dict["beta_max"] = 2.5  # 3.0
    free_param_dict["b"] = math.log(5 / 3)  # math.log(3 / 2)
    free_param_dict["e_distal"] = 0.001
    free_param_dict["rho"] = 1 / 10
    free_param_dict["k_me"] = 1e-5
    # histone demethylation & exchange
    free_param_dict["P_dem"] = 6.4e-3  # 8e-3
    free_param_dict["P_ex"] = 1.5e-3

    return free_param_dict


def update_depend_param(param_dict: OrderedDict[str, float]) -> None:
    """
    Update values of dependent parameters.

    Parameters
    ----------
    param_dict : Dict[str, float]
        Dictionary that maps model parameters to their values.
    """

    f_min, k_me, P_dem = itemgetter("f_min", "k_me", "P_dem")(param_dict)

    # calculate dependent paramters
    param_dict["k_me01"] = 9 * k_me
    param_dict["k_me12"] = 6 * k_me
    param_dict["k_me23"] = k_me
    param_dict["gamma_me01"] = param_dict["k_me01"] / 20
    param_dict["gamma_me12"] = param_dict["k_me12"] / 20
    param_dict["gamma_me23"] = param_dict["k_me23"] / 20
    param_dict["gamma_dem"] = f_min * P_dem


class Parameters:
    """Manager of model parameters."""

    __free_param_dict = load_default_free_param_dict()  # store default values

    def __init__(self, **kwargs: float) -> None:
        """
        Constructor of class `Parameters`.

        Parameters
        ----------
        **kwargs : float
            Keyword arguments that specify the customized free parameter.
        """

        assert set(kwargs).issubset(set(self.__free_param_dict)), (
            f"Acceptable paramter names: {', '.join(self.__free_param_dict)}."
        )

        self.param_dict = copy.deepcopy(self.__free_param_dict)
        self.param_dict.update(kwargs)  # update free paramters
        self._update_depend_param()  # update dependent paramters

    def set_free_param(self, **kwargs: float) -> None:
        """
        Modify the parameters of an existed model.

        Parameters
        ----------
        **kwargs : float
            Keyword arguments that specify the model free parameter. If nothing passed, reset
            all free parameters to their default values.
        """

        if kwargs:
            assert set(kwargs).issubset(set(self.__free_param_dict)), (
                f"Acceptable paramter names: {', '.join(self.__free_param_dict)}."
            )
            self.param_dict.update(kwargs)  # update free paramters
        else:
            self.param_dict.update(self.__free_param_dict)  # update free paramters

        self._update_depend_param()  # update dependent paramters

    def get_param(
        self, param_names: Union[str, Tuple[str, ...]] = "all"
    ) -> Union[float, Tuple[float, ...]]:
        """
        Get values of model parameters.

        Parameters
        ----------
        param_names : str | Tuple[str] | List[str] (default="all")
            Speficy the paramters that to be extract. By default, return all values in the same
            order of parameter dictionary. For string input, besides of single parameter name,
            some special group names are acceptable as well:
                "all" - all parameters;
                "free" - free parameters;
                "depend" - dependent parameters;
                "prot_fp" - parameters used in protein fixed points computation.

        Returns
        -------
        param_vals : float | Tuple[float, ...]
            Values of queried parameters.
        """

        if isinstance(param_names, Tuple):
            param_vals = itemgetter(*param_names)(self.param_dict)
        else:
            if param_names == "all":
                param_vals = tuple(self.param_dict.values())
            elif param_names == "free":
                param_vals = itemgetter(*self.__free_param_dict.keys())(self.param_dict)
            elif param_names == "depend":
                param_vals = itemgetter(
                    "k_me01",
                    "k_me12",
                    "k_me23",
                    "gamma_me01",
                    "gamma_me12",
                    "gamma_me23",
                    "gamma_dem",
                )(self.param_dict)
            elif param_names == "prot_fp":
                param_vals = itemgetter(
                    "alpha",
                    "theta_max",
                    "sigma",
                    "K_d",
                    "f_min",
                    "f_max",
                    "f_lim",
                    "P_t",
                    "gamma_transcr",
                    "n_ppt",
                    "kappa",
                )(self.param_dict)
            else:
                param_vals = self.param_dict[param_names]

        return param_vals

    def _update_depend_param(self) -> None:
        return update_depend_param(param_dict=self.param_dict)

    @classmethod
    def default_free_param_dict(cls) -> OrderedDict[str, float]:
        """Return a dictionary that store the default value of model free parameters."""
        return cls.__free_param_dict
