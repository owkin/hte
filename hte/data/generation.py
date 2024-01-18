"""Data generation module."""

from collections.abc import Callable
import warnings
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Get the absolute path to the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the CSV file
csv_path = os.path.join(script_dir, "../configs/data_configs/semi_synthetic_p1000.csv")

# Load the CSV file
data_tcga = pd.read_csv(csv_path, index_col="patient_id")

# data_tcga = pd.read_csv("../configs/data_configs/semi_synthetic_p1000.csv",
#                         index_col='patient_id')


class DataSurvival():
    """Data Generation Process for time-to-event data.

    The code is based on:
    https://mathildesautreuil.github.io/survMS/articles/how-to-simulate-survival-models.html

    Parameters
    ----------
    model : Name of the survival model: Cox, AH or AFT.
        All these models correspond to transformations on the baseline hazard.
    base_hazard : Name of the functional form of the baseline hazard:
        Weibull or LogNormal.
    a : first parameter of the baseline hazard. It corresponds to lambda for
        Weibull and mu for LogNormal.
    b : second parameter of the baseline hazard. It corresponds to alpha for
        Weibull (shape parameter) and sigma for LogNormal.
    """

    def __init__(self, model='Cox', base_hazard='Weibull', a=1.0, b=1.0):
        self.model = model
        self.base_hazard = base_hazard
        self.params_hazard = {}
        if self.base_hazard == 'Weibull':
            self.params_hazard['lambd'] = a
            self.params_hazard['alpha'] = b
            assert a > 0 and b > 0
        elif self.base_hazard == 'LogNormal':
            self.params_hazard['mu'] = a
            self.params_hazard['sigma'] = b
            assert b > 0
        else:
            raise ValueError(
                f"Baseline Hazard {self.base_hazard} not implemented."
            )

    def cumulative_hazard(self, x):
        """Compute cumulative hazard."""
        if self.base_hazard == "Weibull":
            res = np.power(x, self.params_hazard['alpha'])
            res = res * self.params_hazard['lambd']
        elif self.base_hazard == "LogNormal":
            res = np.log(x) - self.params_hazard['mu']
            res = res / self.params_hazard['sigma']
            res = norm.cdf(res, loc=0, scale=1)
            res = - np.log(1 - res)
        return res

    def psi_fun(self, x):
        """Select the tuple psi depending on the model used."""
        if self.model == "Cox":
            res = [1.0, x]
        elif self.model == "AH":
            res = [x, 1.0 / x]
        elif self.model == "AFT":
            res = [x, 1.0]
        else:
            raise ValueError(
                f"Model {self.model} not implemented."
            )
        return res

    def inv_cumulative_hazard(self, x):
        """Inverse of the cumulative hazard function."""
        if self.base_hazard == 'Weibull':
            res = np.power(
                x / self.params_hazard['lambd'],
                1 / self.params_hazard['alpha']
            )
        elif self.base_hazard == 'LogNormal':
            res = np.exp(
                self.params_hazard['sigma'] * norm.ppf(1 - np.exp(-x))
                + self.params_hazard['mu']
            )
        else:
            raise ValueError(
                "Baseline Hazard {self.base_hazard} not implemented."
            )
        return res

    def generate_times(self, predslin):
        """Generate survival data with covariate effect."""
        psi = self.psi_fun(np.exp(predslin))
        return self.inv_cumulative_hazard(
            - np.log(np.random.uniform(0, 1)) / psi[1]
        ) / psi[0]


def generate_mixture(
    beta0 : float,
    beta1 : float,
    size : int,
    group : Callable[[np.ndarray], int],
    mean : np.ndarray,
    cov : np.ndarray,
    gamma : np.ndarray,
    model : str = 'Cox',
    base_hazard : str = 'Weibull',
    a : float = 1.0,
    b : float = 2.0,
    random_state : int = 42,
    censored : bool = False,
    scale : float = 1.0,
    semi_synth : bool = False,
) -> pd.DataFrame:
    """Generate a data with subgroups.

    For first implementation, the two dgp are only varying w.r.t. beta.

    Parameters
    ----------
    beta0 : float
        The coeff of T in Group 0.
    beta1 : float
        The coeff of T in Group 1.
    size : int
        The number of samples
    group: Callable[[np.array], int]
        Function with 0-1 values to define subgroups.
    mean : np.array
        The mean of the covariates.
    cov : np.array
        The covariance of the covariates.
    gamma : np.array
        Coefficients of the covariates.
    model : str
        Survival analysis model (param of the DataSurvival class).
    base_hazard : str
        Base Hazard of the DataSurvival class.
    a : float
        Parameter of the DataSurvival class.
    b : float
        Parameter of the DataSurvival class.
    random_state : int
        Seed for reproducibility.
    censored : bool
        Whether censorship occurs ot not.
    scale : float
        When censored is True, this is the scale of the exponential law.
    semi_synth : bool
        Whether to use semi-synthetic data or not.

    Returns
    -------
    df : pd.DataFrame of length 2*[size] with columns
        Times | Event | T | X1 | X2 | ... | Group
    """
    # the common underlying dgp
    dgp = DataSurvival(
        model=model,
        base_hazard=base_hazard,
        a=a,
        b=b
    )
    # randomness
    rng = np.random.default_rng(seed=random_state)
    # treatment
    trt_allocation = rng.binomial(
        1,
        0.5,
        size=[size, 1]
    )

    # covariables
    if semi_synth:
        idxs = list(np.random.choice(list(range(len(data_tcga))), size=size))
        X = data_tcga.iloc[idxs].reset_index().drop('patient_id', axis=1)
        X = X.values
    else:
        X = rng.multivariate_normal(
            mean=mean,
            cov=cov,
            size=size
        )
    # find subgroup
    group_allocation = np.array([group(x) for x in X])
    # linear predictions
    beta = [beta0, beta1]
    linpreds = np.array(
        [beta[g] * w for (w, g) in zip(trt_allocation, group_allocation)]
    )
    linpreds = linpreds.reshape(size)
    linpreds += (gamma @ X.T)
    # groundtruth times
    times = [dgp.generate_times(elem) for elem in linpreds]
    # censored times
    if censored:
        scale_to_beta = {
            1: [0.4, 0.4],
            2: [0.3, 1],
            3: [0.2, 2],
            4: [0.05, 10]
        }
        cnsr_times = 20 * np.random.beta(
            a=scale_to_beta[scale][0],
            b=scale_to_beta[scale][1],
            size=size
        )
        obs = []
        for i, elem in enumerate(zip(times, cnsr_times)):
            elem1, elem2 = elem
            # censor if elem2 < elem1
            if elem2 < elem1:
                times[i] = elem2
                obs.append(0)
            elif elem1 < elem2:
                obs.append(1)
    else:
        obs = [1] * size
    # dataframe
    df = pd.DataFrame(
        data=list(zip(times, obs)),
        columns=['T', 'E']
    )
    df['W'] = trt_allocation
    df['G'] = group_allocation
    # dimension of the covariables
    dim_cov = len(mean)
    for ii in range(dim_cov):
        df[f"X{ii + 1}"] = X[:, ii]

    return df


def generate_mixture_oracle(
    beta : float,
    size : int,
    group : Callable[[np.ndarray], int],
    mean : np.ndarray,
    cov : np.ndarray,
    gamma : np.ndarray,
    model : str = 'Cox',
    base_hazard : str = 'Weibull',
    a : float = 1.0,
    b : float = 2.0,
    timepoint : float = 1.,
    censored : bool = False,
    scale: float = 1.0,
    semi_synth : bool = False,
) -> pd.DataFrame:
    """Generate a data sample with groundtruth IARR.

    For first implementation, the two dgp are only varying w.r.t. beta.

    Parameters
    ----------
    beta : float
        Hazard ratio.
    size : int
        Number of samples.
    group: Callable[[np.array], int]
        Function with 0-1 values to define subgroups.
    mean : np.array
        The mean of the covariates.
    cov : np.array
        The covariance of the covariates.
    gamma : np.array
        Coefficients of the covariates.
    model : str
        Survival analysis model (param of the DataSurvival class).
    base_hazard : str
        Base Hazard of the DataSurvival class.
    a : float
        Parameters of the DataSurvival class.
    b : float
        Parameters of the DataSurvival class.
    timepoint : float
        The timepoint where ARR is computed
    censored : bool
        Whether censorship occurs ot not.
    scale : float
        The scale of the exponential law if censored is True.
    semi_synth : bool
        Whether to use semi-synthetic data or not.

    Returns
    -------
    df : pd.DataFrame of length 2*[size] with columns
        Times | Event | T | X1 | X2 | Group
    """
    # the common underlying dgp
    dgp = DataSurvival(
        model=model,
        base_hazard=base_hazard,
        a=a,
        b=b
    )
    # covariables
    if semi_synth:
        idxs = list(np.random.choice(list(range(len(data_tcga))), size=size))
        X = data_tcga.iloc[idxs].reset_index().drop('patient_id', axis=1)
        X = X.values
    else:
        X = np.random.multivariate_normal(
            mean=mean,
            cov=cov,
            size=size
        )
    # find subgroup
    group_allocation = np.array([group(x) for x in X])
    # linear predictions
    linpreds0 = beta * np.zeros(size)
    linpreds1 = beta * np.ones(size)
    linpreds0 = linpreds0.reshape(size)
    linpreds1 = linpreds1.reshape(size)
    linpreds0 += gamma @ X.T
    linpreds1 += gamma @ X.T
    # groundtruth times
    # print('Compute groundtruth times')
    times0 = [dgp.generate_times(elem) for elem in linpreds0]
    times1 = [dgp.generate_times(elem) for elem in linpreds1]
    # censored times
    if censored:
        cnsr_times = np.random.exponential(
            scale=scale,
            size=size
        )
        obs0 = [int(elem1 > elem2) for elem1, elem2 in zip(times0, cnsr_times)]
        obs1 = [int(elem1 > elem2) for elem1, elem2 in zip(times1, cnsr_times)]
    else:
        obs0 = [1] * size
        obs1 = [1] * size
    # probability of event
    survival_0 = [
        np.exp(- dgp.cumulative_hazard(timepoint) * np.exp(elem)) for elem in linpreds0
    ]
    survival_1 = [
        np.exp(- dgp.cumulative_hazard(timepoint) * np.exp(elem)) for elem in linpreds1
    ]
    # compute IARR
    # print('Compute IARR')
    iarr = [elem1 - elem0 for elem1, elem0 in zip(survival_1, survival_0)]
    # dataframe
    df_oracle = pd.DataFrame(
        data=list(zip(times1, obs1, times0, obs0)),
        columns=['T1', 'E1', 'T0', 'E0']
    )
    df_oracle['S1'] = survival_1
    df_oracle['S0'] = survival_0
    dim_cov = len(mean)
    for ii in range(dim_cov):
        df_oracle[f"X{ii + 1}"] = X[:, ii]
    df_oracle['IARR'] = iarr
    df_oracle['G'] = group_allocation
    ######################################
    return df_oracle


if __name__ == '__main__':

    def group_test(x):
        """Test group function for generate mixture."""
        return int(x[0] > 0)

    df = generate_mixture(
        beta0=1.,
        beta1=-1.,
        size=1000,
        group=group_test,
        mean=np.array([0., 0.]),
        cov=np.array([[1., 0.], [0., 1.]]),
        gamma=np.array([1., -1.]),
        model='Cox',
        base_hazard='Weibull',
        a=1.,
        b=2.,
        censored=False,
        scale=0.5
    )

    print(df.head())
