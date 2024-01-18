"""Compute Absolute Risk Reduction."""

import os
import json
import time
import inspect
from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from hte.data.generation import generate_mixture_oracle

np.random.seed(seed=42)


def compute_beta_arr(
    nb_points : int,
    minval : float,
    maxval : float,
    size : int,
    group : Callable[[np.ndarray], int],
    mean : np.ndarray,
    cov : np.ndarray,
    gamma : np.ndarray,
    model : str,
    base_hazard : str,
    a : float,
    b : float,
    timepoint : float,
    save : bool = False,
    semi_synth : bool = False
) -> dict:
    """For fixed generation parameters, compute the function arr(beta).

    Parameters
    ----------
    nb_points : int
        Number of betas.
    minval : float
        Minimum value of the betas.
    maxval : float
        Maximum values of the betas.
    size : int
        Number of observations for Monte-Calro estimation.
    group: Callable
        Function returning 0 or 1, to define the subgroups.
    mean : np.array
        The mean of the covariates.
    cov : np.array
        The covariance of the covariates
    gamma : np.array
        The prognostic vector in the generation.
    model : str
        Survival analysis model (param of the DGP_survival class).
    base_hazard : str
        Base Hazard of the DGP_survival class.
    a: float
        Parameter of the DGP_survival class.
    b: float
        Parameter of the DGP_survival class.
    timepoint : float
        The time horizon to compute IARR and ARR.
    save: bool
        Whether to save dict of result or not.
    semi_synth: bool
        Whether to use semi-synthetic data or not.

    Returns
    -------
    dict
        Several keys:
        - 'param': the chosen parameters
        - 'prop0': proportion of samples in subgroup 0
        - 'arr(beta0)': a dict with keys given by the betas
        - 'arr(beta1)': a dict with keys given by the betas
    """
    transform_str = inspect.getsource(group)

    dict_result = {}
    dict_result['param'] = {
        'nb_points': nb_points,
        'minval': minval,
        'maxval': maxval,
        'Monte Carlo size': size,
        'mean': mean.tolist(),
        'cov': cov.tolist(),
        'gamma': gamma.tolist(),
        'model': model,
        'base_hazard': base_hazard,
        'a': a,
        'b': b,
        'timepoint': timepoint,
        'group_name': group.__name__,
        'group_str': transform_str
    }

    dict_beta0_arr = {}
    dict_beta1_arr = {}
    betas = np.linspace(minval, maxval, nb_points)

    prop0 = 0
    for beta in tqdm(betas):
        df_oracle = generate_mixture_oracle(
            beta=beta,
            size=size,
            group=group,
            mean=mean,
            cov=cov,
            gamma=gamma,
            model=model,
            base_hazard=base_hazard,
            a=a,
            b=b,
            timepoint=timepoint,
            semi_synth=semi_synth
        )
        mask0 = df_oracle['G'] == 0
        prop0 += mask0.sum() / len(mask0)
        dict_beta0_arr[beta] = df_oracle[mask0]['IARR'].mean()
        dict_beta1_arr[beta] = df_oracle[~mask0]['IARR'].mean()
    dict_result['prop'] = {
        'subgroup0': prop0 / len(betas),
        'subgroup1': 1 - prop0 / len(betas),
    }
    dict_result['arr(beta0)'] = dict_beta0_arr
    dict_result['arr(beta1)'] = dict_beta1_arr

    if save:
        json_obj = json.dumps(dict_result, indent=4)
        date = time.strftime("%B_%m_%d_%Y_%H:%M:%S")
        file_name = f"../data/results_compute_arr/{model}_{base_hazard}_{a}_{b}_" \
                    f"dim={len(mean)}_range=[{minval},{maxval}]_nb={nb_points}_" \
                    f"group='{group.__name__}'_{date}.json"
        if os.path.exists(file_name):
            print('The json already exists')
        else:
            with open(file_name, "w", encoding="utf8") as outfile:
                outfile.write(json_obj)
    return dict_result


def find_beta(
    arr : float,
    dict_param : dict,
) -> float:
    """Find the value beta from given ARR.

    The betas are increasingly sorted.
    Therefore, the ARR is decreasing.

    Parameters
    ----------
    arr : float
        The targeted arr.
    dict_param : dict
        Keys: betas.
        Values: ARR(beta).

    Returns
    -------
    float
        The first beta corresponding to
        an ARR below the target arr.
    """
    # sort everything (really needed?)
    betas = [float(elem) for elem in dict_param.keys()]
    betas = sorted(betas)

    first_key = list(dict_param.keys())[0]
    # if dict has been saved, key is a string
    if isinstance(first_key, str):
        arrs = [dict_param[str(elem)] for elem in betas]
    # otherwise it is a float
    elif isinstance(first_key, float):
        arrs = [dict_param[elem] for elem in betas]

    output = None
    for beta, val in zip(betas, arrs):
        if val < arr:
            output = beta
            break

    return output


def sanity_check_beta_arrs(
    dict_param : dict,
) -> bool:
    """Check if the arrs are sorted.

    Parameters
    ----------
    dict_param : dict
        Keys: betas.
        Values: ARR(beta).

    Returns
    -------
    bool
    """
    # get a list of ordered betas
    sorted_betas = sorted(dict_param)

    # create corresponding list of arrs
    arrs = [
        dict_param[beta] for beta in sorted_betas
    ]

    # the sorted version
    sorted_arrs = sorted(arrs, reverse=True)

    # check equality
    for elem0, elem1 in zip(arrs, sorted_arrs):
        if elem0 == elem1:
            pass
        else:
            return False
    return True


def splicing(
    dict_param : dict,
) -> dict:
    """Splice to smooth arr function, so as to make it decreasing.

    Parameters
    ----------
    dict_param : dict
        Keys: betas.
        Values: ARR(beta).

    Returns
    -------
    dict
        Dictonnary with removed entries.
    """
    # get a list of ordered betas
    sorted_betas = sorted(dict_param)

    # create corresponding list of arrs
    arrs = [
        dict_param[beta] for beta in sorted_betas
    ]

    new_betas = [sorted_betas[0]]
    new_arrs = [arrs[0]]

    for beta, arr in zip(sorted_betas, arrs):
        if arr < new_arrs[-1]:
            new_arrs.append(arr)
            new_betas.append(beta)

    nb_removal = len(arrs) - len(new_arrs)
    print(f'Number of removal: {nb_removal}.')

    dict_res = dict(zip(new_betas, new_arrs))

    return dict_res


if __name__ == '__main__':
    def group(x):
        """Define function for subgroups."""
        return int(x[0] > 0.0)

    output = compute_beta_arr(
        nb_points=50,
        minval=-5.,
        maxval=5.,
        size=int(1e5),
        group=group,
        mean=np.array([0, 0]),
        cov=np.array([[1, 0], [0, 1]]),
        gamma=np.array([0.1, 1.]),
        model='Cox',
        base_hazard='Weibull',
        a=1.0,
        b=2.0,
        timepoint=1.,
        save=False
    )

    print(type(output))
