"""Run experiments pipeline for the benchmark paper."""

from collections.abc import Callable

import json
import time
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd

import joblib
from joblib import Parallel, delayed

from lifelines.exceptions import ConvergenceError

from analyze_expe import ProcessResults

from hte.data.generation import generate_mixture
from hte.data.compute_arr import find_beta
from hte.utils.utils import top_n_variables
from hte.models import (univariate,  # noqa: F401 # pylint: disable=W0611
                        multivariate,
                        sides,
                        model_based,
                        interaction_trees,
                        sequential_batting,
                        directed_peeling)
from hte.metrics.metrics import classification_metrics
from hte.experiments.attributes import attributes
from hte.configs.data_configs.subgroups import (  # noqa: F401 # pylint: disable=W0611
    tutorial,
    dim20_pred4_prog0_balanced,
    dim20_pred4_prog1_balanced,
    dim20_pred4_prog3_balanced,
    dim20_pred4_prog4_balanced,
    dim20_pred4_prog4_bis_balanced,
    dim20_pred4_prog2_balanced,
    dim100_pred3_prog2_balanced,
    dim100_pred4_prog4_balanced,
    dim100_pred4_prog0_balanced,
    dim100_pred4_prog2_balanced,
    dim1000_pred4_prog0_balanced
)
from hte.configs.data_configs.subgroups import sbg_dict

np.random.seed(seed=42)


def power_analysis(
    arrs: np.ndarray,
    dict_param_path: str,
    train_size: int,
    test_size: int,
    repet: int,
    censored: bool,
    scale: float,
    methods: str,
    semi_synth: bool = False
) -> pd.DataFrame:
    """Run single expe for several betas.

    Function that calls single_expe multiple times by making beta0 and
    beta1 varying according to a increasing ARR in subgroup0.
    All parameters are taken from a computed arr(beta)
    The function iterates on all methods in methods folder.

    Parameters
    ----------
    arrs : np.ndarray
        The values of the arr in subgroup0, on which experiments will be conducted.
    dict_param_path : str
        The path where parameters and results of compute_arr have been saved.
    train_size : int
        The size of the train set.
    test_size : int
        The size of the test set.
    repet : int
        The number of repetition for each single_experiment
    censored : bool
        Whether generated data are censored.
    scale : float
        The scale of the censorship (cf generation)
    methods : str
        The methods to benchmark in the experiment.
    semi_synth : bool
        Whether to use semi-synthetic data or not.

    Returns
    -------
    pd.DataFrame
        The results organized in a dataframe.
        Everything is save in csv.
    """
    with open(dict_param_path, "r", encoding="utf8") as json_file:
        dict_param = json.load(json_file)

    dict_dgp_param = dict_param['param']

    model = dict_dgp_param['model']
    base_hazard = dict_dgp_param['base_hazard']
    a = dict_dgp_param['a']
    b = dict_dgp_param['b']
    mean = dict_dgp_param['mean']
    cov = dict_dgp_param['cov']
    gamma = dict_dgp_param['gamma']
    minval = dict_dgp_param['minval']
    maxval = dict_dgp_param['maxval']
    nb_points = dict_dgp_param['nb_points']
    group_name = dict_dgp_param['group_name']
    group = sbg_dict[group_name]

    prop = dict_param['prop']['subgroup1']

    list_of_results = []

    betas0 = []
    betas1 = []
    for arr in arrs:
        if arr == 0.:
            betas0.append(0.)
            betas1.append(0.)
        else:
            arr1 = arr
            arr0 = (-1) * (prop / (1 - prop)) * arr1
            beta1 = find_beta(
                arr=arr1,
                dict_param=dict_param["arr(beta1)"]
            )
            betas1.append(float(beta1))
            beta0 = find_beta(
                arr=arr0,
                dict_param=dict_param["arr(beta0)"]
            )
            betas0.append(float(beta0))

    n_jobs = joblib.cpu_count()
    print(f'Number CPUs: {n_jobs}')

    for beta0, beta1, arr in tqdm(zip(betas0, betas1, arrs)):
        results = single_expe(
            beta0=beta0,
            beta1=beta1,
            train_size=train_size,
            test_size=test_size,
            repet=repet,
            mean=mean,
            cov=cov,
            gamma=gamma,
            censored=censored,
            scale=scale,
            group=group,
            model=model,
            base_hazard=base_hazard,
            a=a,
            b=b,
            methods=methods,
            semi_synth=semi_synth
        )
        for res in results:
            for dd in res:
                dd['arr'] = arr
        list_of_results.extend(results)
    flattened_results = [item for sublist in list_of_results for item in sublist]
    df_power_results = pd.DataFrame(flattened_results)
    print(df_power_results.head())
    print(df_power_results.columns)
    df_power_results['file_arr(beta)'] = dict_param_path

    date = time.strftime("%B_%m_%d_%Y_%H:%M:%S")
    df_power_results.to_csv(
        f"./results_expe/raw_results/{model}_{base_hazard}_{a}_{b}_"
        f"dim={len(mean)}_range=[{minval},{maxval}]_nb={nb_points}_"
        f"group='{group.__name__}_rangeARR=[{min(arrs)}, {max(arrs)}]_"
        f"nb={len(arrs)}_train={train_size}_test={test_size}_"
        f"repet={repet}_censored={censored}_scale={scale}_{date}.csv"
    )

    # Process results
    process_results = ProcessResults(pdim=len(mean), subgroup=group_name)
    df_processed_results = process_results.process(df_power_results)
    df_processed_results.to_csv(
        f"./results_expe/processed_results/{model}_{base_hazard}_{a}_{b}_"
        f"dim={len(mean)}_range=[{minval},{maxval}]_nb={nb_points}_"
        f"group='{group.__name__}_rangeARR=[{min(arrs)}, {max(arrs)}]_"
        f"nb={len(arrs)}_train={train_size}_test={test_size}_"
        f"repet={repet}_censored={censored}_scale={scale}_{date}.csv"
    )

    return df_power_results


def single_expe(  # pylint: disable=R0912
    beta0: float,
    beta1: float,
    train_size: int,
    test_size: int,
    repet: int,
    mean: np.array,
    cov: np.array,
    gamma: np.array,
    censored: bool,
    scale: float,
    group : Callable = None,
    model: str = 'Cox',
    base_hazard: str = 'Weibull',
    a: float = 1.0,
    b: float = 2.0,
    methods: str = None,
    semi_synth : bool = False
) -> tuple[dict, pd.DataFrame]:
    """Experiment performed at a fixed arr for subgroup0.

    Parameters
    ----------
    beta0 : float
        The coeff of T in subgroup 0.
    beta1 : float
        The coeff of T in subgroup 1.
    train_size : int
        The size of the train set.
    test_size : int
        The size of the test set.
    repet: int
        The number of resampling.
    mean : np.array
        The mean of the covariates.
    cov : np.array
        The covariance of the covariates.
    gamma : np.array
        Coefficients of the covariates.
    censored : bool
        Whether generated data are censored.
    scale : float
        The scale of the censorship (cf generation)
    group: Callable
        The function defining the subgroup
    model : str
        Survival analysis model (param of the DataSurvival class).
    base_hazard : str
        Base Hazard of the DataSurvival class.
    a : float
        Parameter of the DataSurvival class.
    b : float
        Parameter of the DataSurvival class.
    methods : str
        The methods to benchmark in the experiment.
    semi_synth : bool
        Whether to use semi-synthetic data or not.

    Returns
    -------
    list
        The results of the experiment for one ARR point.
    """
    n_jobs = joblib.cpu_count()
    # n_jobs = 1
    # print(n_jobs)

    # create array of random states of length repet
    # random_state = np.random.randint(np.iinfo(np.int32).max, size=n_vectors)
    # add random_state as argument of single_expe
    # modify generate functions accordingly
    # il faudra sans doute utliser :
    # rng = numpy.random.default_rng(seed=SEED)
    # rng.multivariate_normal(size=SIZE)
    # En résumé, pour chaque ARR,
    # 1) On génère des SEED (100)
    # 2) On utilise ces seed dans single_repet

    rngs0 = np.random.randint(
        np.iinfo(np.int32).max,
        size=repet
    )
    rngs1 = np.random.randint(
        np.iinfo(np.int32).max,
        size=repet
    )
    rngs2 = np.random.randint(
        np.iinfo(np.int32).max,
        size=repet
    )
    rngs3 = np.random.randint(
        np.iinfo(np.int32).max,
        size=repet
    )
    rngs4 = np.random.randint(
        np.iinfo(np.int32).max,
        size=repet
    )
    random_states = [[r0, r1, r2, r3, r4] for r0, r1, r2, r3, r4
                     in zip(rngs0, rngs1, rngs2, rngs3, rngs4)]

    list_dict_res = Parallel(n_jobs=n_jobs)(
        delayed(single_repet)(
            beta0=beta0,
            beta1=beta1,
            train_size=train_size,
            test_size=test_size,
            repet=repet,
            mean=mean,
            cov=cov,
            gamma=gamma,
            group=group,
            model=model,
            base_hazard=base_hazard,
            a=a,
            b=b,
            random_state=rng,
            censored=censored,
            methods=methods,
            scale=scale,
            semi_synth=semi_synth,
        ) for repet, rng in tqdm(zip(range(repet), random_states)))

    return list_dict_res


def single_repet(  # pylint: disable=R0915, W0102
    beta0: float,
    beta1: float,
    train_size: int,
    test_size: int,
    repet: int,
    mean: np.ndarray,
    cov: np.ndarray,
    gamma: np.ndarray,
    group : Callable,
    methods : str,
    model: str = 'Cox',
    base_hazard: str = 'Weibull',
    a: float = 1.0,
    b: float = 2.0,
    random_state: list = [42, 42, 42, 42, 42],
    censored: bool = False,
    scale: float = 1.0,
    semi_synth : bool = False
) -> tuple[dict, pd.DataFrame]:
    """Single point computation.

    Parameters
    ----------
    beta0 : float
        The coeff of T in subgroup 0.
    beta1 : float
        The coeff of T in subgroup 1.
    train_size : int
        The size of the train set.
    test_size : int
        The size of the test set.
    repet: int
        The number of resampling.
    mean : np.array
        The mean of the covariates.
    cov : np.array
        The covariance of the covariates.
    gamma : np.array
        Coefficients of the covariates.
    group: Callable
        The function defining the subgroup
    model : str
        Survival analysis model (param of the DataSurvival class).
    base_hazard : str
        Base Hazard of the DataSurvival class.
    a : float
        Parameter of the DataSurvival class.
    b : float
        Parameter of the DataSurvival class.
    random_state : list
        List of seeds for reproducibility.
    censored : bool
        Whether generated data are censored.
    scale : float
        The scale of the censorship (cf generation)
    methods : str
        The methods to benchmark in the experiment.
    semi_synth : bool
        Whether to use semi-synthetic data or not.

    Returns
    -------
    list
        List of results for each method of the experiment.
    """
    df_train = generate_mixture(
        beta0=beta0,
        beta1=beta1,
        size=train_size,
        group=group,
        mean=mean,
        cov=cov,
        gamma=gamma,
        model=model,
        base_hazard=base_hazard,
        a=a,
        b=b,
        random_state=random_state[0],
        censored=censored,
        scale=scale,
        semi_synth=semi_synth
    )

    df_test = generate_mixture(
        beta0=beta0,
        beta1=beta1,
        size=test_size,
        group=group,
        mean=mean,
        cov=cov,
        gamma=gamma,
        model=model,
        base_hazard=base_hazard,
        a=a,
        b=b,
        random_state=random_state[1],
        censored=censored,
        scale=scale,
        semi_synth=semi_synth
    )

    df_val = generate_mixture(
        beta0=beta0,
        beta1=beta1,
        size=1000,
        group=group,
        mean=mean,
        cov=cov,
        gamma=gamma,
        model=model,
        base_hazard=base_hazard,
        a=a,
        b=b,
        random_state=random_state[2],
        censored=censored,
        scale=scale,
        semi_synth=semi_synth
    )

    # keep subgroup ground truth
    true_groups_train = df_train['G']
    true_groups_test = df_test['G']
    true_groups_val = df_val['G']
    oracle = pd.concat(
        [true_groups_train, true_groups_test],
        axis=0
    )
    oracle.reset_index(drop=True, inplace=True)

    # remove subgroup column for train, test and validation sets
    df_train = df_train.drop('G', axis=1)
    df_test = df_test.drop('G', axis=1)
    df_val = df_val.drop('G', axis=1)

    # concatenate for methods with one single set
    df_all = pd.concat(
        [df_train, df_test],
        axis=0
    )
    df_all.reset_index(drop=True, inplace=True)

    # create oracle df
    df_oracle = df_all[['T', 'E', 'W']].copy()
    df_oracle['G'] = oracle.array

    # create list of methods to iterate over
    methods = methods.split(', ')

    random_states_methods = {
        'Multivariate tree': random_state[3],
        'ARDP': random_state[4]
    }

    list_of_res = []
    for method in methods:  # pylint: disable=R1702
        # print(method)
        dict_res = {
            'repet' : repet,
            'trainsize' : train_size,
            'testsize' : test_size,
            'beta0' : beta0,
            'beta1' : beta1,
            'mean' : mean,
            'cov' : cov,
            'model' : model,
            'base_hazard': base_hazard,
            'a' : a,
            'b' : b,
            'method' : method
        }

        method_model = getattr(
            sys.modules[attributes[method]['module_path']],
            attributes[method]['module_class'])(**attributes[method]['config'])
        if method in random_states_methods:
            method_model.random_state = random_states_methods[method]

        # Fit
        try:
            # Answer Research Questions
            # Q1
            if method == 'Oracle':
                method_model.fit(df_oracle, 'T', 'E', 'W', ['G'])
                dict_res['pval'] = method_model.pval_hte()
            else:
                if attributes[method]['predictive_method']:
                    method_model.fit(df_train, 'T', 'E', 'W')
                    dict_res['pval'] = method_model.pval_hte(df_test)
                else:
                    method_model.fit(df_all, 'T', 'E', 'W')
                    dict_res['pval'] = method_model.pval_hte()
                    if hasattr(method_model, 'saving_pvalues') and (len(mean) == 20):
                        # Add individual pvalues for each variable for univariate
                        _, pval_df = method_model.pval_hte(
                            save_pvals=attributes[method]['config']['saving_pvalues']
                        )
                        for i in pval_df.columns:
                            dict_res[f"pval_{i}"] = pval_df[i].iloc[0]
                # Q2
                dict_res['ranked_var'] = top_n_variables(method_model
                                                         .variables_ranking())

                # Q3
                predicted_subgroups = method_model.predict(df_val)

                acc, tp, tn, fp, fn = classification_metrics(
                    true_groups_val,
                    predicted_subgroups
                )
                dict_res['Accuracy'] = acc
                dict_res['TP'] = tp
                dict_res['TN'] = tn
                dict_res['FP'] = fp
                dict_res['FN'] = fn

        except ConvergenceError as cerr:
            print(f"CONVERGENCE ERROR in {method}: {str(cerr)}")
            # Q1
            dict_res['pval'] = 'NOT FITTED'
            # Q2
            dict_res['ranked_var'] = 'NOT FITTED'
            # Q3
            dict_res['Accuracy'] = 'NOT FITTED'
            dict_res['TP'] = 'NOT FITTED'
            dict_res['TN'] = 'NOT FITTED'
            dict_res['FP'] = 'NOT FITTED'
            dict_res['FN'] = 'NOT FITTED'

        list_of_res.append(dict_res)

    return list_of_res
