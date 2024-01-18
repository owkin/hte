"""Launch of experiments."""
import argparse
import json
import numpy as np
from hte.experiments.run_experiments import power_analysis


def find_upper_bound(dict_param):
    """Find the upper bound of the arrs.

    Parameters
    ----------
    dict_param : dict
        The dictionary containing the parameters of the experiment.

    Returns
    -------
    arr0 : float

    arr1 : float
    """
    dict_param0 = dict_param['arr(beta0)']
    dict_param1 = dict_param['arr(beta1)']
    prop = dict_param['prop']['subgroup1']
    min_val_g0 = min(list(dict_param0.values()))
    for _, v in dict_param1.items():
        arr1 = v
        arr0 = (-1) * (prop / (1 - prop)) * arr1
        if min_val_g0 <= arr0:
            return arr1, arr0
    return None, None


if __name__ == '__main__':
    # example of CLI:
    # python launch_expe.py -f="../data/results_compute_arr/Cox_Weibull_1.0_2.0_\
    # dim=3_range=[-6.0,6.0]_nb=500_group='tutorial'_July_07_12_2023_10:08:51"
    # -g="tutorial" -l=-3. -u=4. -n=10
    parser = argparse.ArgumentParser(
        description="Launch an experiment for several arrs and methods."
    )
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        help="Path of file containing compute_arr results."
    )
    # parser.add_argument(
    #     '-l',
    #     '--minval',
    #     type=float,
    #     help="Min value of the arrs."
    # )
    # parser.add_argument(
    #     '-u',
    #     '--maxval',
    #     type=float,
    #     help="Max value of the arrs."
    # )
    parser.add_argument(
        '-n',
        '--nbpoints',
        type=int,
        help="Number of arrs."
    )
    parser.add_argument(
        '-s',
        '--samplesize',
        type=int,
        help="Size of the dataset."
    )
    parser.add_argument(
        '-tr',
        '--trainsize',
        type=float,
        help="Proportion of the training set."
    )
    parser.add_argument(
        '-r',
        '--repet',
        type=int,
        help="Number of repetitions per arr."
    )
    parser.add_argument(
        '-c',
        '--censored',
        default='False',
        type=str,
        help="Whether censorship occurs or not."
    )
    parser.add_argument(
        '-sc',
        '--scale',
        default=1.0,
        type=float,
        help="Controls intensity of censorship."
    )

    parser.add_argument(
        '-ss',
        '--semi_synth',
        default=False,
        type=bool,
        help="Whether semi-synthetic data is used or not."
    )

    parser.add_argument(
        '-m',
        '--methods',
        default=("Oracle, Univariate interaction, Univariate t_test, "
                 "Multivariate cox, Multivariate tree, MOB, ITree, "
                 "SIDES, SeqBT, ARDP"),
        type=str,
        help="List of methods to use."
    )

    args = parser.parse_args()

    dict_param_path = args.file
    with open(dict_param_path, "r", encoding="utf8") as json_file:
        dict_param = json.load(json_file)
    upper, _ = find_upper_bound(dict_param)
    minval = float(0.)
    maxval = float(upper)

    nb_points = int(args.nbpoints)
    sample_size = int(args.samplesize)
    train_prop = float(args.trainsize)
    train_size = int(sample_size * train_prop)
    test_size = sample_size - train_size
    # test_size = int(args.testsize)
    repet = int(args.repet)
    if args.censored == 'False':
        CENSORED = False
    if args.censored == 'True':
        CENSORED = True
    scale = float(args.scale)

    semi_synth = args.semi_synth

    method_list = args.methods
    # print(censored)

    print(f"path: {type(dict_param_path)}, {dict_param_path}")
    print(f"nb_points: {type(nb_points)}, {nb_points}")
    print(f"minval: {type(minval)}, {minval}")
    print(f"maxval: {type(maxval)}, {maxval}")
    print(f"sample_size: {type(sample_size)}, {sample_size}")
    print(f"train_prop: {type(train_prop)}, {train_prop}")
    print(f"train_size: {type(train_size)}, {train_size}")
    print(f"test_size: {type(test_size)}, {test_size}")
    print(f"repet: {type(repet)}, {repet}")
    print(f"censored: {type(CENSORED)}, {CENSORED}")
    print(f"scale: {type(scale)}, {scale}")
    print(f"semi_synth: {type(semi_synth)}, {semi_synth}")

    arrs = np.linspace(
        minval,
        maxval,
        nb_points
    )
    print(f"arr: {arrs}")

    with open(dict_param_path, "r", encoding="utf8") as json_file:
        dict_param = json.load(json_file)
    values0 = list(dict_param["arr(beta0)"].values())
    values1 = list(dict_param["arr(beta1)"].values())
    min_0, min_1 = min(values0), min(values1)
    max_0, max_1 = max(values0), max(values1)
    prop = dict_param['prop']['subgroup1']

    if maxval > max_1:
        raise Exception(  # pylint: disable=W0719
            f"maxval value is above max value of arr(beta1) dict: "
            f"{maxval} > {max_1}."
        )
    if (-1) * (prop / (1 - prop)) * maxval < min_0:
        raise Exception(  # pylint: disable=W0719
            f"(-1) * prop1/(1-prop1) * maxval value is below min value"
            f"of arr(beta0) dict: {(-1) * (prop / (1 - prop)) * maxval} < {min_0}."
        )

    np.random.seed(seed=42)

    dict_expe = power_analysis(
        arrs=arrs,
        dict_param_path=dict_param_path,
        train_size=train_size,
        test_size=test_size,
        repet=repet,
        censored=CENSORED,
        scale=scale,
        semi_synth=semi_synth,
        methods=method_list
    )
