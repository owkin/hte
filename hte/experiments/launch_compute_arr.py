"""Launch of ARR & Beta computation."""
import argparse
import json
import numpy as np
from hte.data.compute_arr import compute_beta_arr
from hte.configs.data_configs.subgroups import (tutorial,  # noqa: F401, E501 # pylint: disable=W0611
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
                                                dim1000_pred4_prog0_balanced)

from hte.configs.data_configs.subgroups import sbg_dict

np.random.seed(seed=42)


if __name__ == '__main__':
    # example of CLI:
    # python launch_compute_arr.py -f="../configs/data_configs/tutorial.json"
    # -g="tutorial" -l=-3. -u=4. -n=10
    parser = argparse.ArgumentParser(
        description="Launch an experiment computing arrs from a list of betas."
    )
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        help="Path of file containing covariates structure."
    )
    parser.add_argument(
        '-g',
        '--group',
        type=str,
        help="Name of the subgroup function (in subgroups.py)."
    )
    parser.add_argument(
        '-l',
        '--minval',
        type=float,
        help="Minimum value of the betas."
    )
    parser.add_argument(
        '-u',
        '--maxval',
        type=float,
        help="Maximum value of the betas."
    )
    parser.add_argument(
        '-n',
        '--nbpoints',
        type=int,
        help="Number of betas to look at between minval and maxval."
    )
    parser.add_argument(
        '-mc',
        '--montecarlo',
        default=int(1e6),
        type=int,
        help="Monte Carlo sampling size."
    )
    parser.add_argument(
        '-m',
        '--model',
        default='Cox',
        type=str,
        help="Name of the time-to-event model (Cox, AH or AFT)"
    )
    parser.add_argument(
        '-ha',
        '--hazard',
        default='Weibull',
        type=str,
        help="Name of the baseline hazard (Weibull or LogNormal)"
    )
    parser.add_argument(
        '-a',
        '--a',
        default=1.,
        type=float,
        help="First parameter of the baseline hazard."
    )
    parser.add_argument(
        '-b',
        '--b',
        default=2.,
        type=float,
        help="Second parameter (shape) of the baseline hazard."
    )
    parser.add_argument(
        '-tp',
        '--timepoint',
        default=1.,
        type=float,
        help="The timepoint where ARR is computed."
    )

    parser.add_argument(
        '-ss',
        '--semi_synth',
        default=False,
        type=bool,
        help="Whether to use semi-synthetic data or not."
    )

    args = parser.parse_args()

    param_file = args.file
    group_str = args.group
    nb_points = int(args.nbpoints)
    minval = float(args.minval)
    maxval = float(args.maxval)
    size = args.montecarlo
    model = args.model
    base_hazard = args.hazard
    a = float(args.a)
    b = float(args.b)
    timepoint = float(args.timepoint)
    semi_synth = args.semi_synth

    with open(param_file, "r", encoding="utf8") as json_file:
        dict_params = json.load(json_file)

    group = sbg_dict[group_str]

    mean = np.array(dict_params['mean'])
    cov = np.array(dict_params['cov'])
    gamma = np.array(dict_params['gamma'])

    compute_beta_arr(
        nb_points=nb_points,
        minval=minval,
        maxval=maxval,
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
        save=True,
        semi_synth=semi_synth
    )
