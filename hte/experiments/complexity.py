"""Run complexity analysis."""
import time
import numpy as np
import pandas as pd
from hte.models import (univariate,
                        multivariate,
                        sides,
                        model_based,
                        interaction_trees,
                        sequential_batting,
                        directed_peeling)
from hte.data.generation import generate_mixture


method_dict = {
    "univariate_test": univariate.Univariate(timepoint=1.,
                                             method='t_test'),
    "univariate_interaction": univariate.Univariate(timepoint=1.,
                                                    method='interaction'),
    "multivariate_cox": multivariate.Multivariate(estimator='CoxLinear',
                                                  interactions=True,
                                                  timepoint=1.),
    "multivariate_tree": multivariate.Multivariate(estimator='SurvivalTree',
                                                   interactions=True,
                                                   timepoint=1.),
    "sides": sides.SIDES(timepoint=1.),
    "model_based": model_based.MOB(timepoint=1.),
    "interaction_trees": interaction_trees.ITree(timepoint=1.),
    "sequential_batting": sequential_batting.SeqBT(timepoint=1.),
    "directed_peeling": directed_peeling.ARDP(timepoint=1.)
}


def group(x):
    """Group function for generate_mixture."""
    cdt1 = x[0] >= -1.
    cdt2 = x[1] >= 1.
    return int(cdt1 & cdt2)


def complexity(dims : list[int],
               nb_samples : list[int],
               repets : int,
               methods : list[str]):
    """Run complexity analysis."""
    times = []

    for dim in dims:
        mean = np.zeros(dim)
        cov = np.diag(np.ones(dim))
        for n in nb_samples:
            for method in methods:
                obj = method_dict[method]
                start = time.time()
                for _ in range(0, repets):
                    gamma = np.random.normal(size=dim)
                    data = generate_mixture(beta0=1.,
                                            beta1=-1.,
                                            size=n,
                                            group=group,
                                            mean=mean,
                                            cov=cov,
                                            gamma=gamma)

                    _data = data.drop(columns=['G']).copy()
                    obj.fit(data=_data,
                            duration_col='T',
                            event_col='E',
                            trt_col='W')
                end = time.time()

                new_time = {'dim': dim,
                            'nb_samples': n,
                            'method': method,
                            'time': end - start}
                times.append(new_time)
    return pd.DataFrame(times)


if __name__ == '__main__':

    np.random.seed(seed=40)

    dim = [3, 20, 100, 500, 1000]
    nb_samples = [500]
    repet = 10  # pylint: disable=C0103

    methods = ['univariate_test',
               'univariate_interaction',
               'multivariate_cox',
               'multivariate_tree',
               'sides',
               'model_based',
               'interaction_trees',
               'sequential_batting',
               'directed_peeling']

    times = complexity(dim, nb_samples, repet, methods)

    date = time.strftime("%B_%m_%d_%Y_%H:%M:%S")
    times.to_csv(
        f"results_expe/COMPLEXITY_DIM={dim}_NB={nb_samples}"
        f"_REPET={repet}_{date}.csv"
    )
