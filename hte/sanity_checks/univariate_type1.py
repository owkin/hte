"""Inspect type I error of Univariate method."""

# %%
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from hte.models import univariate
from hte.data.generation import generate_mixture
from hte.configs.data_configs.subgroups import (tutorial,
                                                dim20_pred4_prog0_balanced,
                                                dim20_pred4_prog4_balanced,
                                                dim20_pred4_prog4_bis_balanced,
                                                dim20_pred4_prog2_balanced,
                                                dim100_pred3_prog2_balanced,
                                                dim100_pred4_prog4_balanced,
                                                dim100_pred4_prog0_balanced,
                                                dim100_pred4_prog2_balanced,
                                                dim1000_pred4_prog0_balanced)


np.random.seed(seed=42)

sbg_dict = {
    "tutorial": tutorial,
    "dim20_pred4_prog0_balanced": dim20_pred4_prog0_balanced,
    "dim20_pred4_prog4_balanced": dim20_pred4_prog4_balanced,
    "dim20_pred4_prog4_bis_balanced": dim20_pred4_prog4_bis_balanced,
    "dim20_pred4_prog2_balanced": dim20_pred4_prog2_balanced,
    "dim100_pred3_prog2_balanced": dim100_pred3_prog2_balanced,
    "dim100_pred4_prog4_balanced": dim100_pred4_prog4_balanced,
    "dim100_pred4_prog0_balanced": dim100_pred4_prog0_balanced,
    "dim100_pred4_prog2_balanced": dim100_pred4_prog2_balanced,
    "dim1000_pred4_prog0_balanced": dim1000_pred4_prog0_balanced
}

dict_param_path = "../data/results_compute_arr/Cox_Weibull_1.0_2.0_dim=20_\
    range=[-10.0,10.0]_nb=500_group='dim20_pred4_prog0_balanced'_\
        July_07_12_2023_15:15:24.json"

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
    timepoint = dict_dgp_param['timepoint']
    minval = dict_dgp_param['minval']
    maxval = dict_dgp_param['maxval']
    nb_points = dict_dgp_param['nb_points']
    group_name = dict_dgp_param['group_name']
    group = sbg_dict[group_name]

    prop = dict_param['prop']['subgroup1']

rngs0 = np.random.randint(
    np.iinfo(np.int32).max,
    size=1000
)
rngs1 = np.random.randint(
    np.iinfo(np.int32).max,
    size=1000
)
random_states = [[r0, r1] for r0, r1 in zip(rngs0, rngs1)]


# %%
results = []
for _, random_state in tqdm(zip(range(100), random_states)):
    df_train = generate_mixture(
        beta0=0.,
        beta1=0.,
        size=1000,
        group=group,
        mean=mean,
        cov=cov,
        gamma=gamma,
        model=model,
        base_hazard=base_hazard,
        a=a,
        b=b,
        random_state=random_state[0],
        censored=False,
        scale=0.1
    )

    df_test = generate_mixture(
        beta0=0.,
        beta1=0.,
        size=1000,
        group=group,
        mean=mean,
        cov=cov,
        gamma=gamma,
        model=model,
        base_hazard=base_hazard,
        a=a,
        b=b,
        random_state=random_state[1],
        censored=False,
        scale=0.1
    )

    df_all = pd.concat(
        [df_train, df_test],
        axis=0
    )

    obj1 = univariate.Univariate(timepoint=1., method='interaction')
    obj2 = univariate.Univariate(timepoint=1., method='t_test')

    obj1.fit(df_all, 'T', 'E', 'W')
    obj2.fit(df_all, 'T', 'E', 'W')

    pval1, pval_df1 = obj1.pval_hte(method='bonferroni')
    pval2, pval_df2 = obj2.pval_hte(method='bonferroni')

    results.append([pval1, pval2])

# %%
results = np.clip(np.array(results), 0., 1.)
dict_to_plot = {
    'Interaction': results[:, 0],
    'T-test': results[:, 1]
}
plt.figure(figsize=(5, 4))
plt.boxplot(dict_to_plot.values(), labels=dict_to_plot.keys())
plt.show()
# %%
