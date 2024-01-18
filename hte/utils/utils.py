"""Secondary functions for experiments."""
from bisect import bisect
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import norm
import matplotlib.pyplot as plt


class KMPLOTS():
    """Plot KM curves of our dgps."""

    def __init__(self):
        pass

    def plot(self, df : pd.DataFrame, title : str):
        """Plot KM curves."""
        plt.figure()
        plt.title(title)

        mask = df['W'] == 1

        kf_treated_obs = KaplanMeierFitter()
        kf_treated_obs.fit(
            durations=df.loc[mask]['Times'],
            event_observed=df.loc[mask]['Event'],
            label='Treated (observed)'
        )

        kf_untreated_obs = KaplanMeierFitter()
        kf_untreated_obs.fit(
            durations=df.loc[~mask]['Times'],
            event_observed=df.loc[~mask]['Event'],
            label='Placebo (observed)'
        )

        kf_treated_obs.plot_survival_function(
            ci_show=False,
            show_censors=True
        )
        kf_untreated_obs.plot_survival_function(
            ci_show=False,
            show_censors=True
        )

        plt.show()

    def plot_th(self, df_oracle : pd.DataFrame, title : str):
        """Plot function with th."""
        plt.figure()
        plt.title(title)

        kf_treated = KaplanMeierFitter()
        kf_treated.fit(
            durations=df_oracle['times1'],
            event_observed=[1] * len(df_oracle),
            label='Treated (oracle)'
        )

        kf_untreated = KaplanMeierFitter()
        kf_untreated.fit(
            durations=df_oracle['times0'],
            event_observed=[1] * len(df_oracle),
            label='Placebo (oracle)'
        )

        kf_treated.plot_survival_function(
            ci_show=False,
            show_censors=True
        )
        kf_untreated.plot_survival_function(
            ci_show=False,
            show_censors=True
        )

        plt.show()


def ztest(stat_mean : float, stat_std : float, value : float = 0.):
    """Compute the two-sided p-value of a z-test."""
    return 2 * (1 - norm.cdf(np.abs(stat_mean - value) / stat_std))


def top_n_variables(ranking : np.array, n : int = 10):
    """Provide the first n variables in ranked order."""
    if n <= len(ranking):
        return ranking[0:n]
    return ranking


def survival(df : pd.DataFrame,
             duration_col : str,
             event_col : str,
             timepoint : float = 1.):
    """Return mean and variance of the percent survival at time *timepoint*."""
    km_model = KaplanMeierFitter().fit(df[duration_col],
                                       df[event_col]).confidence_interval_
    idx = km_model.index[bisect(km_model.index.values, timepoint) - 1]
    low_ = km_model.at[idx, 'KM_estimate_lower_0.95']
    up_ = km_model.at[idx, 'KM_estimate_upper_0.95']
    return (up_ + low_) / 2, ((up_ - low_) / (2 * norm.ppf(0.975))) ** 2


def survival_did_test(df : pd.DataFrame,
                      gates_col : str,
                      treatment_col : str,
                      duration_col : str,
                      event_col : str,
                      timepoint : float = 1.):
    """Perform survival analysis of treatment between two subgroups (0,1)."""
    # add a cdt for df of size <= 1.

    mask1 = (df[gates_col] == 0) & (df[treatment_col] == 0)
    mask2 = (df[gates_col] == 0) & (df[treatment_col] == 1)
    mask3 = (df[gates_col] == 1) & (df[treatment_col] == 0)
    mask4 = (df[gates_col] == 1) & (df[treatment_col] == 1)

    if mask1.sum() <= 1:
        return np.nan
    if mask2.sum() <= 1:
        return np.nan
    if mask3.sum() <= 1:
        return np.nan
    if mask4.sum() <= 1:
        return np.nan

    surv_g0_t0, var_g0_t0 = survival(
        df[(df[gates_col] == 0) & (df[treatment_col] == 0)],
        duration_col, event_col, timepoint
    )
    surv_g0_t1, var_g0_t1 = survival(
        df[(df[gates_col] == 0) & (df[treatment_col] == 1)],
        duration_col, event_col, timepoint
    )
    surv_g1_t0, var_g1_t0 = survival(
        df[(df[gates_col] == 1) & (df[treatment_col] == 0)],
        duration_col, event_col, timepoint
    )
    surv_g1_t1, var_g1_t1 = survival(
        df[(df[gates_col] == 1) & (df[treatment_col] == 1)],
        duration_col, event_col, timepoint
    )
    var = var_g0_t0 + var_g0_t1 + var_g1_t0 + var_g1_t1
    did = (surv_g1_t1 - surv_g1_t0) - (surv_g0_t1 - surv_g0_t0)

    if (did == 0.) and (var == 0.):
        return 1.
    return ztest(did, np.sqrt(var))


def survival_te_diff(df : pd.DataFrame,
                     treatment_col : str,
                     duration_col : str,
                     event_col : str,
                     var_split : str,
                     val_split : float,
                     timepoint : float = 1.):
    """Compute treatment effect between two subgroups defined by split."""
    mask1 = (df[var_split] <= val_split) & (df[treatment_col] == 0)
    mask2 = (df[var_split] <= val_split) & (df[treatment_col] == 1)
    mask3 = (df[var_split] > val_split) & (df[treatment_col] == 0)
    mask4 = (df[var_split] > val_split) & (df[treatment_col] == 1)

    if mask1.sum() <= 1:
        return np.nan
    if mask2.sum() <= 1:
        return np.nan
    if mask3.sum() <= 1:
        return np.nan
    if mask4.sum() <= 1:
        return np.nan

    subgroup0_control = df[(df[var_split] <= val_split) & (df[treatment_col] == 0)]
    subgroup0_treat = df[(df[var_split] <= val_split) & (df[treatment_col] == 1)]
    subgroup1_control = df[(df[var_split] > val_split) & (df[treatment_col] == 0)]
    subgroup1_treat = df[(df[var_split] > val_split) & (df[treatment_col] == 1)]

    surv_g0_t0, var_g0_t0 = survival(subgroup0_control,
                                     duration_col,
                                     event_col,
                                     timepoint)
    surv_g0_t1, var_g0_t1 = survival(subgroup0_treat,
                                     duration_col,
                                     event_col,
                                     timepoint)
    surv_g1_t0, var_g1_t0 = survival(subgroup1_control,
                                     duration_col,
                                     event_col,
                                     timepoint)
    surv_g1_t1, var_g1_t1 = survival(subgroup1_treat,
                                     duration_col,
                                     event_col,
                                     timepoint)

    var = var_g0_t0 + var_g0_t1 + var_g1_t0 + var_g1_t1
    did = (surv_g1_t1 - surv_g1_t0) - (surv_g0_t1 - surv_g0_t0)
    return ztest(did, np.sqrt(var))


def survival_te(df : pd.DataFrame,
                treatment_col : str,
                duration_col : str,
                event_col : str,
                timepoint : float = 1.):
    """Compute treatment effect in one subgroup."""
    mask1 = df[treatment_col] == 0
    mask2 = df[treatment_col] == 1

    if mask1.sum() <= 1:
        return np.nan, 0
    if mask2.sum() <= 1:
        return np.nan, 0

    group_control = df[(df[treatment_col] == 0)]
    group_treat = df[(df[treatment_col] == 1)]

    surv_t0, var_t0 = survival(group_control,
                               duration_col,
                               event_col,
                               timepoint)
    surv_t1, var_t1 = survival(group_treat,
                               duration_col,
                               event_col,
                               timepoint)

    var = var_t0 + var_t1
    did = surv_t1 - surv_t0
    trt_benefit = int(did > 0)
    return ztest(did, np.sqrt(var)) / 2, trt_benefit
