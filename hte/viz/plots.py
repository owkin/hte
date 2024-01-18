"""Plots helper fot KM curves."""

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


class KaplanMeierPlots():
    """Helper to plot KM curves of our dgps."""

    def __init__(self):
        pass

    def plot(
        self,
        df : pd.DataFrame,
        title : str
    ):
        """Plot a KM curve from df.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe with time-to-event data.
        title: str
            The title of the plot
        """
        plt.figure()
        plt.title(title)

        mask = df['W'] == 1

        kf_treated_obs = KaplanMeierFitter()
        kf_treated_obs.fit(
            durations=df.loc[mask]['T'],
            event_observed=df.loc[mask]['E'],
            label='Treated (observed)'
        )

        kf_untreated_obs = KaplanMeierFitter()
        kf_untreated_obs.fit(
            durations=df.loc[~mask]['T'],
            event_observed=df.loc[~mask]['E'],
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

        plt.xlim(0., 1.1)

        plt.show()

    def plot_th(
        self,
        df_oracle,
        title
    ):
        """Plot a KM curve from df_oracle.

        Parameters
        ----------
        df_oracle : pd.DataFrame
            The dataframe with time-to-event data.
        title: str
            The title of the plot
        """
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
