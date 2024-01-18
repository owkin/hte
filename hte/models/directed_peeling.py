"""Adaptive refinement by directed peeling procedure."""
import math
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from hte.utils.utils import survival_te, survival_did_test


class ARDP:
    """Adaptive Refinement by Directed Peeling class.

    Parameters
    ----------
    timepoint : the timepoint at which to evaluate the treatment effect (TE)
    min_size : the minimal size
    """

    def __init__(self,
                 alpha : float = 0.1,
                 timepoint : int = 1.,
                 random_state : int = 42,
                 min_size_prop : float = 0.3,
                 min_peel : int = 10):
        """Initialize a class instance."""
        self.alpha = alpha
        self.t = timepoint
        self.min_size_prop = min_size_prop
        self.min_peel = min_peel
        self.dict_standard = {}

        self.pair_dir = {}
        self.col_var = []

        self.subgroup = []
        self.drop_cols = []
        self.rule = []
        self.used_var = []

        self.size = 0
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)

    def fit(self,
            data : pd.DataFrame,
            duration_col : str,
            event_col : str,
            trt_col : str):
        """Fit the model to obtain the final subgroup after peeling.

        Parameters
        ----------
        data : pd.Dataframe
            Contains the data to fit the model.
            Should contain at least the following columns:
            Duration column, event column, treatment column
        duration_col : str
            Name of data column containing the time of event or censoring
        event_col : str
            Name of data column containing whether the event happened
        trt_col : str
            Name of data column containing the treatment status
        """
        self.size = len(data.index)
        self.dict_standard['W'] = trt_col
        self.dict_standard['T'] = duration_col
        self.dict_standard['E'] = event_col

        reduced_col = [duration_col, event_col, trt_col]
        col_var = [col for col in data.columns if col not in reduced_col]
        self.col_var = col_var

        # fit a Cox model
        cox_fitter = CoxPHFitter(penalizer=0.1)
        cox_fitter.fit(
            df=data,
            duration_col=duration_col,
            event_col=event_col
        )
        for var in self.col_var:
            if cox_fitter.params_[var] > 0:
                self.pair_dir[var] = '<='
            else:
                self.pair_dir[var] = '>'

        _data = data.copy()
        n = len(_data.index)
        best_split = []
        best_data = _data
        min_size = self.min_size_prop * n
        while n > min_size:
            best_te = 1
            for var in self.col_var:
                if self.pair_dir[var] == '<=':
                    _data = _data.sort_values(by=var, ascending=False)
                    to_remove = max(self.alpha * n, self.min_peel)
                    index_to_remove = int(n - to_remove)
                    threshold_value = _data[var].iloc[index_to_remove]
                    filtered_data = _data.loc[_data[var] > threshold_value]
                elif self.pair_dir[var] == '>':
                    _data = _data.sort_values(by=var, ascending=True)
                    to_remove = max(self.alpha * n, self.min_peel)
                    index_to_remove = int(n - to_remove)
                    threshold_value = _data[var].iloc[index_to_remove]
                    filtered_data = _data.loc[_data[var] <= threshold_value]
                trt_effect, trt_benefit = survival_te(filtered_data,
                                                      trt_col,
                                                      duration_col,
                                                      event_col)

                if trt_benefit == 0 and not math.isnan(trt_effect):
                    trt_effect = trt_effect + 1.
                if trt_effect < best_te and not math.isnan(trt_effect):
                    best_te = trt_effect
                    best_split = [var,
                                  self.pair_dir[var],
                                  threshold_value,
                                  best_te,
                                  len(filtered_data.index)]
                    best_data = filtered_data
            if best_te != 1:
                self.rule.append(best_split)
                _data = best_data
                n = len(_data.index)
                best_te = 1
            else:
                break

    def _extract_subgroup_cdt(self):
        subgroup = {}
        if self.rule:
            for r in self.rule:
                var, direction, value, _, _ = r
                # if var already here, check if refinement of lower / upper bounds
                if var in subgroup.keys():  # pylint: disable=C0201
                    if direction == ">":
                        subgroup[var]['upper_bound'] = min(value,
                                                           subgroup[var]['upper_bound'])
                    else:
                        subgroup[var]['lower_bound'] = max(value,
                                                           subgroup[var]['upper_bound'])
                # otherwise add var to the dict
                elif direction == ">":
                    subgroup[var] = {
                        'lower_bound': np.nan,
                        'upper_bound': value
                    }
                else:
                    subgroup[var] = {
                        'lower_bound': value,
                        'upper_bound': np.nan
                    }
        self.subgroup = subgroup

    def predict(self, data : pd.DataFrame):
        """Predict in which group a new sample falls.

        Parameters
        ----------
        data : pd.Dataframe
            Contains the data to be predicted.
            Should contain at least the following columns:
            Duration column, event column, treatment column

        Returns
        -------
        array of group indicators
        """
        X = data.copy()
        self._extract_subgroup_cdt()
        if self.subgroup:
            masks = []
            self.subgroup.keys()
            for var in self.subgroup.keys():  # pylint: disable=C0201,C0206
                if np.isnan(self.subgroup[var]['lower_bound']):
                    mask = X[var] <= self.subgroup[var]['upper_bound']
                else:
                    mask = X[var] > self.subgroup[var]['lower_bound']
                masks.append(mask)
            group_cdt = masks[0]
            for mask_ in masks:  # [1:]
                group_cdt = group_cdt & mask_
                X['Group'] = np.where(group_cdt, 1, 0)
        else :
            X['Group'] = 0
        return np.array(X['Group'])

    def pval_hte(self, data : pd.DataFrame):
        """Compute the p-value for heterogeneity of treatment effect.

        Parameters
        ----------
        data : pd.Dataframe
            Contains the data to be predicted using predict function.
            Should contain at least the following columns:
            Duration column, event column, treatment column

        Returns
        -------
        p_val : float
            p-value testing for treatment effect
        """
        _data = data.copy()

        # Predict subgroup
        group_est = self.predict(data=data)

        # Perform heterogeneity test
        if self.rule:
            _data['gates_col'] = group_est
            p_val = survival_did_test(
                df=_data,
                gates_col='gates_col',
                treatment_col=self.dict_standard['W'],
                duration_col=self.dict_standard['T'],
                event_col=self.dict_standard['E'],
                timepoint=self.t
            )
            return p_val
        rng = np.random.default_rng(seed=self.random_state)
        p_val = rng.uniform(0, 1)
        return p_val

    def variables_ranking(self):
        """Rank the variables by importance in HTE.

        Returns
        -------
        List of ranked variable names
        """
        if self.rule:
            df_rule = pd.DataFrame(self.rule,
                                   columns=['var', 'dir', 'split', 'pval', 'size'])
            ranking = []
            for var_ in df_rule['var'].dropna().unique():
                var_imp = 0
                var_df = df_rule[df_rule['var'] == var_]
                for _, row in var_df.iterrows():
                    node_imp = (row['size'] / (self.size + 1E-15))\
                        * (1 / (row['pval'] + 1E-15))
                    var_imp = var_imp + node_imp
                ranking.append({'var' : var_,
                                'imp' : var_imp})
            ranking_df = pd.DataFrame(ranking)
            ranked_var = np.array(
                ranking_df.sort_values(by='imp', ascending=False)['var']
            )
            ranked_var_list = list(ranked_var)
        else:
            ranked_var_list = []
        return ranked_var_list


if __name__ == '__main__':

    np.random.seed(seed=40)

    # Fake data
    SIZE_TRAIN = 1000
    W = np.random.binomial(1, 0.5, size=SIZE_TRAIN)
    E = np.random.binomial(1, 0.75, size=SIZE_TRAIN)
    X1 = np.random.normal(0, 1, size=SIZE_TRAIN)
    X2 = np.random.normal(0, 1, size=SIZE_TRAIN)
    X3 = np.random.normal(0, 1, size=SIZE_TRAIN)
    T = np.random.exponential(1, size=SIZE_TRAIN) * X1 * X3
    data = pd.DataFrame(
        np.c_[W, E, T, X1, X2, X3],
        columns=['W', 'E', 'T', 'X1', 'X2', 'X3']
    )
    # print(data)

    # TEST SIDES

    # Fit the SIDES tree
    ardp = ARDP()
    ardp.fit(data, 'T', 'E', 'W')

    # Assign new data points to their corresponding groups
    SIZE_TEST = 100
    W_test = np.random.binomial(1, 0.5, size=SIZE_TEST)
    E_test = np.random.binomial(1, 0.75, size=SIZE_TEST)
    X1_test = np.random.normal(0, 1, size=SIZE_TEST)
    X2_test = np.random.normal(0, 1, size=SIZE_TEST)
    X3_test = np.random.normal(0, 1, size=SIZE_TEST)
    T_test = np.random.exponential(1, size=SIZE_TEST) * X1_test * X3_test
    data_test = pd.DataFrame(
        np.c_[W_test,
              E_test,
              T_test, X1_test, X2_test, X3_test],
        columns=['W', 'E', 'T', 'X1', 'X2', 'X3']
    )

    print(data_test)

    output = ardp.predict(data_test)
    print(ardp.subgroup)
    print(output)

    print(ardp.pval_hte(data_test))

    print(ardp.variables_ranking())
