"""Interaction trees analysis."""

import warnings
from typing import Union
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from hte.utils.utils import survival_te
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


class ITree:
    """Interaction trees class.

    Parameters
    ----------
    alpha : the threshold p-value for significance
    timepoint : the timepoint at which to evaluate the treatment effect (TE)
    nb_splits : the number of splits to consider at each node
    min_nb_samples : the minimal acceptable size for a child node
    """

    def __init__(self,
                 alpha : float = 0.1,
                 timepoint : float = 1.,
                 nb_splits : int = 3,
                 min_nb_samples : Union[float, int] = 0.1):
        """Initialize a class instance."""
        self.alpha = alpha
        self.t = timepoint
        self.nb_splits = nb_splits
        self.min_nb_samples = min_nb_samples
        self.min_nb_samples_stop = min_nb_samples
        self.subgroup = []
        self.drop_cols = []
        self.tree = {}
        self.df_tree = pd.DataFrame()
        self.rules = []
        self.rng = np.random.default_rng(123)
        self.dict_standard = {}
        self.col_var = []
        self.dim_covariables = 0
        self.top_sbg = ''

    def _node_fit(self,  # pylint: disable=R0915
                  data : pd.DataFrame,
                  duration_col : str,
                  event_col : str,
                  trt_col : str):
        """Perform split at each node.

        Parameters
        ----------
        data : pd.Dataframe
            Contains the data to fit the model at current node.
            Should contain at least the following columns:
            Duration column, event column, treatment column
        duration_col : str
            Name of data column containing the time of event or censoring
        event_col : str
            Name of data columncontaining whether the event happened
        trt_col : str
            Name of data column containing the treatment status

        Returns
        -------
        Union[bool, str, int, pd.Dataframe, pd.Dataframe]
            Tuple containing:
                stop_cdt : bool
                    Whether a split was performed
                var : str
                    Split variable for the subgroup
                split : int
                    Split value for the subgroup
                subgroup0 : pd.Dataframe
                    First resulting subgroup data
                subgroup1 : pd.Dataframe
                    Second resulting subgroup data
        """
        reduced_col = [duration_col, event_col, trt_col]
        self.drop_cols = reduced_col
        col_var = [col for col in data.columns if col not in reduced_col]
        self.col_var = col_var

        # define the splits for each variable
        q_tick = 1 / (self.nb_splits + 1)
        q_splits = [i * q_tick for i in range(1, self.nb_splits + 1)]
        splits = data[self.col_var].quantile(q=q_splits).values.T

        # for each pair (variable, split), define the thresholded var
        _data = data.copy()
        col_pair_var_split = []
        col_interactions = []
        for i, var in enumerate(self.col_var):
            var_splits = splits[i]
            var_splits = set(list(var_splits))
            for s in var_splits:
                _data = pd.concat((_data.reset_index(drop=True),
                                   (_data[var] <= s).replace({True: 1, False: 0})
                                   .reset_index(drop=True)
                                   .rename(f"({var} <= {s})")), axis=1)
                _data = pd.concat((_data.reset_index(drop=True),
                                   (_data[f"({var} <= {s})"] * _data[trt_col])
                                   .reset_index(drop=True)
                                   .rename(f"Int ({var} <= {s})")), axis=1)
                col_pair_var_split.append(f"({var} <= {s})")
                col_interactions.append(f"Int ({var} <= {s})")
        _data.index = data.index

        # for each pair (variable, split), fit a Cox model with interaction
        pvals = {}
        for v, i in zip(col_pair_var_split, col_interactions):
            # fit the Cox
            cox_fitter = CoxPHFitter(penalizer=0.1)
            # random_noise_e = self.rng.normal(0, 1, len(_data[v])) * 1E-13
            random_noise_w = self.rng.normal(0, 1, len(_data[v])) * 1E-13
            random_noise_v = self.rng.normal(0, 1, len(_data[v])) * 1E-13
            random_noise_i = self.rng.normal(0, 1, len(_data[i])) * 1E-13
            # e_noisy = _data['E'] + random_noise_e
            w_noisy = _data['W'] + random_noise_w
            v_noisy = _data[v] + random_noise_v
            i_noisy = _data[i] + random_noise_i
            df_noisy = _data
            # df_noisy['E'] = e_noisy
            df_noisy['W'] = w_noisy
            df_noisy[v] = v_noisy
            df_noisy[i] = i_noisy
            # print(df_noisy[reduced_col + [v, i]])
            cox_fitter.fit(
                df=df_noisy[reduced_col + [v, i]],
                duration_col=duration_col,
                event_col=event_col,
            )
            pval = cox_fitter.summary.loc[i]['p']
            pvals[pval] = v
            # if (pval := cox_fitter.summary.loc[i]['p']) < current_pval:
            #     # if pval < current_pval:
            #     current_pval = pval
            #     current_criterion = v
        min_pvals = min(list(pvals.keys()))
        criterion = pvals[min_pvals]
        adj_min_pvals = min_pvals * self.dim_covariables
        if adj_min_pvals < self.alpha:  # pylint: disable=R6103
            var, split = criterion.split(' <= ')
            var = var[1:]
            split = float(split[:-1])
            stop_cdt = False
            var_low_idx = _data[var] <= split
            var_upper_idx = _data[var] > split
            idx_low = var_low_idx[var_low_idx == True].index  # noqa : E712
            idx_high = var_low_idx[var_upper_idx == True].index  # noqa : E712
            len0 = len(idx_low)
            len1 = len(idx_high)
            if min(len0, len1) >= self.min_nb_samples_stop:
                # if current_pval <= self.alpha:
                return stop_cdt, var, split, idx_low, idx_high, adj_min_pvals
            stop_cdt = True
            return stop_cdt, var, split, idx_low, idx_high, adj_min_pvals
        stop_cdt = True
        var, split = criterion.split(' <= ')
        var = var[1:]
        split = float(split[:-1])
        return stop_cdt, var, split, None, None, adj_min_pvals

    def fit(self,
            data : pd.DataFrame,
            duration_col : str,
            event_col : str,
            trt_col : str):
        """Fit the tree-based model until user-specified depth is reached.

        Parameters
        ----------
        data : pd.Dataframe
            Contains the data to fit the model.
            Should contain at least the following columns:
            Duration column, event column, treatment column
        duration_col : str
            Name of data column containing the time of event or censoring
        event_col : str
            Name of data columncontaining whether the event happened
        trt_col : str
            Name of data column containing the treatment status
        """
        self.dim_covariables = data.shape[1] - 3
        self.dict_standard['W'] = trt_col
        self.dict_standard['T'] = duration_col
        self.dict_standard['E'] = event_col

        if isinstance(self.min_nb_samples, float):
            self.min_nb_samples_stop = self.min_nb_samples * len(data)

        stack = [
            {
                'node_id': '0',
                'node_depth': 0,
                'node_idx': data.index,
                'node_len': len(data.index)
            }
        ]
        self.tree = []
        while len(stack) > 0:
            # look at last available node
            elem = stack.pop()
            current_node = elem['node_id']
            current_depth = elem['node_depth']
            current_idx = elem['node_idx']

            # try to split at the node
            output = self._node_fit(
                data=data.iloc[current_idx],
                duration_col=duration_col,
                event_col=event_col,
                trt_col=trt_col
            )

            # if split has been performed, proceed
            if not output[0]:
                best_var = output[1]
                best_split = output[2]
                idx_left = output[3]
                idx_right = output[4]
                pval = output[5]

                elem['is_leaf'] = False
                elem['best_var'] = best_var
                elem['best_split'] = best_split
                elem['pval'] = pval

                self.tree.append(elem)

                child_left = {
                    'node_id': current_node + '0',
                    'node_depth': current_depth + 1,
                    'node_idx': idx_left,
                    'node_len': len(idx_left),
                }

                child_right = {
                    'node_id': current_node + '1',
                    'node_depth': current_depth + 1,
                    'node_idx': idx_right,
                    'node_len': len(idx_right)
                }

                stack.append(child_left)
                stack.append(child_right)

            # otherwise, add leaf
            else:
                elem['is_leaf'] = True
                elem['pval'] = output[5]
                elem['best_var'] = output[1]
                elem['best_split'] = output[2]
                self.tree.append(elem)

        self.df_tree = pd.DataFrame(self.tree).drop(columns=['node_idx'])

    def _extract_subgroup_cdt(self):
        leafs = self.df_tree[self.df_tree['is_leaf']].index
        rules = []
        # get the rules by browsing the tree
        for leaf in leafs:
            leaf_rule = []
            current_node = self.df_tree.iloc[leaf]['node_id']
            while len(current_node) > 1:
                below = current_node[-1] == '0'
                # if current_node[-1] == '0':
                #    below = True
                # else:
                #    below = False
                current_node = current_node[:-1]
                var = self.df_tree[self.df_tree['node_id']
                                   == current_node]['best_var'].to_numpy()[0]
                split = self.df_tree[self.df_tree['node_id']
                                     == current_node]['best_split'].to_numpy()[0]

                r = [var, split, below]
                leaf_rule.append(r)
            rules.append(leaf_rule)
        self.rules = rules
        # simplify the rules
        # model of rule:
        # {
        #   'X1': {
        #       'lower_bound': xx,
        #       'upper_bound': xx
        #   }
        # }
        groups = []
        for rule in rules:
            current_dict = {}
            for r in rule:
                var, split, below = r
                # if var already here, check if refinement of lower / upper bounds
                if var in current_dict.keys():  # pylint: disable=C0201
                    lower = current_dict[var]['lower_bound']
                    upper = current_dict[var]['upper_bound']
                    if below:
                        # if np.isnan(lower):
                        #     current_dict[var]['upper_bound'] = split
                        if np.isnan(upper):
                            current_dict[var]['upper_bound'] = split
                        elif split < upper:
                            current_dict[var]['upper_bound'] = split
                    elif np.isnan(lower):
                        current_dict[var]['lower_bound'] = split
                    else:
                        current_dict[var]['lower_bound'] = split
                # otherwise add var to the dict
                elif below:
                    current_dict[var] = {
                        'lower_bound': np.nan,
                        'upper_bound': split
                    }
                else:
                    current_dict[var] = {
                        'lower_bound': split,
                        'upper_bound': np.nan
                    }
            groups.append(current_dict)
        self.subgroup = groups

    def predict(  # pylint: disable=R0912
        self,
        data : pd.DataFrame,
        responders : str = 'top',
        prop : float = 1.
    ):
        """Predict in which group a new sample falls.

        Parameters
        ----------
        data : pd.Dataframe
            Contains the data to be predicted.
            Should contain at least the following columns:
            Duration column, event column, treatment column

        responders : str
            Informs whether to provide the top group of
            responders ("top") or to aggregate some leafs
            together to form the subgroup of best responders

        prop : float
            Corresponds to the proportion of subgroups
            to aggregate to form the subgroup of responders.
            Only used if responders is "aggreg".

        Returns
        -------
        array of group indicators
        """
        X = data.copy()
        self._extract_subgroup_cdt()
        # print(self.subgroup)
        # X = X[self.col_var].copy()
        for i, group in enumerate(self.subgroup):
            # print(i)
            masks = []
            for var in group.keys():
                if np.isnan(group[var]['lower_bound']):
                    mask = X[var] <= group[var]['upper_bound']
                elif np.isnan(group[var]['upper_bound']):
                    mask = X[var] > group[var]['lower_bound']
                else:
                    mask = (X[var] <= group[var]['upper_bound']) \
                        & (X[var] > group[var]['lower_bound'])
                masks.append(mask)
            if masks:
                # if (group_cdt := masks[0])
                group_cdt = masks[0]
                for mask_ in masks[1:]:
                    group_cdt = group_cdt & mask_
            # X['Group {}'.format(i)] = group_cdt
                X[f'Group {i}'] = group_cdt
            else :
                X['Group 0'] = True
        # print(X.drop(self.drop_cols, axis=1))
        X['Group'] = X.drop(self.drop_cols, axis=1)\
            .apply(lambda x: int(x[x == True]  # noqa : E712
                                 .index[0]
                                 .split(' ')[1]), axis=1)

        # Estimate treatment effect in each subgroup
        te_dict = {}
        for i in X['Group'].unique():
            X_i = X[X['Group'] == i]
            tr_effect, trt_benefit = survival_te(df=X_i.drop('Group', axis=1),
                                                 treatment_col=self.dict_standard['W'],
                                                 duration_col=self.dict_standard['T'],
                                                 event_col=self.dict_standard['E'])
            if trt_benefit == 1:
                te_dict[i] = tr_effect
            else:
                te_dict[i] = tr_effect + 1.
        X['TE'] = X['Group'].replace(te_dict)
        ranked_effects = X['TE']\
            .sort_values(ascending=True, na_position="last").unique()

        # rule of the top subgroup
        top_te = ranked_effects[0]
        idx_top_sbg = X[X['TE'] == top_te].iloc[0]['Group']
        if len(self.subgroup[0]) == 0:
            self.top_sbg = 'All'
        else:
            self.top_sbg = self.subgroup[idx_top_sbg]

        if responders == 'top':
            # Return only the best group of good responders
            good_responders_te = ranked_effects[0]
            X['Good_Responders'] = np.where(X['TE'] == good_responders_te, 1, 0)
            return np.array(X['Good_Responders'])

        # Else,
        if responders == 'aggreg':
            nb_sub = len(ranked_effects)
            if (number_aggreg := round(prop * nb_sub)) <= nb_sub:
                good_responders_te = ranked_effects[0:number_aggreg]
                # print(ranked_effects[0:number_aggreg])
                X['Good_Responders'] = np.where(X['TE'].isin(good_responders_te), 1, 0)
                return np.array(X['Good_Responders'])
            return 'wrong proportion, should be between 0 and 1'
        return 'wrong type of good responders subgroup'

    def pval_hte(
        self
    ):
        """Compute the p-value for heterogeneity of treatment effect.

        Returns
        -------
        p_val: float
            P-value testing for treatment effect
        """
        p_val = self.df_tree.iloc[0]['pval']
        return p_val

    def variables_ranking(self):
        """Rank the variables by importance in HTE.

        Returns
        -------
        List of ranked variable names
        """
        samples_total = self.df_tree.iloc[0]['node_len']
        ranking = []
        if len(self.df_tree) == 1:
            ranked_var = self.df_tree.iloc[0]['best_var']
            ranked_var_list = [ranked_var]
        else:
            for var_ in self.df_tree[
                self.df_tree['is_leaf'] == False  # noqa : E712
            ]['best_var'].unique():
                var_imp = 0
                var_df = self.df_tree[
                    (self.df_tree['best_var'] == var_)
                    & (self.df_tree['is_leaf'] == False)  # noqa : E712
                ]
                for _, row in var_df.iterrows():
                    node_imp = (row['node_len'] / samples_total) \
                        * (1 / (row['pval'] + 1E-15))
                    var_imp = var_imp + node_imp
                ranking.append({'var' : var_,
                                'imp' : var_imp})
            ranking_df = pd.DataFrame(ranking)
            ranked_var = np.array(
                ranking_df.sort_values(by='imp', ascending=False)['var']
            )
            ranked_var_list = list(ranked_var)
        return ranked_var_list


if __name__ == '__main__':

    np.random.seed(seed=10)

    # fake data
    SIZE_TRAIN = 1000
    W = np.random.binomial(1, 0.5, size=SIZE_TRAIN)
    E = np.random.binomial(1, 0.75, size=SIZE_TRAIN)
    X1 = np.random.normal(0, 1, size=SIZE_TRAIN)
    X2 = np.random.normal(0, 1, size=SIZE_TRAIN)
    X3 = np.random.normal(0, 1, size=SIZE_TRAIN)
    # X4 = np.random.normal(0, 1, size=SIZE_TRAIN)
    T = np.random.exponential(1, size=SIZE_TRAIN) * X1 * X2**2
    data = pd.DataFrame(
        np.c_[W, E, T, X1, X2, X3],
        columns=['W', 'E', 'T', 'X1', 'X2', 'X3']
    )
    # print(data.head())

    # test the interaction tree
    int_tree = ITree(
        alpha=0.5,
        nb_splits=3,
        min_nb_samples=0.1
    )
    int_tree.fit(data, 'T', 'E', 'W')

    # show the tree
    # print(int_tree.tree)
    # print(int_tree.df_tree)

    # try rules
    # mob._extract_subgroup_cdt()

    # rules
    # mob._extract_subgroup_cdt()
    # for rule in mob.rules:
    #    print(rule)
    # for group in mob.groups:
    #     print(group)
    # print(rules)
    # print(len(rules))
    # print(len(mob.df_tree[mob.df_tree['is_leaf'] == True]))

    # test set
    SIZE_TEST = 100
    W_test = np.random.binomial(1, 0.5, size=SIZE_TEST)
    E_test = np.random.binomial(1, 0.75, size=SIZE_TEST)
    X1_test = np.random.normal(0, 1, size=SIZE_TEST)
    X2_test = np.random.normal(0, 1, size=SIZE_TEST)
    X3_test = np.random.normal(0, 1, size=SIZE_TEST)
    T_test = np.random.exponential(1, size=SIZE_TEST) * X1_test * X3_test
    data_test = pd.DataFrame(
        np.c_[W_test, E_test, T_test, X1_test, X2_test, X3_test],
        columns=['W', 'E', 'T', 'X1', 'X2', 'X3']
    )

    output = int_tree.predict(data_test)
    print(output)

    print(int_tree.pval_hte())

    print(int_tree.variables_ranking())

    print(int_tree.top_sbg)
