"""SIDES algorithm based on method proposed by Lipkovich et al (2011).

As opposed to a traditional tree method, each parent node in SIDES leads to
numerous child subgroups possibly defined by different variables and different
split values.
It is not based on binary splits, and child groups aren't complementary. Instead,
it provides numerous plausible groups to choose from depending on the study context.
"""
from typing import Union
import pandas as pd
import numpy as np
from hte.utils.utils import survival_did_test, survival_te, survival_te_diff


class SIDES:
    """SIDES class.

    Parameters
    ----------
    alpha : the threshold p-value used to control the type I error
    timepoint : the timepoint at which to evaluate the treatment effect (TE)
    nb_splits : the maximal number of possible splits (thresholds) to consider
        for each covariate considered
    min_nb_samples : the minimal acceptable size for a subgroup
    max_splits : the maximal number of possible child subgroups to further
        investigate as parent groups
    gamma : the required improvement provided by a child subgroup
        compared to its parent group
    resampling : the number of repetitions in the type I error control process
    depth : the maximal depth (number of layers) acceptable

    """

    def __init__(self,
                 alpha : int = 0.1,
                 timepoint : int = 1.,
                 nb_splits : int = 3,
                 min_nb_samples : Union[float, int] = 0.1,
                 max_splits : int = 3,
                 gamma : int = 0.8,
                 resampling : int = 5,
                 depth : int = 4):
        """Initialize a class instance."""
        self.alpha = alpha
        self.t = timepoint
        self.nb_splits = nb_splits
        self.min_nb_samples = min_nb_samples
        self.min_nb_samples_stop = min_nb_samples
        self.max_splits = max_splits
        self.gamma = gamma
        self.resampling = resampling
        self.depth = depth - 1
        self.subgroup = []
        self.rules = []
        self.drop_cols = []
        self.tree = {}
        self.df_tree = pd.DataFrame()
        self.dict_standard = {}
        self.final_nodeid = ''
        self.candidate_subgroups = []

    def _splitting_criterion(self,
                             data : pd.DataFrame,
                             duration_col : str,
                             event_col : str,
                             trt_col : str,
                             used_cols : list) :
        """Compute the splitting criterion defining promising subgroups.

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
        used_cols : list
            Names of columns previously used for current data split

        Returns
        -------
        df : promising subgroups as pd.DataFrame with columns
            Var | Split | TE_child | Dir | TE_parent
        """
        # Keep only covariate columns
        covariates_df = data.drop(self.drop_cols, axis=1)
        if len(used_cols) != 0:
            # Exclude covariates already used in higher levels of that node path
            covariates_df = covariates_df.drop(used_cols, axis=1)
        promising_subgroups = []
        pvalues_list = []

        # Consider all covariates not previously used in higher levels of that node path
        for col in covariates_df.columns:
            # Define the splits to be considered based on specified number of splits
            q_tick = 1 / (self.nb_splits + 1)
            q_splits = [i * q_tick for i in range(1, self.nb_splits + 1)]
            var_splits = data[col].quantile(q=q_splits).values

            # Consider the splits defined above
            for s in var_splits:
                var_low_idx = data[col] <= s
                var_high_idx = data[col] > s
                if min(sum(var_low_idx), sum(var_high_idx)) >= self.min_nb_samples_stop:
                    pval = survival_te_diff(data,
                                            trt_col,
                                            duration_col,
                                            event_col,
                                            col,
                                            s,
                                            self.t)
                    # Each split is defined by:
                    #  the variable on which its performed,
                    #  the threshold (split value),
                    #  the p-value of TE difference between child subgroups
                    subgroup0_te, trt_benefit0 = survival_te(data[var_low_idx],
                                                             trt_col,
                                                             duration_col,
                                                             event_col,
                                                             self.t)
                    if trt_benefit0 == 0:
                        subgroup0_te = 1. - 1E-10
                    subgroup1_te, trt_benefit1 = survival_te(data[var_high_idx],
                                                             trt_col,
                                                             duration_col,
                                                             event_col,
                                                             self.t)
                    if trt_benefit1 == 0:
                        subgroup1_te = 1. - 1E-10
                    pvalues_list.append({'Var': col,
                                         'Split': s,
                                         'pvalue': pval,
                                         'subgroup0_te': subgroup0_te,
                                         'subgroup1_te': subgroup1_te})
        pvalues_df = pd.DataFrame(pvalues_list,
                                  columns=['Var',
                                           'Split',
                                           'pvalue',
                                           'subgroup0_te',
                                           'subgroup1_te'])

        # Consider only a subgroup of best splits as defined by max_splits
        pvalues_df = pvalues_df.sort_values('pvalue', ascending=True)
        best_pairs = pvalues_df.iloc[0:self.max_splits]
        # For each best pair, keep only the child maximizing the TE
        for _, pair in best_pairs.iterrows():
            direction = np.where(pair['subgroup0_te'] < pair['subgroup1_te'], '<=', '>')
            promising_subgroups.append({'Var': pair['Var'], 'Split': pair['Split'],
                                        'TE_child': min(pair['subgroup0_te'],
                                                        pair['subgroup1_te']),
                                        'Dir': direction,
                                        'TE_parent': survival_te(data,
                                                                 trt_col,
                                                                 duration_col,
                                                                 event_col,
                                                                 self.t)[0]})
        return pd.DataFrame(promising_subgroups,
                            columns=['Var', 'Split', 'TE_child', 'Dir', 'TE_parent'])

    def _selection_criterion(self,
                             data : pd.DataFrame,
                             duration_col : str,
                             event_col : str,
                             trt_col : str):
        """Compute selection criterion defining which subgroups are candidates.

        A candidate subgroup is a subgroup with desirable efficacy and
        acceptable safety characteristics,
        selected as such based on type I error control considerations.

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
        is_candidate : bool
        """
        # Keep only covariate columns
        reduced_df = data[self.drop_cols]
        covariates_df = data.drop(self.drop_cols, axis=1)

        prop_adjusted = np.zeros(self.df_tree.shape[0])

        # Repeat the process N times as defined by the user-defined resampling parameter
        for _ in range(0, self.resampling):
            # Construct null model sets:
            # Permute the covariates data while maintaining the outcome;treatment pairs
            data_perm = pd.concat([reduced_df.reset_index(drop=True),
                                   covariates_df.
                                   iloc[np.random.permutation(len(covariates_df))].
                                   reset_index(drop=True)], axis=1)

            # Identify promising subgroups from permuted sets
            perm_promising_subgroups = self._tree_fit(data_perm,
                                                      duration_col,
                                                      event_col,
                                                      trt_col)

            if not perm_promising_subgroups.empty:
                p_min = min(perm_promising_subgroups['TE_pval'])
            else:
                p_min = 1

        # Control the type I error
        # by accounting for significant results obtained from null (permuted) sets
            for c in range(0, self.df_tree.shape[0]):
                if p_min < self.df_tree['TE_pval'].iloc[c]:
                    prop_adjusted[c] = prop_adjusted[c] + 1
        p_adjusted = prop_adjusted / self.resampling

        # A subgroup is a candidate subgroup
        # if its adjusted p-value respects the user-defined p threshold parameter
        is_candidate = []
        for index, _ in self.df_tree.iterrows():
            if p_adjusted[index] < self.alpha:
                is_candidate.append(True)
            else:
                is_candidate.append(False)
        return is_candidate

    def _node_fit(self,
                  data : pd.DataFrame,
                  duration_col : str,
                  event_col : str,
                  trt_col : str,
                  used_cols : list = None):
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
        used_cols : list
            Names of columns previously used for current data split

        Returns
        -------
        Union[bool, str, int, str, bool]
            Tuple containing:
                stop_cdt : bool
                    Whether a split was performed
                split_var : str
                    Split variable for the subgroup
                split_val : int
                    Split value for the subgroup
                split_dir : str
                    Split direction for the subgroup
                split_candidate : bool
                    Whether the subgroup is candidate
        """
        # Define possible child subgroups from parent group (node)
        # Test the splitting criterion to obtain promising subgroups
        promising_subgroups = self._splitting_criterion(data,
                                                        duration_col,
                                                        event_col,
                                                        trt_col,
                                                        used_cols)

        # Test continuation criterion on promising subgroups
        # to define future parent groups
        stop_cdt = []
        split_var = []
        split_val = []
        split_dir = []
        split_pval = []

        for _, row in promising_subgroups.iterrows():
            if row['TE_child'] <= self.gamma * row['TE_parent']:
                stop_cdt.append(False)
                split_var.append(row['Var'])
                split_val.append(row['Split'])
                split_dir.append(str(row['Dir']))
                split_pval.append(row['TE_child'])
            else:
                stop_cdt.append(True)
                split_var.append(None)
                split_val.append(None)
                split_dir.append(None)
                split_pval.append(None)

        # Testing selection criterion on promising subgroups
        # to define candidate subgroups

        return stop_cdt, split_var, split_val, split_dir, split_pval

    def _tree_fit(self,
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

        Returns
        -------
        tree_df : pd.DataFrame
        """
        self.dict_standard['W'] = trt_col
        self.dict_standard['T'] = duration_col
        self.dict_standard['E'] = event_col

        self.drop_cols = [duration_col, event_col, trt_col]
        stack = [{'node_id': '0', 'node_depth': 0, 'node_data': data, 'used_covs': []}]
        self.tree = []

        # Initialize the tree by adding the root node
        root_node = {'node_id': 0,
                     'node_depth': 0,
                     'node_data': data,
                     'best_var': None,
                     'best_split': None,
                     'best_dir': None,
                     'is_leaf': False,
                     'TE_pval': survival_te(data,
                                            trt_col,
                                            duration_col,
                                            event_col,
                                            self.t)[0]}
        self.tree.append(root_node)

        # Iterate until there is no more node to try splitting
        while len(stack) > 0:  # pylint: disable=R1702
            # Look at the last available node in the stack
            elem = stack.pop()
            current_node = elem['node_id']
            current_depth = elem['node_depth']
            current_data = elem['node_data']
            current_cols = elem['used_covs']

            # Only if this node isn't at the maximal depth, try to split
            if current_depth < self.depth:
                output = self._node_fit(
                    data=current_data,
                    duration_col=duration_col,
                    event_col=event_col,
                    trt_col=trt_col,
                    used_cols=current_cols
                )
                # If at least one split has been performed:
                # (1) Add the new nodes to the tree
                # (2) Add whether new nodes are parent groups
                if sum(output[0]) < len(output[0]):
                    vars_ = output[1]
                    splits = output[2]
                    dirs = output[3]
                    pvals = output[4]
                    child_n = 0
                    child_id = 0
                    # For all defined child nodes:
                    while child_n < len(vars_):
                        if vars_[child_n] is not None:
                            if dirs[child_n] == "<=":
                                data_child = data[data[vars_[child_n]]
                                                  <= splits[child_n]]
                            else:
                                data_child = data[data[vars_[child_n]]
                                                  > splits[child_n]]

                            # Add the child to the tree
                            elem = {'node_id': current_node + str(child_id),
                                    'node_depth': current_depth + 1,
                                    'node_data': data_child,
                                    'best_var': vars_[child_n],
                                    'best_split': splits[child_n],
                                    'best_dir': dirs[child_n],
                                    'is_leaf': np.where(current_depth + 1 == self.depth,
                                                        True, False),
                                    'TE_pval': pvals[child_n]
                                    }
                            self.tree.append(elem)

                            # Add the child to the stack of nodes to be tested for split
                            child = {
                                'node_id': current_node + str(child_id),
                                'node_depth': current_depth + 1,
                                'node_data': data_child,
                                'used_covs': [*current_cols, vars_[child_n]]
                            }

                            stack.append(child)
                            child_id = child_id + 1
                        child_n = child_n + 1
                else:
                    for elem in self.tree:
                        if elem['node_id'] == current_node:
                            elem['is_leaf'] = True
        # Obtain the final tree
        return pd.DataFrame(self.tree).drop(columns=['node_data'])

    def fit(self,
            data : pd.DataFrame,
            duration_col : str,
            event_col : str,
            trt_col : str):
        """Fit the SIDES model.

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
        if isinstance(self.min_nb_samples, float):
            self.min_nb_samples_stop = self.min_nb_samples * len(data)

        self.df_tree = self._tree_fit(data,
                                      duration_col,
                                      event_col,
                                      trt_col)

        # Define which tree subgroups are candidate subgroups
        is_candidate = self._selection_criterion(data,
                                                 duration_col,
                                                 event_col,
                                                 trt_col)

        self.df_tree['is_candidate'] = is_candidate

        if len(self.df_tree.index) == 1:
            # repeat first split to find the most promising split
            most_promising_repeat = self._splitting_criterion(data,
                                                              duration_col,
                                                              event_col,
                                                              trt_col,
                                                              [])
            promising_var = most_promising_repeat \
                .iloc[most_promising_repeat['TE_child'].idxmin()]['Var']
            promising_pval = most_promising_repeat['TE_child'].min()
            self.df_tree['TE_pval'] = promising_pval
            self.df_tree['best_var'] = promising_var

    def _extract_subgroup_cdt(self):
        """Extract the subgroup rules from the final tree.

        Rules extracted are rules coming from leaf nodes
        that also qualify as a candidate subgroup.
        Each subgroup rule is defined as
        the combination of the rules of each node
        between the root node and this subgroup node (path).
        """
        leafs = self.df_tree[self.df_tree['is_leaf'] == 1]
        self.candidate_subgroups = leafs[leafs['is_candidate']].index
        rules = []
        final_rule = []
        final_rule_pval = 1
        # self.final_nodeid = ''
        # Get the rules by browsing the tree
        for candidate in self.candidate_subgroups:
            candidate_rule = []
            current_node = self.df_tree.iloc[candidate]['node_id']
            while len(current_node) > 1:
                if current_node != '0':
                    var = self.df_tree[self.df_tree['node_id']
                                       == current_node]['best_var'].to_numpy()[0]
                    split = self.df_tree[self.df_tree['node_id']
                                         == current_node]['best_split'].to_numpy()[0]
                    dir_ = self.df_tree[self.df_tree['node_id']
                                        == current_node]['best_dir'].to_numpy()[0]

                    r = [var, split, dir_]
                    candidate_rule.append(r)
                current_node = current_node[:-1]
            rules.append(candidate_rule)
            if self.df_tree.iloc[candidate]['TE_pval'] < final_rule_pval:
                final_rule = candidate_rule
                final_rule_pval = self.df_tree.iloc[candidate]['TE_pval']
                self.final_nodeid = self.df_tree\
                    .iloc[candidate]['node_id']  # pylint: disable=W0201

        self.rules = rules
        groups = []
        final_group = {}
        for rule in rules:
            current_dict = {}
            for r in rule:
                var, split, dir_ = r
                if dir_ == "<=":
                    current_dict[var] = {
                        'lower_bound': np.nan,
                        'upper_bound': split
                    }
                else:
                    current_dict[var] = {
                        'lower_bound': split,
                        'upper_bound': np.nan
                    }
            if rule == final_rule:
                final_group = current_dict
            groups.append(current_dict)
        self.subgroup = groups

        self.best_rule = final_rule  # pylint: disable=W0201
        self.best_group = final_group  # pylint: disable=W0201

    def predict(self, data):
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
        X = X.drop(self.drop_cols, axis=1)
        self.col_var = X.columns  # pylint: disable=W0201
        masks = []
        if (group := self.best_group):
            for var, value in group.items():
                if np.isnan(value['lower_bound']):
                    mask = X[var] <= value['upper_bound']
                else:
                    mask = X[var] <= value['lower_bound']
                masks.append(mask)
            group_cdt = masks[0]
            for mask in masks[1:]:
                group_cdt = group_cdt & mask
        else:
            group_cdt = False

        X['Best Group'] = group_cdt
        X['Check'] = X.drop(columns=self.col_var).sum(axis=1)

        X['Group'] = np.where(X['Best Group'], 1, 0)
        return np.array(X['Group'])

    def pval_hte(self,
                 data : pd.DataFrame):
        """Compute the p-value for heterogeneity of treatment effect.

        Parameters
        ----------
        data : pd.Dataframe
            Contains the data to be predicted using predict function.
            Should contain at least the following columns:
            Duration column, event column, treatment column

        Returns
        -------
        p_val: float
            P-value testing for treatment effect
        """
        _data = data.copy()
        X = _data

        # Predict subgroup
        group_est = self.predict(data=X)

        if self.best_group:
            # Perform heterogeneity test
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
        if len(self.candidate_subgroups) == 0:
            is_first_generation = self.df_tree['node_id'] \
                .apply(lambda x: len(str(x)) == 2)
            df_reduced = self.df_tree[['node_id',
                                       'TE_pval',
                                       'best_var']][is_first_generation]
            return df_reduced['TE_pval'].min()
        if len(self.df_tree.index) == 1:
            return self.df_tree['TE_pval'].iloc[0]
        return np.nan

    def variables_ranking(self):
        """Rank the variables by importance in HTE.

        Returns
        -------
        List of ranked variable names
        """
        ids = self.final_nodeid
        if ids == '' and len(self.df_tree.index) == 1:
            return [self.df_tree['best_var'].iloc[0]]
        if len(self.candidate_subgroups) == 0:
            is_first_generation = self.df_tree['node_id'] \
                .apply(lambda x: len(str(x)) == 2)
            df_reduced = self.df_tree[['node_id',
                                       'TE_pval',
                                       'best_var']][is_first_generation]
            return [df_reduced.loc[df_reduced['TE_pval'].idxmin(), 'best_var']]
        ranked_var = []
        while len(ids) > 1:
            ranked_var.append(self.df_tree[self.df_tree['node_id'] == ids]['best_var']
                              .to_numpy()[0])
            ids = ids[:-1]
        output = ranked_var
        return output


if __name__ == '__main__':

    np.random.seed(seed=40)

    # Fake data
    SIZE_TRAIN = 1000
    W = np.random.binomial(1, 0.5, size=SIZE_TRAIN)
    E = np.random.binomial(1, 0.75, size=SIZE_TRAIN)
    X1 = np.random.normal(0, 1, size=SIZE_TRAIN)
    X2 = np.random.normal(0, 1, size=SIZE_TRAIN)
    X3 = np.random.normal(0, 1, size=SIZE_TRAIN)
    T = np.random.exponential(1, size=SIZE_TRAIN) * X1 * X1 * X2
    data = pd.DataFrame(
        np.c_[W, E, T, X1, X2, X3],
        columns=['W', 'E', 'T', 'X1', 'X2', 'X3']
    )

    # TEST SIDES

    # Fit the SIDES tree
    sides_t = SIDES(min_nb_samples=100)
    sides_t.fit(data, 'T', 'E', 'W')

    # Show the tree structure
    print(sides_t.df_tree)

    # Show subgroup rules
    # sides_t._extract_subgroup_cdt()

    # print(sides_t.best_group)

    # Assign new data points to their corresponding groups
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

    output = sides_t.predict(data_test)
    print(output)

    print(sides_t.pval_hte(data_test))

    print(sides_t.variables_ranking())
