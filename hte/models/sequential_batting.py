"""Sequential batting procedure."""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from scipy.stats.distributions import chi2
from hte.utils.utils import survival_te, survival_did_test


class SeqBT:
    """Sequential Batting class.

    Parameters
    ----------
    alpha : the threshold p-value for significance
    timepoint : the timepoint at which to evaluate the treatment effect (TE)
    nb_splits : the number of splits to consider for each variable
    repet : the number of bootstrap sets
    """

    def __init__(
        self,
        alpha : float = 0.1,
        timepoint : int = 1.,
        nb_splits : int = 3,
        repet : int = 10
    ):
        """Initialize a class instance."""
        self.alpha = alpha
        self.t = timepoint
        self.nb_splits = nb_splits
        self.repet = repet
        self.dict_standard = {}
        self.subgroup = []
        self.drop_cols = []
        self.col_var = []
        self.rule = []
        self.rng = np.random.default_rng(123)
        self.used_var = []
        self.found_rule = False
        self.no_split = []

    def _find_split_val(self,
                        data : pd.DataFrame,
                        duration_col : str,
                        event_col : str):
        """Find the best split for each variable with bootstrap.

        Parameters
        ----------
        data : pd.Dataframe
            Contains the data to fit the model.
            Should contain at least the following columns:
            Duration column, event column

        duration_col : str
            Name of data column containing the time of event or censoring
        event_col : str
            Name of data column containing whether the event happened
        """
        q_tick = 1 / (self.nb_splits + 1)
        q_splits = [i * q_tick for i in range(1, self.nb_splits + 1)]
        splits = data[self.col_var].quantile(q=q_splits).values.T

        fitting_data = data.copy()
        best_splits_all = []
        for i, var in enumerate(self.col_var):
            best_split_var = []
            for _ in range(self.repet):
                bootstrap_data = pd.DataFrame(fitting_data
                                              .iloc[np.random
                                                    .randint(len(fitting_data),
                                                             size=len(fitting_data))])
                bootstrap_data = bootstrap_data[bootstrap_data
                                                .columns
                                                .intersection(self.drop_cols + [var])]
                bootstrap_data["rule_col"] = self.rule

                best_pval = 1
                best_split = 0
                for s in splits[i]:
                    bootstrap_data["rule_col"] = bootstrap_data["rule_col"]\
                        & np.where(bootstrap_data[var] > s, True, False)
                    random_noise = self.rng\
                        .normal(0, 1, len(bootstrap_data['rule_col'])) * 1E-13
                    noisy_rule_col = bootstrap_data['rule_col'] + random_noise
                    df_noisy = bootstrap_data.copy()
                    df_noisy['rule_col'] = noisy_rule_col
                    try:
                        cox_fitter = CoxPHFitter(penalizer=0.1)
                        cox_fitter.fit(df=df_noisy[self.drop_cols + ["rule_col"]],
                                       duration_col=duration_col,
                                       event_col=event_col)
                        if (current_pval := cox_fitter.summary.loc['rule_col']['p']) < best_pval:  # noqa : E125 pylint: disable=C0301
                            best_pval = current_pval
                            best_split = s
                            best_split_var.append(best_split)
                    except ConvergenceError as cerr:
                        print(f"ConvergenceError in findsplit: {str(cerr)}")
            best_split_var = np.median(best_split_var)
            best_splits_all.append({'var' : var, 'split' : best_split_var})
        self.best_splits_all = best_splits_all  # pylint: disable=W0201

    def _find_split_var(self,
                        data : pd.DataFrame,
                        duration_col : str,
                        event_col : str,
                        trt_col : str):
        """Find the best split among all variables.

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

        Returns
        -------
        best_pair :
            The pair made of the variable and its split value leading
            to the best new binary rule
        best_rule :
            The newly updated rule
        """
        best_pair = {}
        best_rule = []
        best_pval = 1
        _data = data.copy()
        for pair in self.best_splits_all:
            # print('new pair', pair)
            _data['rule_col'] = self.rule\
                & np.where(_data[pair['var']] > pair['split'], True, False)
            random_noise = self.rng.normal(0, 1, len(_data['rule_col'])) * 1E-13
            noisy_rule_col = _data['rule_col'] + random_noise
            df_noisy = _data.copy()
            df_noisy['rule_col'] = noisy_rule_col
            try:
                cox_fitter = CoxPHFitter(penalizer=0.1)
                cox_fitter.fit(df=df_noisy[self.drop_cols + ['rule_col']],
                               duration_col=duration_col,
                               event_col=event_col)
            except ConvergenceError as cerr:
                print(f"ConvergenceError in findsplit: {str(cerr)}")
            if (current_pval := cox_fitter.summary.loc['rule_col']['p']) < best_pval:
                best_pval = current_pval
                best_pair = pair

        if best_pair:
            pval_left, trt_benefit_left = survival_te((_data[_data[best_pair['var']]
                                                             <= best_pair['split']]),
                                                      trt_col,
                                                      duration_col,
                                                      event_col)
            pval_right, trt_benefit_right = survival_te((_data[_data[best_pair['var']]
                                                               > best_pair['split']]),
                                                        trt_col,
                                                        duration_col,
                                                        event_col)
            if (trt_benefit_left == 1) and (trt_benefit_right == 0):
                best_pair['dir'] = '<='
                best_rule = self.rule\
                    & np.where(_data[best_pair['var']] <= best_pair['split'],
                               True,
                               False)
            elif (trt_benefit_left == 0) and (trt_benefit_right == 1):
                best_pair['dir'] = '>'
                best_rule = self.rule\
                    & np.where(_data[best_pair['var']] > best_pair['split'],
                               True,
                               False)
            elif (trt_benefit_left == 1) and (trt_benefit_right == 1):
                if pval_right < pval_left:
                    best_pair['dir'] = '>'
                    best_rule = self.rule\
                        & np.where(_data[best_pair['var']] > best_pair['split'],
                                   True,
                                   False)
                else :
                    best_pair['dir'] = '<='
                    best_rule = self.rule\
                        & np.where(_data[best_pair['var']] <= best_pair['split'],
                                   True,
                                   False)
            elif (trt_benefit_left == 0) and (trt_benefit_right == 0):
                if pval_right > pval_left:
                    best_pair['dir'] = '>'
                    best_rule = self.rule\
                        & np.where(_data[best_pair['var']] > best_pair['split'],
                                   True,
                                   False)
                else :
                    best_pair['dir'] = '<='
                    best_rule = self.rule\
                        & np.where(_data[best_pair['var']] <= best_pair['split'],
                                   True,
                                   False)

        return best_pair, best_rule

    def fit(self,
            data : pd.DataFrame,
            duration_col : str,
            event_col : str,
            trt_col : str):
        """Fit the model to obtain the multiplicative rule defining the subgroup.

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
        self.dict_standard['W'] = trt_col
        self.dict_standard['T'] = duration_col
        self.dict_standard['E'] = event_col

        self.drop_cols = [duration_col, event_col, trt_col]
        self.col_var = [col for col in data.columns if col not in self.drop_cols]

        _data = data.copy()

        pval = 0
        self.rule = [True for n in range(len(_data))] & _data[trt_col].astype(bool)
        self.found_rule = False
        _data['rule_col'] = self.rule
        random_noise = self.rng.normal(0, 1, len(_data['rule_col'])) * 1E-5
        noisy_rule_col = _data['rule_col'] + random_noise
        data_noisy = _data.copy()
        data_noisy['rule_col'] = noisy_rule_col
        likelihood_rule = CoxPHFitter(penalizer=0.1)\
            .fit(df=data_noisy[self.drop_cols + ['rule_col']],
                 duration_col=duration_col,
                 event_col=event_col).log_likelihood_

        while pval < self.alpha and self.col_var:
            self._find_split_val(_data, duration_col, event_col)
            new_group, new_rule = self._find_split_var(_data,
                                                       duration_col,
                                                       event_col,
                                                       trt_col)
            if new_group:
                data_temp = _data.copy()
                data_temp['rule_col'] = new_rule
                random_noise = self.rng.normal(0, 1, len(data_temp['rule_col'])) * 1E-13
                noisy_rule_col = data_temp['rule_col'] + random_noise
                data_temp_noisy = data_temp.copy()
                data_temp_noisy['rule_col'] = noisy_rule_col
                new_likelihood_rule = CoxPHFitter(penalizer=0.1)\
                    .fit(df=data_temp_noisy[self.drop_cols + ['rule_col']],
                         duration_col=duration_col,
                         event_col=event_col).log_likelihood_
                likelihood_ratio = 2 * (new_likelihood_rule - likelihood_rule)
                pval = chi2.sf(likelihood_ratio, 1)
                if pval < self.alpha :  # pylint: disable=R6103
                    likelihood_rule = new_likelihood_rule
                    self.rule = new_rule
                    _data['rule_col'] = self.rule
                    self.col_var.remove(new_group['var'])
                    self.subgroup.append(new_group)
                    self.found_rule = True
            else:
                break

        if not self.found_rule:
            self._find_split_val(_data, duration_col, event_col)
            new_group, _ = self._find_split_var(_data,
                                                duration_col,
                                                event_col,
                                                trt_col)
            pval, trt_benefit = survival_te((_data[_data[new_group['var']]
                                                   <= new_group['split']]),
                                            trt_col,
                                            duration_col,
                                            event_col)
            if trt_benefit == 1:
                self.no_split = [True, pval, new_group['var']]
            else:
                self.no_split = [True, 1., new_group['var']]

    def predict(self,
                data : pd.DataFrame):
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
        X = data.copy().drop(self.drop_cols, axis=1)
        self.col_var = X.columns
        masks = []
        if self.subgroup:
            for subrule in self.subgroup:
                if subrule['dir'] == '<=':
                    mask = X[subrule['var']] <= subrule['split']
                else:
                    mask = X[subrule['var']] > subrule['split']
                masks.append(mask)

            group_cdt = masks[0]
            for mask in masks[1:]:
                group_cdt = group_cdt & mask
        else:
            group_cdt = False
        X['Group'] = np.where(group_cdt, 1, 0)
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
        p_val : float
            p-value testing for treatment effect
        """
        _data = data.copy()

        # Predict subgroup
        group_est = self.predict(data=data)

        # Perform heterogeneity test
        if self.subgroup:
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
        return self.no_split[1]

    def variables_ranking(self):
        """Rank the variables by importance in HTE.

        Returns
        -------
        List of ranked variable names
        """
        ranked_var = []
        if self.subgroup:
            for subrule in self.subgroup:
                ranked_var.append(subrule['var'])
            return ranked_var
        return [self.no_split[2]]


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
    seqbt = SeqBT()
    seqbt.fit(data, 'T', 'E', 'W')

    print("the final rule is", seqbt.subgroup)

    # Show subgroup rules
    # seqbt._extract_subgroup_cdt()

    # print(seqbt.best_group)

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

    # print(data_test)

    output = seqbt.predict(data_test)
    print(output)

    print(seqbt.pval_hte(data_test))

    print(seqbt.variables_ranking())

    print(seqbt.subgroup)
