"""Univariate analysis."""

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import fdrcorrection
from hte.utils.utils import survival_did_test


class Univariate():
    """The Univariate method."""

    def __init__(
        self,
        method : str = 'interaction',
        nb_splits : int = 1,
        timepoint : float = 1.,
        saving_pvalues : bool = False
    ):
        """Instantiate object with hyper-parameters.

        Parameters
        ----------
        method : str
            Current version: two choices.
            - 'interaction' -> the canonical method where
            a Cox model is fited with interaction term.
            - 't_test' -> perform t-test between two subgroups
            defined by thresholding along 1D covariates.
        nb_splits : int
            Only used for t_test. The number of splits for each 1D covariate.
        timepoint : float
            The point where ARR is computed.
        saving_pvalues : bool
            Whether to save the univariate pvalues in a dataframe.
        """
        self.method = method
        self.nb_splits = nb_splits
        self.t = timepoint
        self.saving_pvalues = saving_pvalues

        self.pvals_detailed = {}
        self.pvals = {}
        self.median = {}

    def fit(
        self,
        data : pd.DataFrame,
        duration_col : str,
        event_col : str,
        trt_col : str,
        features : list[str] = None
    ):
        """Perform test for each covariate of interest.

        Parameters
        ----------
        data : pd.DataFrame
            The data under consideration.
        duration_col : str
            The duration column of data.
        event_col : str
            The event column of data.
        trt_col : str
            The treatment allocation column of data.
        features : List[str]
            The list of covariates one want to test.
            By default equal to None -> everything is tested.
        """
        # prepare data
        _data = data.copy()
        reduced_col = [duration_col, event_col, trt_col]
        if features is None:
            features = []
            for elem in data.columns:
                if elem not in [duration_col, event_col, trt_col]:
                    features.append(elem)

        # loop on features and test
        for feature in features:
            if (_data[feature].nunique()) == 2:
                if self.method == 't_test':
                    # perform t-test
                    p_val = survival_did_test(
                        df=_data,
                        gates_col=feature,
                        treatment_col=trt_col,
                        duration_col=duration_col,
                        event_col=event_col,
                        timepoint=self.t
                    )
                    self.pvals[feature] = p_val
                if self.method == 'interaction':
                    # add the interaction term
                    _data[f'interaction_{feature}'] = _data[feature] * _data[trt_col]
                    # fit Cox-model with interaction
                    cox_fitter = CoxPHFitter(penalizer=0.1)
                    cox_fitter.fit(
                        df=_data[reduced_col + [feature, f'interaction_{feature}']],
                        duration_col=duration_col,
                        event_col=event_col,
                    )
                    p_val = cox_fitter.summary.loc[f'interaction_{feature}']['p']
            else:
                if self.method == 'interaction':
                    # add the interaction term
                    _data[f'interaction_{feature}'] = _data[feature] * _data[trt_col]
                    # fit Cox-model with interaction
                    cox_fitter = CoxPHFitter(penalizer=0.1)
                    cox_fitter.fit(
                        df=_data[reduced_col + [feature, f'interaction_{feature}']],
                        duration_col=duration_col,
                        event_col=event_col,
                    )
                    p_val = cox_fitter.summary.loc[f'interaction_{feature}']['p']
                    self.pvals[feature] = p_val
                    self.median[feature] = np.median(_data[feature])
                if self.method == 't_test':
                    # compute splits
                    q_tick = 1 / (self.nb_splits + 1)
                    q_splits = [i * q_tick for i in range(1, self.nb_splits + 1)]
                    splits = data[feature].quantile(q=q_splits).values.T
                    # for each split, compute p_val
                    self.pvals_detailed[feature] = []
                    self.pvals[feature] = 1.
                    for s in splits:
                        # new column with splited var
                        _data[f'g_{feature}'] = (_data[feature] >= s).astype(int)
                        # perform t-test
                        p_val = survival_did_test(
                            df=_data,
                            gates_col=f'g_{feature}',
                            treatment_col=trt_col,
                            duration_col=duration_col,
                            event_col=event_col,
                            timepoint=self.t
                        )
                        self.pvals_detailed[feature].append((s, p_val))
                        if p_val < self.pvals[feature]:
                            self.pvals[feature] = p_val
                            self.median[feature] = np.median(_data[feature])

    def pval_hte(
        self,
        method : str = 'bonferroni',
        save_pvals : bool = False
    ):
        """Output an adjusted pvalue for hte testing.

        Parameters
        ----------
        method : str
            The name of the adjustment method.
            Choices include:
                - 'bonferroni'
                - 'fdr' (-> Benjamini-Hochberg method)
        save_pvals : bool
            Whether to save the pvalues in a dataframe.

        Returns
        -------
        float
            The adjusted pval.
        """
        pvals_array = np.array(list(self.pvals.values()))
        nb_pvals = len(pvals_array)
        if method == 'bonferroni':
            pvals_adj = pvals_array * nb_pvals
            # pvals_adj = np.where(pvals_adj > 1, 1, pvals_adj)
        elif method == 'fdr':
            pvals_adj = fdrcorrection(
                pvals=pvals_array,
                alpha=0.05
            )[1]
        for i, key in enumerate(self.pvals.keys()):
            self.pvals[key] = pvals_adj[i]
        if save_pvals:
            return min(pvals_adj), pd.DataFrame([self.pvals])
        return min(pvals_adj)

    def variables_ranking(self):
        """Output list of covariates ranked according to their pvalues."""
        ordered_dict = dict(sorted(self.pvals.items(), key=lambda item: item[1]))
        ranked_var = list(ordered_dict.keys())
        self.var_to_split = ranked_var[0]  # pylint: disable=W0201
        return ranked_var

    def predict(self, X : pd.DataFrame) -> np.ndarray:
        """Predict subgroups of good/bad responders.

        Parameters
        ----------
        X : pd.DataFrame
            The data under study.

        Returns
        -------
        np.ndarray
            The prediction
        """
        thres_grp = (X[self.var_to_split] > self.median[self.var_to_split]).astype(int)
        return np.array(thres_grp.values)


if __name__ == '__main__':

    np.random.seed(seed=40)

    # fake data
    W = np.random.binomial(1, 0.5, size=1000)
    E = np.random.binomial(1, 0.75, size=1000)
    X1 = np.random.normal(0, 1, size=1000)
    X2 = np.random.normal(0, 1, size=1000)
    X3 = np.random.normal(0, 1, size=1000)
    T = np.random.exponential(1, size=1000)  # * X3 + W
    W = W & (X1 <= 0.0)
    data = pd.DataFrame(
        np.c_[W, E, T, X1, X2, X3],
        columns=['W', 'E', 'T', 'X1', 'X2', 'X3']
    )

    # univar = Univariate(method = 't_test', nb_splits = 4)
    univar = Univariate()
    univar.fit(data, 'T', 'E', 'W')
    print(univar.pvals)

    pval_adj = univar.pval_hte(method='fdr')
    print(pval_adj)

    var_ranking = univar.variables_ranking()
    print(var_ranking)
