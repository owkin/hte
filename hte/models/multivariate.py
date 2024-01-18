"""Multivariate analysis."""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sksurv.tree import SurvivalTree
from hte.utils.utils import survival_did_test


class Multivariate():
    """A class to perform subgroup identification based on CATE estimation.

    In this first version, we restrict the scope to S-learner.
    """

    def __init__(
        self,
        estimator : str = 'CoxLinear',
        interactions : bool = True,
        timepoint : int = 1,
        random_state : int = 42,
    ):
        """Instantiate object with hyper-parameters.

        Parameters
        ----------
        estimator : str
            The estimator used for the S-learner.
            Choices include "CoxLinear".
            @TODO: add xgboost

        interactions : bool
            Whether interactions terms X * W will be added
            to the S-learner inputs.

        timepoint : int
            Timepoint at which to evaluate
            the treatment effect

        random_state : int
            Random state for reproducibility.
        """
        self.estimator = estimator
        self.interactions = interactions
        self.t = timepoint
        self.drop_cols = []
        self.features = []
        self.features_baseline = []
        self.features_interactions = []
        self.dict_standard = {}
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=self.random_state)
        self.fitter = None

    def fit(
        self,
        data : pd.DataFrame,
        duration_col : str,
        event_col : str,
        trt_col : str,
        features : list[str] = None
    ):
        """Fit the S-learner to the data.

        Parameters
        ----------
        data : pd.Dataframe
            Contains several columns:
            - duration,
            - event,
            - treatment,
            - all remaining are the covariates.

        duration_col : str
            Name of the duration column.

        event_col : str
            Name of the event column.

        trt_col : str
            Name of the treatment allocation column.

        features : list[str]
            List of features to consider
            in the multivariate model.
        """
        self.dict_standard['W'] = trt_col
        self.dict_standard['T'] = duration_col
        self.dict_standard['E'] = event_col

        _data = data.copy()
        reduced_col = [duration_col, event_col, trt_col]
        self.drop_cols = reduced_col
        if features is None:
            self.features = []
            for elem in data.columns:
                if elem not in reduced_col:
                    self.features.append(elem)
        self.features_baseline = self.features.copy()

        # if interactions is True, add the columns
        if self.interactions:
            self.features_interactions = [f'{f}_x_{trt_col}' for f in self.features]
            _data[self.features_interactions] = _data[self.features].values \
                * _data[[trt_col]].values
            self.features = self.features + self.features_interactions

        # train the model
        if self.estimator == 'CoxLinear':
            self.fitter = CoxPHFitter(penalizer=0.1)  # pylint: disable=W0201
            self.fitter = self.fitter.fit(   # pylint: disable=W0201
                df=_data,
                duration_col=duration_col,
                event_col=event_col
            )
        if self.estimator == 'SurvivalTree':
            _data_surv = _data.drop(columns=['T', 'E'])
            _y_surv = _data[['E', 'T']].apply(
                lambda row: (bool(row['E']), row['T']), axis=1
            ).values
            _y_surv = np.array(_y_surv, dtype=[('E', bool), ('T', float)])
            self.fitter = SurvivalTree(  # pylint: disable=R0204
                splitter='best',
                min_samples_leaf=0.05,
            )
            self.fitter = self.fitter.fit(
                X=_data_surv,
                y=_y_surv
            )

    def predict_iarr(
        self,
        data : pd.DataFrame
    ) -> np.array:
        """Predict good (1) or bad (0) responder for a given X.

        Parameters
        ----------
        data : pd.Dataframe
            It must contain:
            - covariates column (only)
            (column names must match those of training).

        Returns
        -------
        np.array
            The predictions made by multivariate model.
        """
        X = data.copy()

        if self.estimator == 'SurvivalTree':
            idx = np.searchsorted(self.fitter.unique_times_, self.t)
            # proxy_t = self.fitter.unique_times_[idx]
            # if proxy_t >= self.t:
            if self.fitter.unique_times_[idx] >= self.t:
                # if (proxy_t := self.fitter.unique_times_[idx]) >= self.t:
                idx_ = idx - 1
            else:
                idx_ = idx + 1

        # Predict under treatment
        X['W'] = 1
        if self.interactions:
            X[self.features_interactions] = X[self.features_baseline]
            if self.estimator == 'CoxLinear':
                y_1 = self.fitter.predict_survival_function(X,
                                                            times=(self.t,)).loc[self.t]
            if self.estimator == 'SurvivalTree':
                for feat in self.features_interactions:
                    X[feat] = X[feat.split("_")[0]]
                cols = list(X.columns)
                cols.remove('W')
                cols.insert(0, 'W')
                X = X[cols]
                preds = self.fitter.predict_survival_function(X, return_array=True)
                y_1 = (preds[:, idx] + preds[:, idx_]) / 2
        # predict under control
        X['W'] = 0
        if self.interactions:
            X[self.features_interactions] = 0
            if self.estimator == 'CoxLinear':
                y_0 = self.fitter.predict_survival_function(X,
                                                            times=(self.t,)).loc[self.t]
            if self.estimator == 'SurvivalTree':
                for feat in self.features_interactions:
                    X[feat] = X[feat.split("_")[0]]
                cols = list(X.columns)
                cols.remove('W')
                cols.insert(0, 'W')
                X = X[cols]
                preds = self.fitter.predict_survival_function(X, return_array=True)
                y_0 = (preds[:, idx] + preds[:, idx_]) / 2

        # estimate the ARR and threshold it
        iarr_est = y_1 - y_0
        return iarr_est

    def predict(
        self,
        data : pd.DataFrame,
        threshold : float = 0.
    ) -> np.array:
        """Predict good (1) or bad (0) responder for a given X.

        Parameters
        ----------
        data : pd.Dataframe
            It must contain:
            - covariates column (only)
            (column names must match those of training).

        threshold : float
            The ARR threshold to create the two groups.

        Returns
        -------
        np.array
            The predictions made by multivariate model.
        """
        X = data.copy()
        X = X.drop(self.drop_cols, axis=1)
        iarr_est = self.predict_iarr(X)
        if self.estimator == 'CoxLinear':
            thres_arr_est = (iarr_est > threshold).astype(int)
            thres_arr_est = np.array(thres_arr_est.values)
            return thres_arr_est
        if self.estimator == 'SurvivalTree':
            thres_arr_est = (iarr_est > threshold).astype(int)
            return thres_arr_est
        return np.empty(shape=len(iarr_est), dtype=int)

    def pval_hte(
        self,
        data : pd.DataFrame,
        threshold : float = 0.
    ):
        """Output an adjusted pvalue for hte testing.

        Parameters
        ----------
        data : pd.DataFrame
            The new data to predict subgroups
            and treatment effect.

        threshold : float
            The ARR threshold to create the two groups.

        Returns
        -------
        p_val: float
            P-value testing for treatment effect
        """
        _data = data.copy()
        X = _data

        # Predict subgroup
        group_est = self.predict(data=X,
                                 threshold=threshold)

        # Check if all elements of group_est are identical
        if (group_est == group_est[0]).all():
            rng = np.random.default_rng(seed=self.random_state)
            p_val = rng.uniform(0, 1)
        else:

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

    def variables_ranking(self):
        """Output list of covariates ranked according to their pvalues."""
        if self.estimator == 'CoxLinear':
            if not self.interactions:
                return 'No ranking when no interactions'

            pvals_df = self.fitter.summary['p']\
                .loc[self.features_interactions].sort_values()
            ranked_var = list(pvals_df.index)
            ranked_var_new = []
            for elem in ranked_var:
                ranked_var_new.append(elem.split('_')[0])
            return ranked_var_new
        # if self.estimator == 'SurvivalTree':
        return ['']


if __name__ == '__main__':

    np.random.seed(seed=40)

    # fake data
    SIZE_TRAIN = 100
    W = np.random.binomial(1, 0.5, size=SIZE_TRAIN)
    E = np.random.binomial(1, 0.75, size=SIZE_TRAIN)
    X1 = np.random.normal(0, 1, size=SIZE_TRAIN)
    X2 = np.random.normal(0, 1, size=SIZE_TRAIN)
    X3 = np.random.normal(0, 1, size=SIZE_TRAIN)
    T = np.random.exponential(1, size=SIZE_TRAIN) * X1**2 * X2**2

    data = pd.DataFrame(
        np.c_[W, E, T, X1, X2, X3],
        columns=['W', 'E', 'T', 'X1', 'X2', 'X3']
    )

    cate = Multivariate(interactions=True, estimator='SurvivalTree')
    cate.fit(data, 'T', 'E', 'W')

    X = pd.DataFrame(
        np.c_[X1, X2, X3],
        columns=['X1', 'X2', 'X3']
    )

    print(cate.estimator)

    output = cate.predict(X)
    print(output)

    pval = cate.pval_hte(data)
    print(pval)

    output = cate.variables_ranking()
    print(output)
