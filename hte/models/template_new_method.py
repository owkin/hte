"""Sample file to explain how to add a new method to benchmark."""

# Place here the necessary imports.
import pandas as pd

# The method should be a class.

class new_method:
    """New method class.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Functions
    ---------
    Should contain at least the following functions:
    fit()
        Fits the model to the data.
    predict()
        Predicts the good/bad responders group assignment (0 or 1).
    pval_hte()
        Returns the p-value corresponding to the model estimation
        of heterogeneity existence.
    variables_ranking()
        Returns a ranking of variables based on
        their estimated contribution to heterogeneity.

    """

    def __init__(
        self,
        param1,
        param2
    ):
        self.param1 = param1
        self.param2 = param2


    def fit(self,
            data : pd.DataFrame,
            duration_col : str,
            event_col : str,
            trt_col : str):
        """Fit the model to obtain the good responder subgroup rule: self.rule.

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
        # Write fit function here.

    def predict(self,
                data : pd.DataFrame):

        """Predict in which group a new sample falls.

        Parameters
        ----------
        data : pd.Dataframe
            Contains the data to be predicted.
            Should contain at least the following columns:
            Duration column, event column, treatment column,
            named as for the data used in the fitting step.

        Returns
        -------
        array of group indicators
        """
        # Write predict function here.

    def pval_hte(self):
        """Compute the p-value for heterogeneity of treatment effect.

        Parameters
        ----------
        data : pd.Dataframe
            Only if the new method is a predictive method.
            Contains the data to be predicted using predict function.
            Should contain at least the following columns:
            Duration column, event column, treatment column

        Returns
        -------
        p_val : float
            p-value testing for treatment effect
        """

    def variables_ranking(self):
        """Rank the variables by importance in HTE.

        Returns
        -------
        List of ranked variable names
        """
