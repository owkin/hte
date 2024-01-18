"""Process and analyze the results of the experiments."""  # noqa: DAR000

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score

from hte.configs.data_configs.subgroups import subgroup_index_dict


def set_ground_truths(dim: int, subgroup: str):
    """Set the ground truths for the experiment analysis.

    Parameters
    ----------
    dim : int
        The dimension of the data.
    subgroup : str
        The subgroup name.

    Returns
    -------
    range_features : range
        The range of features.
    features : set
        The set of features.
    pred_var : set
        The set of predictive variables.
    y_true : np.array
        The ground truth.
    """
    range_features = range(1, dim + 1)
    features = set(range(1, dim + 1))
    pred_var = {'X' + str(i + 1): i for i in subgroup_index_dict[subgroup]}
    y_true = np.zeros(dim)
    y_true[subgroup_index_dict[subgroup]] = 1
    return range_features, features, pred_var, y_true


class ProcessResults():
    """A class to process the .csv results."""

    def __init__(self, pdim : int, subgroup : str):
        self.dim = pdim
        self.subgroup_name = subgroup
        self.range, self.feat, self.pred_var, self.y_true = set_ground_truths(
            self.dim,
            self.subgroup_name)

    def process(self, df : pd.DataFrame):
        """Process the results to answer the research questions.

        Returns
        -------
        df : pd.DataFrame
            A dataframe containing processed results.
        """
        df['ranked_var'].fillna("['']", inplace=True)
        df = df.drop(labels=df[df['pval'] == 'NOT FITTED']['method'].index)
        df['pval'] = df['pval'].apply(lambda x: float(x) if isinstance(x, str) else x)
        df['Accuracy'] = df['Accuracy'] \
            .apply(lambda x: float(x) if isinstance(x, str) else x)
        df['TP'] = df['TP'].apply(lambda x: float(x) if isinstance(x, str) else x)
        df['TN'] = df['TN'].apply(lambda x: float(x) if isinstance(x, str) else x)
        df['FP'] = df['FP'].apply(lambda x: float(x) if isinstance(x, str) else x)
        df['FN'] = df['FN'].apply(lambda x: float(x) if isinstance(x, str) else x)

        # df['ranked_var'] = df['ranked_var'].apply(lambda x: self.add_brackets(x))
        # df['ranked_var'] = df['ranked_var'].apply(lambda x: self.into_list(x))
        df['ranked_var'] = df['ranked_var'].apply(self.add_brackets)
        df['ranked_var'] = df['ranked_var'].apply(self.into_list)

        df['pval'] = df.apply(lambda row: row['pval'] * self.dim * 3
                              and ((len(row['ranked_var']) == 1))
                              if (row['method'] == 'SIDES') else row['pval'],
                              axis=1)
        df['pval'] = df['pval'].apply(lambda x: np.clip(x, 0, 1))
        df['thresh_pval'] = df['pval'].apply(lambda x: int(x < 0.05))
        df.drop(columns=['trainsize',
                         'testsize',
                         'beta0',
                         'beta1',
                         'mean',
                         'cov',
                         'model',
                         'base_hazard',
                         'a',
                         'b',
                         'file_arr(beta)'],
                inplace=True)

        df = self.rankvar(df)

        return df

    def rankvar(self, df : pd.DataFrame):
        """Rank the variables according to the pvalues.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the results.

        Returns
        -------
        df : pd.DataFrame
            The dataframe with the ranked variables.
        """
        # define the top var selected
        var_selected_columns = []
        for feat in [f"X{i}" for i in self.range]:
            # for i in self.range:
            # feat = f"X{i}"
            df[f"selected_{feat}"] = df['ranked_var'].apply(
                lambda x, feat=feat: int(x[0] == feat))
        # for i in self.range:
            # feat = f"X{i}"
            # df[f"selected_{feat}"] = df['ranked_var'] \
            # .apply(lambda x: int(x[0] == feat))
            var_selected_columns.append(f"selected_{feat}")
        # df['top_var_selected'] = df.apply(lambda row: self.top_var_selected(row),
        #                                  axis=1)
        df['top_var_selected'] = df.apply(self.top_var_selected,
                                          axis=1)
        # complete the ranking
        df['ranked_var_int'] = df['ranked_var'].apply(
            lambda x: [int(elem[1:]) for elem in x if x[0] != ''])
        df['completed_ranking'] = df['ranked_var_int'].apply(
            lambda x: self.complete_ranking(x, self.feat))
        # put in prob format and compute area under curves (ROC and PR)
        df['prob_predictive'] = df['completed_ranking'].apply(
            # lambda x: self.norm_ranking(x))
            self.norm_ranking)
        df['auc_score'] = df['prob_predictive'].apply(
            lambda x: roc_auc_score(self.y_true, x))
        df['averaged_precision_score'] = df['prob_predictive'].apply(
            lambda x: average_precision_score(self.y_true, x))
        return df

    def add_brackets(self, x : str):
        """Add brackets to the string.

        Parameters
        ----------
        x : str
            The string to modify.

        Returns
        -------
        x : str
            The modified string.
        """
        if isinstance(x, list):
            x = str(x)
        elif x[0] != '[':
            x = "['" + x + "']"
        return x

    def into_list(self, x : str):
        """Transform the string into a list.

        Parameters
        ----------
        x : str
            The string to modify.

        Returns
        -------
        output : list
            The modified string.
        """
        cleaned_string = x.strip("[]")
        output = [elem[1:-1] for elem in cleaned_string.split(', ')]
        return output

    def top_var_selected(self, row : pd.Series):
        """Find the top variable is selected.

        Parameters
        ----------
        row : pd.Series
            A row of the dataframe.

        Returns
        -------
        output : int
            The top variable selected for this row.
        """
        predictive_columns = [f"selected_{feat}" for feat in self.pred_var]
        output = 0
        for s in predictive_columns:
            output += row[s]
        return output

    def complete_ranking(self, x : list, features : set):
        """Fill up the ranking with nonranked features.

        Parameters
        ----------
        x : list
            The list of ranked features.

        features : set
            The set of all features to be compared to ranked.

        Returns
        -------
        output : list
            The completed ranking.
        """
        delta_set = np.array(list(features.difference(set(x))))
        output = x + list(np.random.permutation(delta_set))
        return output

    def norm_ranking(self, x):
        """Normalize the ranking."""
        return 1 - (np.argsort(x) + 1) / len(x)
