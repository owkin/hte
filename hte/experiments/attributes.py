"""Define methods parameters for benchmarking."""

attributes = {  # pylint: disable=R6101
    'Oracle': {
        'module_path': 'hte.models.univariate',
        'module_class': 'Univariate',
        'config': {
            'timepoint': 1.,
            'method': 't_test'
        },
        'predictive_method': False
    },
    'Univariate interaction': {
        'module_path': 'hte.models.univariate',
        'module_class': 'Univariate',
        'config': {
            'timepoint': 1.,
            'method': 'interaction',
            'saving_pvalues': True
        },
        'predictive_method': False,
    },
    'Univariate t_test': {
        'module_path': 'hte.models.univariate',
        'module_class': 'Univariate',
        'config': {
            'timepoint': 1.,
            'method': 't_test',
            'saving_pvalues': True
        },
        'predictive_method': False,
    },
    'Multivariate cox': {
        'module_path': 'hte.models.multivariate',
        'module_class': 'Multivariate',
        'config': {
            'timepoint': 1.,
            'estimator': 'CoxLinear',
            'interactions': True
        },
        'predictive_method': True
    },
    'Multivariate tree': {
        'module_path': 'hte.models.multivariate',
        'module_class': 'Multivariate',
        'config': {
            'timepoint': 1.,
            'estimator': 'SurvivalTree',
            'interactions': True
        },
        'predictive_method': True
    },
    'MOB': {
        'module_path': 'hte.models.model_based',
        'module_class': 'MOB',
        'config': {
            'timepoint': 1.,
        },
        'predictive_method': False
    },
    'ITree': {
        'module_path': 'hte.models.interaction_trees',
        'module_class': 'ITree',
        'config': {
            'timepoint': 1.,
        },
        'predictive_method': False
    },
    'SIDES': {
        'module_path': 'hte.models.sides',
        'module_class': 'SIDES',
        'config': {
            'timepoint': 1.,
        },
        'predictive_method': True
    },
    'SeqBT': {
        'module_path': 'hte.models.sequential_batting',
        'module_class': 'SeqBT',
        'config': {
            'timepoint': 1.
        },
        'predictive_method': True
    },
    'ARDP': {
        'module_path': 'hte.models.directed_peeling',
        'module_class': 'ARDP',
        'config': {
            'timepoint': 1.
        },
        'predictive_method': True
    }
}
