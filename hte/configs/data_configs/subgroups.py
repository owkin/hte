"""Define subgroups functions and subgroup dictionaries."""


def tutorial(x):
    """Define a tutorial subgroup function."""
    cdt1 = x[0] >= -1.
    cdt2 = x[1] >= -1.
    cdt = cdt1 & cdt2
    return int(cdt)


def dim20_pred4_prog0_balanced(x):
    """Define a subgroup function for dim=20, 4 pred variables, 0 prog variable."""
    cdt1 = x[16] >= -1.
    cdt2 = x[17] >= -1.
    cdt3 = x[18] >= -1.
    cdt4 = x[19] >= -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4)


def dim20_pred4_prog1_balanced(x):
    """Define a subgroup function for dim=20, 4 pred variables, 1 prog variable."""
    cdt1 = x[8] >= -1.
    cdt2 = x[12] >= -1.
    cdt3 = x[13] >= -1.
    cdt4 = x[14] >= -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4)


def dim20_pred4_prog2_balanced(x):
    """Define a subgroup function for dim=20, 4 pred variables, 2 prog variables."""
    cdt1 = x[8] >= -1.
    cdt2 = x[9] >= -1.
    cdt3 = x[10] >= -1.
    cdt4 = x[11] >= -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4)


def dim20_pred4_prog3_balanced(x):
    """Define a subgroup function for dim=20, 4 pred variables, 3 prog variables."""
    cdt1 = x[5] >= -1.
    cdt2 = x[6] >= -1.
    cdt3 = x[7] >= -1.
    cdt4 = x[19] >= -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4)


def dim20_pred4_prog4_balanced(x):
    """Define a subgroup function for dim=20, 4 pred variables, 4 prog variables."""
    cdt1 = x[0] >= -1.
    cdt2 = x[1] >= -1.
    cdt3 = x[6] >= -1.
    cdt4 = x[7] >= -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4)


def dim20_pred4_prog4_bis_balanced(x):
    """Define a subgroup function for dim=20, 4 pred variables, 4 prog variables."""
    cdt1 = x[6] >= -1.
    cdt2 = x[7] >= -1.
    cdt3 = x[8] >= -1.
    cdt4 = x[9] >= -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4)


def dim100_pred3_prog2_balanced(x):
    """Define a subgroup function for dim=100, 3 pred variables, 2 prog variables."""
    cdt1 = x[1] >= -1.
    cdt2 = x[6] >= -1.
    cdt3 = x[90] >= 1.
    return int(cdt1 & cdt2 & cdt3)


def dim100_pred4_prog4_balanced(x):
    """Define a subgroup function for dim=100, 4 pred variables, 4 prog variables."""
    cdt1 = x[0] >= -1.
    cdt2 = x[1] >= -1.
    cdt3 = x[2] >= -1.
    cdt4 = x[3] >= -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4)


def dim100_pred4_prog0_balanced(x):
    """Define a subgroup function for dim=100, 4 pred variables, 0 prog variable."""
    cdt1 = x[40] >= -1.
    cdt2 = x[60] >= -1.
    cdt3 = x[80] >= -1.
    cdt4 = x[90] >= -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4)


def dim100_pred4_prog2_balanced(x):
    """Define a subgroup function for dim=100, 4 pred variables, 2 prog variables."""
    cdt1 = x[2] >= -1.
    cdt2 = x[3] >= -1.
    cdt3 = x[80] >= -1.
    cdt4 = x[90] >= -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4)


def dim1000_pred4_prog0_balanced(x):
    """Define a subgroup function for dim=1000, 4 pred variables, 0 prog variable."""
    cdt1 = x[996] >= -1.
    cdt2 = x[997] >= -1.
    cdt3 = x[998] >= -1.
    cdt4 = x[999] >= -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4)


def semi_synth_sbg(x):
    """Define a subgroup function for semi-synthetic data with 230/503 in sgb 1."""
    cdt1 = x[0] > -1.
    cdt2 = x[1] > -1.
    cdt3 = x[2] > -1.
    cdt4 = x[4] > -1.
    cdt5 = x[5] > -1.
    cdt6 = x[6] > -1.
    return int(cdt1 & cdt2 & cdt3 & cdt4 & cdt5 & cdt6)


def semi_synth_sbg_1000(x):
    """Define a subgroup function for semi-synthetic data with 248/503 in sgb 1."""
    cdt1 = x[0] > -1.1
    cdt2 = x[2] > -1.1
    cdt3 = x[6] > -1.1
    cdt4 = x[7] > -1.1
    cdt5 = x[10] > -1.1
    cdt6 = x[11] > -1.1
    return int(cdt1 & cdt2 & cdt3 & cdt4 & cdt5 & cdt6)


# This dict provides the index of predictive variables for each subgroup.
subgroup_index_dict = {
    'dim20_pred4_prog0_balanced': [16, 17, 18, 19],
    'dim20_pred4_prog1_balanced': [8, 12, 13, 14],
    'dim20_pred4_prog2_balanced': [8, 9, 10, 11],
    'dim20_pred4_prog3_balanced': [5, 6, 7, 19],
    'dim20_pred4_prog4_balanced': [0, 1, 6, 7],
    'dim20_pred4_prog4_bis_balanced': [6, 7, 8, 9],
    'dim100_pred3_prog2_balanced': [1, 6, 90],
    'dim100_pred4_prog4_balanced': [0, 1, 2, 3],
    'dim100_pred4_prog0_balanced': [40, 60, 80, 90],
    'dim100_pred4_prog2_balanced': [2, 3, 80, 90],
    'dim1000_pred4_prog0_balanced': [996, 997, 998, 999],
    'semi_synth_sbg': [0, 1, 2, 4, 5, 6],
    'semi_synth_sbg_1000': [0, 2, 6, 7, 10, 11],
    'tutorial': [0, 1],
}

# This dict provides a link between subgroup name (str) and subgroup function.
sbg_dict = {
    "tutorial": tutorial,
    "dim20_pred4_prog0_balanced": dim20_pred4_prog0_balanced,
    "dim20_pred4_prog1_balanced": dim20_pred4_prog1_balanced,
    "dim20_pred4_prog3_balanced": dim20_pred4_prog3_balanced,
    "dim20_pred4_prog4_balanced": dim20_pred4_prog4_balanced,
    "dim20_pred4_prog4_bis_balanced": dim20_pred4_prog4_bis_balanced,
    "dim20_pred4_prog2_balanced": dim20_pred4_prog2_balanced,
    "dim100_pred3_prog2_balanced": dim100_pred3_prog2_balanced,
    "dim100_pred4_prog4_balanced": dim100_pred4_prog4_balanced,
    "dim100_pred4_prog0_balanced": dim100_pred4_prog0_balanced,
    "dim100_pred4_prog2_balanced": dim100_pred4_prog2_balanced,
    "dim1000_pred4_prog0_balanced": dim1000_pred4_prog0_balanced,
    'semi_synth_sbg_1000': semi_synth_sbg_1000
}
