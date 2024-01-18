"""Generate json of dgp's parameters."""

import os
import json
import numpy as np

np.random.seed(seed=42)

# Variance Covariance structure of the covariates X.
dim = 1000  # pylint: disable=C0103
mean = np.zeros(dim)
cov = np.diag(np.ones(dim))

# The prognostic vector gamma
gamma = np.array([1.] * 10 + [-1.] * 10 + [0.1] * 50 + [-0.1] * 50
                 + [0.01] * 200 + [-0.01] * 200 + [0.] * 480)

# Put in a dictionnary
dict_params = {
    "mean": mean.tolist(),
    "cov": cov.tolist(),
    "gamma": gamma.tolist()
}

# Give a name to the json
file_name = 'dim1000_isotropic_pro(+)10_pro(-)10_noise480'  # pylint: disable=C0103

# Save as json
if os.path.exists(f"./{file_name}.json"):
    print('The json already exists')
else:
    json_obj = json.dumps(dict_params, indent=4)
    with open(f"./{file_name}.json", "w", encoding="utf8") as outfile:
        outfile.write(json_obj)
