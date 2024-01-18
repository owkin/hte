"""Inspect compte_arr results."""

import os
import json
import matplotlib.pyplot as plt


path = "Cox_Weibull_1.0_2.0_dim=3_range=[-5.0,5.0]_\
    nb=100__July_07_06_2023_09:16:48.json"

wd = os.getcwd()
with open("{}".format(path), "r") as json_file:
    dict_param = json.load(json_file)
dict_param0 = dict_param['arr(beta0)']
dict_param1 = dict_param['arr(beta1)']

plt.figure(figsize=(8, 5))
plt.plot(
    [float(elem) for elem in dict_param0.keys()],
    list(dict_param0.values()),
    label='Group 0'
)
plt.plot(
    [float(elem) for elem in dict_param1.keys()],
    list(dict_param1.values()),
    label='Group 1'
)
plt.xlabel('beta')
plt.ylabel('ARR')
plt.legend(loc='best')
plt.show()
