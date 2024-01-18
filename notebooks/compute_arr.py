#!/usr/bin/env python
# coding: utf-8

# <h1> Compute ARR functionnalities </h1>

# What you'll find in this notebook:
# <ul>
#     <li> The <code>compute_beta_arr</code> function, which computes the values of the ARR from a given range of values of $\beta$, </li>
#     <li> The <code>splicing</code> and <code>sanity_check_beta_arrs</code> functions, which allow to check the monotonicity of the estimated ARR function, and to smooth potential errors, </li>
#     <li> The <code>find_beta</code> function, which finds the correct value of $\beta$ achieving a pre-specified ARR level.</li>
# </ul>
#
# Let's go!

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from hte.data.compute_arr import compute_beta_arr, splicing, sanity_check_beta_arrs, find_beta


# <h2> Computing the curve $(\beta, ARR(\beta))$ </h2>

# The understanding of this notebook require the reading of the <b>data_generation</b> notebook, in which the data genarating process is explained.
#
# Using the same notations, the function compute_beta_arr takes as input:
#
# <ul>
#     <li> The range of values for the $\beta$ parameter, through the specification of
#         <ul>
#             <li> The number of points, nb_points, </li>
#             <li> The min value, minval, </li>
#             <li> The max value, maxval, </li>
#         </ul>
#     </li>
#     <li> The number of sample used for the Monte-Carlo estimation of $ARR(\beta)$ for each value of $\beta$, </li>
#     <li> The parameters of the law of X (mean, covariance, baseline hazard, model, censorship) </li>
#     <li> The function <b>group</b> that defines the two subgroups, </li>
#     <li> The time horizon at which the ARR is computed, </li>
#     <li> A boolean 'save' parameter to store the output dictionnary in a json format, </li>
# </ul>
#
# It outputs a dictionnary containing several informations, in four keys:
# <ul>
#     <li> 'param': contains all the information of the parameters of the function: law of $X$, $\gamma$, Monte-Carlo size, group function etc. The function is stored as a string, using the inspect.getsource method, </li>
#     <li> 'prop0': the proportion of samples in subgroup 0, </li>
#     <li> 'arr(beta0)': a dict with keys given by the values of beta and values given by the corresponding ARRs, for samples falling in subgroup 0, </li>
#     <li> 'arr(beta1)': a dict with keys given by the values of beta and values given by the corresponding ARRs, for samples falling in subgroup 1. </li>
# <ul>

# In[2]:


def group(x):
    return int(x[0] > 0.0)

output = compute_beta_arr(
    nb_points = 50,
    minval = -5.,
    maxval = 5.,
    size = int(1e5),
    group = group,
    mean = np.array([0, 0]),
    cov = np.array([[1, 0], [0, 1]]),
    gamma = np.array([0.1, 1.]),
    model = 'Cox',
    base_hazard = 'Weibull',
    a = 1.0,
    b = 2.0,
    horizon = 1.,
    save = False
)


# In[3]:


output.keys()


# In[4]:


print(output['param']['transform'])


# <h2> Use the splicing function to smooth estimates and to order betas </h2>
#
# The functions arr(beta) are stored in json format which does not ensure the beta values are increasing as expected. </br>
#
# Moreover, since we are relying on Monte-Carlo estimation, there could be some variance resulting in situations where $\hat{ARR}(\beta + \varepsilon) > \hat{ARR}(\beta)$ while we know for sure that the opposite inequality must hold for the real quantities.
#
# The function <b>splicing</b> is designed to resolve the two above issues. It first sort the beta keys in increasing order and then delete entries that break the theoretical inequality constraints.

# In[5]:


arr_beta0 = output['arr(beta0)']
arr_beta0_spliced = splicing(arr_beta0)

arr_beta1 = output['arr(beta1)']
arr_beta1_spliced = splicing(arr_beta1)


# Once this is done, we can sanity check the monotonicity constraint with the sanity_check_beta_arrs function:

# In[6]:


sanity_check_beta_arrs(arr_beta0), sanity_check_beta_arrs(arr_beta1)


# In[7]:


sanity_check_beta_arrs(arr_beta0_spliced), sanity_check_beta_arrs(arr_beta1_spliced)


# Let us visualize the relation between $\beta$ and the ARR.

# In[8]:


plt.figure(figsize=(8,5))
plt.plot(
    [float(elem) for elem in arr_beta0_spliced.keys()],
    list(arr_beta0_spliced.values()),
    label = 'Group 0'
)
plt.plot(
    [float(elem) for elem in arr_beta1_spliced.keys()],
    list(arr_beta1_spliced.values()),
    label = 'Group 1'
)
plt.title('ARR as a function of beta, depending on the subgroup')
plt.xlabel('beta')
plt.ylabel('ARR')
plt.legend(loc = 'best')
plt.show()


# <h2> Finding the right $\beta_0$ and $\beta_1$ using the find_beta function </h2>

# For a given value of ARR, what is the corresponding value of $\beta_0$ and $\beta_1$ needed to achieve this ARR in subgroup 0 and subroup 1? The answers is given by the fint_beta function.

# In[11]:


output = find_beta(
    arr = 0.3,
    dict_param = arr_beta0_spliced
)
print(output)
print(arr_beta0[output])
