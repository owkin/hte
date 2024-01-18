#!/usr/bin/env python
# coding: utf-8

# # A Quick Introduction to HTE

# ### The imports

# In[1]:


import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from hte.data.compute_arr import find_beta
from hte.data.generation import generate_mixture
from hte.viz.plots import KaplanMeierPlots

from hte.models.univariate import Univariate
from hte.models.multivariate import Multivariate
from hte.models.interaction_trees import ITree
from hte.models.model_based import MOB


from hte.configs.data_configs.subgroups import (tutorial,
                                                dim20_pred4_prog0_balanced,
                                                dim20_pred4_prog4_balanced,
                                                dim20_pred4_prog4_bis_balanced,
                                                dim20_pred4_prog2_balanced)

sbg_dict = {
    "tutorial": tutorial,
    "dim20_pred4_prog0_balanced": dim20_pred4_prog0_balanced,
    "dim20_pred4_prog4_balanced": dim20_pred4_prog4_balanced,
    "dim20_pred4_prog4_bis_balanced": dim20_pred4_prog4_bis_balanced,
    "dim20_pred4_prog2_balanced": dim20_pred4_prog2_balanced
}


# ## First step: generate a synthetic dataset so as to control the ARR in two subgroups

# ### The model
# $$ \mathbb{P}(T \geq t \, | \, W, X) = h_0(t) \exp\big( \beta_0 W + (\beta_1 - \beta_0)WG(X) + \gamma^T X  \big) $$
#
# For this demo, we take:
# - X in dimension 20, all entries i.i.d. $\mathcal{N}(0,1)$,
# - W treatment, randomly assigned,
# - Interaction term $ (\beta_1 - \beta_0) W G(X)$, where $G(X)$ takes values in $\{0,1\}$. We will take a multiple condition to belong in the subgroup of good responders
# $$G(X) = (X_{17} > -1) \, \& \, (X_{18} > -1) \, \& \, (X_{19} > -1) \, \& \, (X_{20} > -1)$$

# ## Monte-Carlo estimation of the ARR

# We can generate and save the value of the ARR in function of the subgroup and of the value of $\beta$. In a nutshell, the pipeline consists in:
# - Fix all parameters ($h_0$, law of $X$, prognostic coefficient $\gamma$, etc),
# - Take a range of values of $\beta$s,
# - For each $\beta$, generate data from the two arms: $\mathbb{P}(T \geq t \, | \, W = 0, X) = h_0(t) \exp\big( \beta W + \gamma^T X  \big)$ and $\mathbb{P}(T \geq t \, | \, W = 1, X) = h_0(t) \exp\big( \beta W + \gamma^T X  \big)$
# - Compute the individual treatment effects
# - Averaged things out and compute the ARRs in subgroup 0 and in subgroup 1
# - Save all results in a json
#
# Let's have a look.

# In[2]:


path = "Cox_Weibull_1.0_2.0_dim=20_range=[-10.0,10.0]_nb=500_group='dim20_pred4_prog0_balanced'_July_07_12_2023_15:15:24.json"
#path = "Cox_Weibull_1.0_2.0_dim=20_range=[-10.0,10.0]_nb=500_group='dim20_pred4_prog4_bis_balanced'_July_07_19_2023_10:28:22.json"

wd = os.getcwd()
with open("../hte/data/results_compute_arr/{}".format(path), "r") as json_file:
    dict_param = json.load(json_file)
dict_param0 = dict_param['arr(beta0)']
dict_param1 = dict_param['arr(beta1)']

plt.figure(figsize=(15,5))
plt.plot(
    [float(elem) for elem in dict_param0.keys()],
    list(dict_param0.values()),
    label = 'Group 0'
)
plt.plot(
    [float(elem) for elem in dict_param1.keys()],
    list(dict_param1.values()),
    label = 'Group 1'
)
plt.xlabel('beta')
plt.ylabel('ARR')
plt.legend(loc = 'best')
plt.show()


# ## Generate synthetic data for a fixed ARR difference

# ### Parameters retrieval

# Say we want the ARR to be equal to 0.3 in the subgroup of good responders, and equal to -0.3 in the other group. We can use the previous json together with the function find beta.

# In[9]:


beta0 = find_beta(-0.2, dict_param0)
beta1 = find_beta(0.2, dict_param1)


# Then, we take all other parameters of the dgp and generate some data.

# In[10]:


mean = np.array(dict_param['param']['mean'])
cov = np.array(dict_param['param']['cov'])
model = dict_param['param']['model']
base_hazard = dict_param['param']['base_hazard']
a = dict_param['param']['a']
b = dict_param['param']['b']
gamma = np.array(dict_param['param']['gamma'])
group_name = dict_param['param']['group_name']
group = sbg_dict[group_name]


# ### Generation

# In[11]:


df_train = generate_mixture(
    beta0=beta0,
    beta1=beta1,
    size=1000,
    group=group,
    mean=mean,
    cov=cov,
    gamma=gamma,
    model=model,
    base_hazard=base_hazard,
    a=a,
    b=b,
    censored=False,
    scale=1.
)

df_test = generate_mixture(
    beta0=beta0,
    beta1=beta1,
    size=1000,
    group=group,
    mean=mean,
    cov=cov,
    gamma=gamma,
    model=model,
    base_hazard=base_hazard,
    a=a,
    b=b,
    censored=False,
    scale=1.
)


# ### Visualization

# Lets' have a look at the KM plots.

# In[12]:


km = KaplanMeierPlots()
km.plot(
    df=df_train,
    title='Good responders'
)


# In[13]:


km = KaplanMeierPlots()
km.plot(
    df=df_train[df_train['G']==1],
    title='Good responders'
)


# In[14]:


km = KaplanMeierPlots()
km.plot(
    df=df_train[df_train['G']==0],
    title='Bad responders'
)


# ### Subgroup analysis methods

# Let's finally train a model, say Interaction Tree.

# In[15]:


df_train_ = df_train.drop(columns=['G']).copy()
df_test_ = df_test.drop(columns=['G']).copy()


# All methods have:
# - A fit,
# - A pval_hte (p value of test detection presence of heterogeneity)
# - A variables_ranking (which variables explain heterogeneity?)
# - A predict (what are the good responders?)

# In[27]:


multi = Multivariate()


# In[28]:


multi.fit(
    data=df_train_,
    duration_col='T',
    event_col='E',
    trt_col='W'
)


# In[29]:


multi.pval_hte(df_test)


# In[30]:


multi.variables_ranking()


# In[31]:


df_test['pred_G'] = multi.predict(df_test_)
(df_test['G'] == df_test['pred_G']).mean()


# Let's have a look at the interaction tree method

# In[21]:


int_tree = ITree(
    alpha=0.5,
    nb_splits=3,
    min_nb_samples=0.1
)


# In[22]:


int_tree.fit(
    data=df_train_,
    duration_col='T',
    event_col='E',
    trt_col='W'
)


# In[23]:


int_tree.pval_hte(df_test_)


# In[24]:


int_tree.variables_ranking()


# In[25]:


df_test['pred_G'] = int_tree.predict(df_test_)
(df_test['G'] == df_test['pred_G']).mean()


# In[26]:


int_tree.rules


# In[16]:


mob = MOB(
    alpha=0.5,
    nb_splits=5,
    min_nb_samples=0.1
)


# In[17]:


mob.fit(
    data=df_train_,
    duration_col='T',
    event_col='E',
    trt_col='W'
)


# In[18]:


mob.pval_hte(df_test_)


# In[19]:


mob.variables_ranking()


# In[20]:


df_test['pred_G'] = mob.predict(df_test_)
(df_test['G'] == df_test['pred_G']).mean()


# In[ ]:
