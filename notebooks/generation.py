#!/usr/bin/env python
# coding: utf-8

# <h1> Data generation </h1>

# ## Data

# What you'll find in this notebook:
# <ul>
#     <li> The <code>generate_mixture</code> function, which outputs a dataframe of time-to-event data with two hidden subgroups, </li>
#     <li> The <code>generate_mixture_oracle</code> function, which outputs a dataframe of time-to-event data with two hidden subgroups, together with the groundtruth counterfactual (both treatment and placebo outcomes are generated), </li>
#     <li> The helper <code>KM_plot</code> class, which allows to quickly draw Kaplan-Meier curves from the output of the two previous functions. </li>
# </ul>
#
# Let's go!

# In[1]:


import numpy as np

from hte.data.generation import generate_mixture, generate_mixture_oracle
from hte.viz.plots import KaplanMeierPlots


# <h2> Description of the generating process </h2>
#
#
# Our objective is to generate time-to-event data with two planted subgroups having different response to treatment. </br>
# We use the following framework:
# <ul>
#     <li> $X$ denotes the baseline covariates. We will draw X from a multidimensional normal distribution. The corresponding mean and variance-covariance matrix will be tunable parameters. </li>
#     <li> $W$ denotes the treatment variable. As we place ourselves in a Randomized Control Trial setting, we draw W from a Bernoulli(1/2) distribution. </li>
#     <li> $T$ denotes the time-to-event variable. </li>
#     <li> $S: \mathcal{X} \rightarrow \{0, 1\}$ is a function corresponding to the subgroup assigment. From a practical perspective, it will be a python function. </li>
# </ul>
# We use the following model on the tail probability function of $T$:
#
# $$ \mathbb{P}(T \in [t, t + \mathrm{d}t) \, | \, X, W, T \geq t) = h_0(t) \exp\left( \beta_0 W + (\beta_1 - \beta_0)W S(X) + \gamma^T X \right). $$
#
# The interpretation of $\beta_0, \beta_1, \gamma$ is quite straightforward:
# <ul>
#     <li> $\beta_0$ is the hazard ratio in subgroup $S(X)=0$, </li>
#     <li> $\beta_1$ is the hazard ratio in subgroup $S(X)=1$, </li>
#     <li> $\gamma$ corresponds to the prognostic effect of X. </li>
# </ul>

# <h2> How to generate data with the generation module </h2>

# <h3> Generate empirical observation with generate_mixture </h3>
#
# Rouhly speaking, the function generate_mixture takes as inputs:
# <ul>
#     <li> The paramaters of the generating process $\beta_0, \beta_1, \gamma, S$. The function S corresponds to the argument $group$, </li>
#     <li> The number of observation, $size$,
#     <li> The parameters of the law of $X$: mean, covariance, model, base hazard, a, b, and the existence of censored observation. Beware that the dimrension of $X$ is implicity encoded in the shapes of mean and covariance. </li>
# </ul>
#
# It outputs a dataframe whose columns are
# <ul>
#     <li> 'Times': the times of the process, </li>
#     <li> 'Event': takes value 1 if observed, 0 if censored, </li>
#     <li> 'W': takes value 1 uf Treated, 0 if Placebo, </li>
#     <li> 'G': the group of the observation, </li>
#     <li> 'X1, ...': the covariables. </li>
# </ul>

# In[2]:


df = generate_mixture(
    beta0 = 1.,
    beta1 = -1.,
    size = 100,
    group = lambda x: int(1),
    mean = np.array([0, 0]),
    cov = np.array([[1, 0], [0, 1]]),
    gamma = np.array([0.1, 1.]),
    model = 'Cox',
    base_hazard = 'Weibull',
    a = 1.0,
    b = 2.0,
    censored = True,
    scale = 0.5
)
print(df['Event'].mean())
df.head()


# In[3]:


km_plots = KaplanMeierPlots()
km_plots.plot(df, 'Example of plot')


# <h3> Generate data for a fixed $\beta$ with generate_oracle </h3>
#
# In order to control the degree of heterogeneity between subgroups 0 and 1, we need to choose the right values of $\beta_0$ and $\beta_1$ in the above function. Let us give more details about this. </br>
#
# The way we measure treatment effect is through the Absolute Risk Reduction at a timepoint $t$, which is defined by:
#
# $$ ARR(t) := \mathbb{P}( T \geq t \, | \, W=1) - \mathbb{P}( T \geq t \, | \, W=0). $$
#
# In words, it is the difference of probability of event between the treatment and placebo subgroups. </br>
#
# In our setting, the amount of heterogeneity is measured by the absolute difference of ARR between the two subgroups:
#
# $$ \text{Heterogeneity} := | ARR_1(t) - ARR_0(t) |, $$
#
# where, for $\varepsilon \in \{0, 1\}$:
#
# $$ ARR_\varepsilon(t) = \mathbb{P}( T \geq t \, | \, W=1, S=\varepsilon ) - \mathbb{P}( T \geq t \, | \, W=0, S=\varepsilon).  $$
#
# Fixing the parameters of the law of $X$ and the prognostic parameter $\gamma$:
# <ul>
#     <li> $ARR_0$ is a function of $\beta_0$:
#     $$ ARR_0(t, \beta_0) = \mathbb{E} \big[  \exp\left( - H_0(t) \exp\left( \beta_0 + \gamma^T X \right) \right)  -   \exp\left( - H_0(t) \exp\left( \gamma^T X \right) \right)  \, | \, S(X) = 0 \big] $$
#     </li>
#     <li> $ARR_1$ is a function of $\beta_1$:
#     $$ ARR_1(t, \beta_1) = \mathbb{E} \big[  \exp\left( - H_0(t) \exp\left( \beta_0 + \gamma^T X \right) \right) -   \exp\left( - H_0(t) \exp\left( \gamma^T X \right) \right) \, | \, S(X) = 1 \big] $$
#     </li>
# </ul>
#
# <b>About the monotonicity of the ARR.</b> From the above formulas, it is clear that the ARR is decreasing in its $\beta$ argument. Otherwise saying, a larger $\beta$ is beneficial for the treated.

# In[4]:


df_oracle = generate_mixture_oracle(
    beta = -1. ,
    size = int(1e5),
    group = lambda x: int((x[0] > -0.4) & (x[1] > -0.4)), # a function
    mean = np.array([0, 0]),
    cov = np.array([[1, 0], [0, 1]]),
    gamma = np.array([0.1, 1.]),
    model = 'Cox',
    base_hazard = 'Weibull',
    a = 1.0,
    b = 2.0,
    horizon = 1.,
    censored = True,
    scale = 0.5
)
df_oracle.head()


# Notice that $S1 = \mathbb{P}(T \leq t \, | \, W=1)$ and $S0 = \mathbb{P}(T \leq t \, | \, W=0)$.

# From this dataframe, we can now obtained Monte-Carlo estimates of $ARR_0$ and $ARR_1$ by simply slicing on the G column. Beware that the variance of the estimates are not the same for each subgroup, depending on their respective prevalence.

# In[5]:


ARR0 = df_oracle[df_oracle['G'] == 0]['IARR'].mean()
ARR1 = df_oracle[df_oracle['G'] == 1]['IARR'].mean()
print('ARR0 is {}'.format(ARR0))
print('ARR1 is {}'.format(ARR1))


# Notice that the ARRs are not the same in each subgroup as their definition depends on $X$ itself.
