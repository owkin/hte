#!/usr/bin/env python
# coding: utf-8

# # Analysis semi-synth expe

# ## Import and colors

# In[2]:


import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from sklearn.metrics import roc_auc_score, average_precision_score

sns.set_style('whitegrid')
sns.set_palette("bright")  # tab10
print(sns.color_palette().as_hex())


# In[3]:


plot_dict = {
    'Oracle': ['#00d7ff', 'd', 'solid'],
    'Univariate interaction': ['#023eff', 'o', 'solid'],
    'Univariate t_test': ['#ff7c00', '^', 'solid'],
    'Multivariate Cox': ['#1ac938', '<', 'dashed'],
    'Multivariate Tree': ['#e8000b', '>', 'dashed'],
    'MOB': ['#8b2be2', 's', 'dotted'],
    'ITree': ['#9f4800', 'D', 'dotted'],
    'SIDES': ['#f14cc1', 'p', 'dashdot'],
    'SeqBT': ['#a3a3a3', '8', 'dashdot'],
    'ARDP': ['#ffc400', 'X', 'dashdot']
}


# In[4]:


dict_scale = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6'
}


# ## Loading file

# In[5]:


csv_file = "./processed_results/Cox_Weibull_1.0_2.0_dim=1000_range=[-10.0,10.0]_nb=500_group='semi_synth_sbg_1000_rangeARR=[0.0, 0.3785731461020696]_nb=10_train=250_test=250_repet=100_censored=True_scale=1.0_December_12_22_2023_08:38:15.csv"


# In[6]:


reader = pd.read_csv(
    csv_file,
    iterator=True,
    chunksize=1000
)


# In[7]:


chunks = []
k = 1
for chunk in reader:
    print(k)
    k += 1
    chunks.append(chunk)


# In[8]:


df = pd.concat(chunks, ignore_index=True).drop(columns='Unnamed: 0')


# ## RQ1

# In[9]:


table = df[['arr', 'method', 'thresh_pval']].groupby(['arr', 'method']).mean().unstack(level='method')
table.columns = table.columns.droplevel(0)
table = table.reset_index()
table = table[['arr', 'Oracle', 'Univariate interaction', 'Univariate t_test', 'Multivariate cox', 'Multivariate tree', 'MOB', 'ITree', 'ARDP']].round(decimals=2)
table.to_csv("./averaged_results/rq1/power_[semi_synth]_train={}_test={}.csv".format(250, 250))
table


# In[31]:


latex_table = table.to_latex()
#print(latex_table)


# ## RQ2

# In[10]:


# look at the values
df['arr'].unique()


# In[33]:


# choose your value here
arr_ = df['arr'].unique()[4]


# In[34]:


table = df[['arr', 'method', 'top_var_selected', 'averaged_precision_score', 'auc_score']].groupby(['arr', 'method']).mean()
table.to_csv("./averaged_results/rq2/[semi_synth]_train={}_test={}.csv".format(250, 250))
table


# In[11]:


table = df[['arr', 'method', 'averaged_precision_score']].groupby(['arr', 'method']).mean().unstack(level='method')
table.columns = table.columns.droplevel(0)
table = table.reset_index()
table = table[['arr', 'Univariate interaction', 'Univariate t_test', 'Multivariate cox', 'MOB', 'ITree', 'ARDP']].round(decimals=2)
table


# In[ ]:


latex_table = table.to_latex()
#print(latex_table)


# ## RQ3

# In[12]:


table = df[['arr', 'method', 'Accuracy']].groupby(['arr', 'method']).mean().unstack(level='method')
table.columns = table.columns.droplevel(0)
table.to_csv("./averaged_results/rq3/[semi_synth]_train={}_test={}.csv".format(250, 250))
table = table.reset_index()
table = table[['arr', 'Univariate interaction', 'Univariate t_test', 'Multivariate cox', 'Multivariate tree', 'MOB', 'ITree', 'ARDP']].round(decimals=2)
table


# In[ ]:


latex_table = table.to_latex()
#print(latex_table)


# In[ ]:
