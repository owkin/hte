#!/usr/bin/env python
# coding: utf-8

# # Analysis of complexity results csv

# ## Imports and colors

# In[23]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_palette("bright")  # tab10
print(sns.color_palette().as_hex())


# In[24]:


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


# # Complexity

# In[43]:


df_complexity = pd.read_csv('./COMPLEXITY_DIM=[3, 20, 100, 500, 1000]_NB=[500]_REPET=1_October_10_14_2023_08:04:07.csv').drop('Unnamed: 0', axis=1)
df_complexity.head()


# In[44]:


methods_name_dict = {"univariate_test" : 'Univariate t_test',
                     "univariate_interaction" : 'Univariate interaction',
                     "multivariate_cox": 'Multivariate Cox',
                     "multivariate_tree" : 'Multivariate Tree',
                     "sides": 'SIDES',
                     "model_based": 'MOB',
                    "interaction_trees": 'ITree',
                    "sequential_batting": 'SeqBT',
                    "directed_peeling": 'ARDP'}
df_complexity = df_complexity.replace({"method": methods_name_dict})


# In[50]:


plot_df = df_complexity.drop("nb_samples", axis=1)
#plot_df = plot_df.pivot_table("time", "dim", "method")


# In[75]:


# complexity (time) for method fitting step
fig, axes = plt.subplots(1, 1,figsize=(6, 4))
#for i, dim in enumerate([20, 100, 1000]):
for k, p in plot_dict.items():
    c, s, l = p
    if (k != 'Oracle'):
        sns.lineplot(
            data=plot_df[(plot_df['method'] == k)],
            x='dim',
            y='time',
            color=c,
            marker=s,
            linestyle=l,
            label=k,
            errorbar=None
        )
plt.yscale('log')
plt.xlabel('Dimension', fontsize=15)
plt.ylabel('Run time', fontsize=15)
#plt.legend(loc)
plt.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=13)
#sns.move_legend(plot, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=13)
plt.savefig("./figures/main_paper/complexity.pdf", bbox_inches='tight')


# In[ ]:
