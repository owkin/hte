#!/usr/bin/env python
# coding: utf-8

# # Analysis of csv results

# ## Imports and colors

# In[1]:


import os
import re
import csv
import json
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import ast
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from sklearn.metrics import roc_auc_score, average_precision_score

sns.set_style('whitegrid')
sns.set_palette("bright")  # tab10
print(sns.color_palette().as_hex())

def mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    mean, std = np.mean(data), np.std(data, ddof=1)
    margin = std * norm.ppf(97.5 / 100) / np.sqrt(n)
    return f"{mean:.2f} +/- {margin:.2f}"


# In[2]:


plot_dict = {
    'Oracle': ['#00d7ff', 'd', 'solid'],
    'Univariate interaction': ['#023eff', 'o', 'solid'],
    'Univariate t_test': ['#ff7c00', '^', 'solid'],
    'Multivariate cox': ['#1ac938', '<', 'dashed'],
    'Multivariate tree': ['#e8000b', '>', 'dashed'],
    'MOB': ['#8b2be2', 's', 'dotted'],
    'ITree': ['#9f4800', 'D', 'dotted'],
    'SIDES': ['#f14cc1', 'p', 'dashdot'],
    'SeqBT': ['#a3a3a3', '8', 'dashdot'],
    'ARDP': ['#ffc400', 'X', 'dashdot']
}


# In[3]:


dict_scale = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6'
}


# ## Prob top var predictive, Averaged precision score, AUC score

# To produce a table with
# - the probability of the top var being predictive
# - the averaged precision score
# - the AUC score
#
# associated to a subgroup, you need to write, in the cell below:
#
# - the name of the subgroup,
# - the size of the training set
# - the dimension p

# In[19]:


sbg_name = "dim20_pred4_prog0_balanced"
p = 20
nb = 10  # 1 if typeI error, 10 otherwise
train_size = 250
test_size = 250


# In[20]:


pattern = re.compile((
    r"^(?!COM)(.*_group='{}_.*)(.*_nb={}_.*)(.*train={}_.*)(.*test={}_.*)(.*scale=(0.1|1.0|2.0|3.0)_.*)\.csv$"
).format(sbg_name, nb, train_size, test_size))
files = os.listdir('./processed_results/')
selected_files = [file for file in files if pattern.match(file)]
#del selected_files[1]
#del selected_files[2]

list_df = []
for file in selected_files:
    if file.split('scale=')[1].split('_')[0] in ['0.1', '1.0', '2.0', '3.0']:
        print(file)
        df_ = pd.read_csv('./processed_results/' + file, index_col=0)
        df_['censor'] = file.split('scale=')[1].split('_')[0]
        df_['censor'].replace('0.1', '0.0')
        print(file.split('scale=')[1].split('_')[0])
        list_df.append(df_)
df = pd.concat([d for d in list_df], ignore_index=True)

table = (
    df[df['arr'].isin([df['arr'].unique()[i] for i in[0, 3, 6, 9]])][['arr', 'method', 'censor', 'top_var_selected', 'averaged_precision_score', 'auc_score']]
    .groupby(['arr', 'method', 'censor'])
    .agg(lambda x: mean_confidence_interval(x))
    .unstack(level='censor')
)
table.columns = table.columns.droplevel(0)
table.to_csv("./averaged_results/rq2/[{}]_train={}_test={}.csv".format(sbg_name, train_size, test_size))
table


# In[21]:


# only average precision score for paper's appendix tables
list_df = []
for file in selected_files:
    if file.split('scale=')[1].split('_')[0] in ['0.1', '1.0', '2.0', '3.0']:
        print(file)
        df_ = pd.read_csv('./processed_results/' + file, index_col=0)
        df_['censor'] = file.split('scale=')[1].split('_')[0]
        df_['censor'].replace('0.1', '0.0')
        print(file.split('scale=')[1].split('_')[0])
        list_df.append(df_)
df = pd.concat([d for d in list_df], ignore_index=True)

table = (
    df[df['arr'].isin([df['arr'].unique()[i] for i in[0, 3, 6, 9]])][['arr', 'method', 'censor', 'averaged_precision_score']]
    .groupby(['arr', 'method', 'censor'])
    .agg(lambda x: mean_confidence_interval(x))
    .unstack(level='censor')
)
table.columns = table.columns.droplevel(0)
table


# In[22]:


latex_table = table.to_latex()
print(latex_table)


# ## Top var selected plot and averaged precision score plot (main paper)

# In[4]:


# name of the files
csv_dim20 = "./processed_results/Cox_Weibull_1.0_2.0_dim=20_range=[-10.0,10.0]_nb=500_group='dim20_pred4_prog0_balanced_rangeARR=[0.0, 0.432410680647274]_nb=10_train=250_test=250_repet=100_censored=False_scale=0.1_December_12_20_2023_01:21:02.csv"
csv_dim100 = "./processed_results/Cox_Weibull_1.0_2.0_dim=100_range=[-10.0,10.0]_nb=500_group='dim100_pred4_prog0_balanced_rangeARR=[0.0, 0.4330995288724966]_nb=10_train=250_test=250_repet=100_censored=False_scale=0.1_December_12_27_2023_10:46:14.csv"
csv_dim1000 = "./processed_results/Cox_Weibull_1.0_2.0_dim=1000_range=[-10.0,10.0]_nb=500_group='dim1000_pred4_prog0_balanced_rangeARR=[0.0, 0.4371639648096365]_nb=10_train=250_test=250_repet=100_censored=False_scale=0.1_December_12_24_2023_12:12:01.csv"


# In[5]:


# load dim20 and dim100
df_20 = pd.read_csv(csv_dim20, index_col=0)

df_100 = pd.read_csv(csv_dim100, index_col=0)


# In[6]:


# load dim1000 (next two cells)
reader = pd.read_csv(csv_dim1000, iterator=True, chunksize=1000)
chunks = []
k = 1
for chunk in reader:
    print(k)
    k += 1
    chunks.append(chunk)


# In[7]:


df_1000 = pd.concat(chunks, ignore_index=True).drop(columns='Unnamed: 0')


# Join DataFrames and plot.

# In[8]:


df_20['dim'] = 20
df_100['dim'] = 100
df_1000['dim'] = 1000
df = pd.concat([df_20, df_100, df_1000], ignore_index=True)


# In[9]:


fig, axes = plt.subplots(1, 3,figsize=(16, 4))
for i, dim in enumerate([20, 100, 1000]):
    for k, p in plot_dict.items():
        c, s, l = p
        if (k != 'Oracle') & (k != 'Multivariate tree'):
            sns.lineplot(
                data=df[(df['dim'] == dim) & (df['method'] == k)],
                x='arr',
                y='top_var_selected',
                color=c,
                marker=s,
                linestyle=l,
                label=k,
                errorbar=None,
                ax=axes[i]
            )
        if dim == 20:
            handles, labels = axes[0].get_legend_handles_labels()

axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].get_legend().remove()

axes[0].set_xlabel('ARR', fontsize=15)
axes[1].set_xlabel('ARR', fontsize=15)
axes[2].set_xlabel('ARR', fontsize=15)

axes[0].set_ylabel('Probability of top variable \n being predictive', fontsize=14)
axes[1].set_ylabel('')
axes[2].set_ylabel('')

axes[0].set_ylim(-0.05, 1.05)
axes[1].set_ylim(-0.05, 1.05)
axes[2].set_ylim(-0.05, 1.05)

axes[0].set_title('p = 20', fontsize=15)
axes[1].set_title('p = 100', fontsize=15)
axes[2].set_title('p = 1000', fontsize=15)

#fig.legend(labels=labels)
#plt.legend(loc='best')
fig.legend(ncol=4, labels=labels, bbox_to_anchor = (0.5, 1.15), loc='upper center', fontsize=13)
#fig.tight_layout()
fig.savefig("./figures/main_paper/prob_top_var_predictive.pdf", bbox_inches='tight')
plt.show()


# In[10]:


fig, axes = plt.subplots(1, 3,figsize=(16, 4))
for i, dim in enumerate([20, 100, 1000]):
    for k, p in plot_dict.items():
        c, s, l = p
        if (k != 'Oracle') & (k != 'Multivariate tree'):
            sns.lineplot(
                data=df[(df['dim'] == dim) & (df['method'] == k)],
                x='arr',
                y='averaged_precision_score',
                color=c,
                marker=s,
                linestyle=l,
                label=k,
                errorbar=None,
                ax=axes[i]
            )
        if dim == 20:
            handles, labels = axes[0].get_legend_handles_labels()

axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].get_legend().remove()

axes[0].set_xlabel('ARR', fontsize=15)
axes[1].set_xlabel('ARR', fontsize=15)
axes[2].set_xlabel('ARR', fontsize=15)

axes[0].set_ylabel('Averaged precision score', fontsize=14)
axes[1].set_ylabel('')
axes[2].set_ylabel('')

axes[0].set_ylim(-0.05, 1.05)
axes[1].set_ylim(-0.05, 1.05)
axes[2].set_ylim(-0.05, 1.05)

axes[0].set_title('p = 20', fontsize=15)
axes[1].set_title('p = 100', fontsize=15)
axes[2].set_title('p = 1000', fontsize=15)

#fig.legend(labels=labels)
#plt.legend(loc='best')
fig.legend(ncol=4, labels=labels, bbox_to_anchor = (0.5, 1.15), loc='upper center', fontsize=13)
#fig.tight_layout()
fig.savefig("./figures/main_paper/averaged_precision_score.pdf", bbox_inches='tight')
plt.show()


# ## Histograms (main paper)

# In[11]:


csv_dim20 = "./processed_results/Cox_Weibull_1.0_2.0_dim=20_range=[-10.0,10.0]_nb=500_group='dim20_pred4_prog0_balanced_rangeARR=[0.0, 0.432410680647274]_nb=10_train=250_test=250_repet=100_censored=False_scale=0.1_December_12_20_2023_01:21:02.csv"


# In[12]:


# load dim20
df = pd.read_csv(csv_dim20, index_col=0)


# In[13]:


df_ = df[(df['arr'].isin([0., 0.432410680647274])) & (df['method'] != 'Multivariate tree') & (df['method'] != 'Oracle')].copy()
df_['arr'] = df_['arr'].apply(lambda x: 0.0 if x == 0. else 0.43)
df_['top_var'] = df_['ranked_var'].apply(lambda x: int(ast.literal_eval(x)[0][1:]) if x != "['']" else np.random.randint(1, 21))


# In[14]:


plt.figure(figsize=(15, 6))
g = sns.FacetGrid(data=df_[df_['method'].isin(['Univariate interaction', 'Univariate t_test', 'Multivariate cox', 'MOB'])], row='arr', col='method', margin_titles=True)
g.map_dataframe(sns.histplot, x='top_var', bins=21, binrange=(1,21), discrete=True, color='royalblue', stat='probability')
g.set_titles(size=12)
g.axes[1,0].set_xlabel('Top variable', fontsize = 12)
g.axes[1,1].set_xlabel('Top variable', fontsize = 12)
g.axes[1,2].set_xlabel('Top variable', fontsize = 12)
g.axes[1,3].set_xlabel('Top variable', fontsize = 12)

g.axes[0,0].set_title('Univariate interaction', fontsize = 14)
g.axes[0,1].set_title('Univariate t-test', fontsize = 14)
g.axes[0,2].set_title('Multivariate cox', fontsize = 14)
g.axes[0,3].set_title('MOB', fontsize = 14)
plt.savefig("./figures/main_paper/histograms_1.pdf")
plt.show()


# In[15]:


plt.figure(figsize=(15, 6))
g = sns.FacetGrid(data=df_[df_['method'].isin(['ITree', 'SIDES', 'SeqBT', 'ARDP'])], row='arr', col='method', margin_titles=True)
g.map_dataframe(sns.histplot, x='top_var', bins=21, binrange=(1,21), discrete=True, color='royalblue', stat='probability')
g.set_titles(size=12)
g.axes[1,0].set_xlabel('Top variable', fontsize = 12)
g.axes[1,1].set_xlabel('Top variable', fontsize = 12)
g.axes[1,2].set_xlabel('Top variable', fontsize = 12)
g.axes[1,3].set_xlabel('Top variable', fontsize = 12)

g.axes[0,0].set_title('ITree', fontsize = 14)
g.axes[0,1].set_title('SIDES', fontsize = 14)
g.axes[0,2].set_title('SeqBT', fontsize = 14)
g.axes[0,3].set_title('ARDP', fontsize = 14)
plt.savefig("./figures/main_paper/histograms_2.pdf")
plt.show()


# In[ ]:
