#!/usr/bin/env python
# coding: utf-8

# # Analysis of csv results

# ## Imports and colors

# In[16]:


import os
import re
import csv
import json
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


# In[17]:


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


# In[18]:


dict_scale = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6'
}


# ## Power tables

# To produce a table with the type I error rate (arr=0) and power associated to a subgroup, you need to write, in the cell below:
# - the name of the subgroup,
# - the size of the training set
# - the dimension p

# In[31]:


sbg_name = "dim20_pred4_prog0_balanced"
p = 20
nb = 10
train_size = 250
test_size = 250


# In[32]:


pattern = re.compile((
    r"^(?!COM)(.*_group='{}_.*)(.*_nb={}_.*)(.*train={}_.*)(.*test={}_.*)(.*scale=(0.1|1.0|2.0|3.0)_.*)\.csv$"
).format(sbg_name, nb, train_size, test_size))
files = os.listdir('./processed_results/')
selected_files = [file for file in files if pattern.match(file)]
selected_files


# In[33]:


pattern = re.compile((
    r"^(?!COM)(.*_group='{}_.*)(.*_nb={}_.*)(.*train={}_.*)(.*test={}_.*)(.*scale=(0.1|1.0|2.0|3.0)_.*)\.csv$"
).format(sbg_name, nb, train_size, test_size))
files = os.listdir('./processed_results/')
selected_files = [file for file in files if pattern.match(file)]

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
    df[df['arr'].isin([df['arr'].unique()[i] for i in[0, 3, 6, 9]])][['arr', 'method', 'censor', 'thresh_pval']]
    .groupby(['arr', 'method', 'censor'])
    .agg(lambda x: mean_confidence_interval(x))
    .unstack(level='censor')
)

table.columns = table.columns.droplevel(0)
table.to_csv("./averaged_results/rq1/type1_[{}]_train={}_test={}.csv".format(sbg_name, train_size, test_size))
table


# In[35]:


latex_table = table.to_latex()
print(latex_table)


# ## Type I error (main paper)

# To produce a dataframe with the typeI error rate associated to a subgroup, you need to write, in the cell below:
# - the name of the subgroup,
# - the size of the training set
# - the dimension p

# In[13]:


sbg_name = "dim1000_pred4_prog0_balanced"
p = 1000
nb = 1
train_size = 250
test_size = 250


# In[14]:


pattern = re.compile((
    r"^(?!COM)(.*_group='{}_.*)(.*_nb={}_.*)(.*train={}_.*)(.*test={}_.*)(.*scale=(0.1|1.0|2.0|3.0)_.*)\.csv$"
).format(sbg_name, nb, train_size, test_size))
files = os.listdir('./processed_results/')
selected_files = [file for file in files if pattern.match(file)]
selected_files


# In[15]:


pattern = re.compile((
    r"^(?!COM)(.*_group='{}_.*)(.*_nb={}_.*)(.*train={}_.*)(.*test={}_.*)(.*scale=(0.1|1.0|2.0|3.0)_.*)\.csv$"
).format(sbg_name, nb, train_size, test_size))
files = os.listdir('./processed_results/')
selected_files = [file for file in files if pattern.match(file)]

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
    df[['arr', 'method', 'censor', 'thresh_pval']]
    .groupby(['arr', 'method', 'censor'])
    .mean()
    .unstack(level='censor')
)

table.columns = table.columns.droplevel(0)
table.to_csv("./averaged_results/rq1/type1_[{}]_train={}_test={}.csv".format(sbg_name, train_size, test_size))
table


# ## Power curve 20, 100, 1000 (main paper)

# In[8]:


# name of the files
csv_dim20 = "./processed_results/Cox_Weibull_1.0_2.0_dim=20_range=[-10.0,10.0]_nb=500_group='dim20_pred4_prog0_balanced_rangeARR=[0.0, 0.432410680647274]_nb=10_train=250_test=250_repet=100_censored=False_scale=0.1_December_12_20_2023_01:21:02.csv"
csv_dim100 = "./processed_results/Cox_Weibull_1.0_2.0_dim=100_range=[-10.0,10.0]_nb=500_group='dim100_pred4_prog0_balanced_rangeARR=[0.0, 0.4330995288724966]_nb=10_train=250_test=250_repet=100_censored=False_scale=0.1_December_12_27_2023_10:46:14.csv"
csv_dim1000 = "./processed_results/Cox_Weibull_1.0_2.0_dim=1000_range=[-10.0,10.0]_nb=500_group='dim1000_pred4_prog0_balanced_rangeARR=[0.0, 0.4371639648096365]_nb=10_train=250_test=250_repet=100_censored=False_scale=0.1_December_12_24_2023_12:12:01.csv"


# In[9]:


# load dim20 and dim100
df_20 = pd.read_csv(csv_dim20, index_col=0)
df_100 = pd.read_csv(csv_dim100, index_col=0)


# In[10]:


# load dim1000 (next two cells)
reader = pd.read_csv(csv_dim1000, iterator=True, chunksize=1000)
chunks = []
k = 1
for chunk in reader:
    print(k)
    k += 1
    chunks.append(chunk)


# In[11]:


df_1000 = pd.concat(chunks, ignore_index=True).drop(columns='Unnamed: 0')


# Join the DataFrames and plot

# In[12]:


df_20['dim'] = 20
df_100['dim'] = 100
df_1000['dim'] = 1000
df = pd.concat([df_20, df_100, df_1000], ignore_index=True)


# In[13]:


# probability of top predicted variable being predictive
fig, axes = plt.subplots(1, 3,figsize=(15, 4))
for i, dim in enumerate([20, 100, 1000]):
    for k, p in plot_dict.items():
        c, s, l = p
        sns.lineplot(
            data=df[(df['dim'] == dim) & (df['method'] == k)],
            x='arr',
            y='thresh_pval',
            color=c,
            marker=s,
            linestyle=l,
            label=k,
            errorbar=None,
            #errorbar=('ci', 95),  # errorbar=('ci', 95)
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

axes[0].set_ylabel('Power', fontsize=15)
axes[1].set_ylabel('')
axes[2].set_ylabel('')

axes[0].set_ylim(0, 1.05)
axes[1].set_ylim(0, 1.05)
axes[2].set_ylim(0, 1.05)

axes[0].set_title('p = 20', fontsize=15)
axes[1].set_title('p = 100', fontsize=15)
axes[2].set_title('p = 1000', fontsize=15)

#fig.legend(labels=labels)
#plt.legend(loc='best')
fig.legend(handles=handles, ncol=5, labels=labels, bbox_to_anchor = (0.5, 1.15), loc='upper center', fontsize=13)
#fig.tight_layout()
fig.savefig("./figures/main_paper/power.pdf", bbox_inches='tight')
plt.show()


# ## Univariate power (main paper)

# In[14]:


train_size = str(250)
pdim = 20


# In[15]:


# list files in results folder
list_df = []
for elem in os.listdir('./processed_results/'):
    if elem[:3] != 'COM':
        if elem[-3:] == 'csv':
            if elem.split("censored=")[1][:5] == 'False':
                if elem.split("'")[1][:5] == "dim20":
                    if elem.split("train=")[1][:len(train_size)] == train_size:
                        if elem.split("test=")[1][:len(train_size)] == train_size:
                            if elem.split("scale=")[1].split("_")[0] in ['0.1']:
                                if elem.split("_train")[0].split("nb=")[2] == "10":
                                    print(elem)
                                    df_ = pd.read_csv('./processed_results/'+elem, index_col=0)
                                    df_['prog'] = int(elem.split("prog")[1].split("_")[0])
                                    list_df.append(df_)
df = pd.concat([d for d in list_df], ignore_index=True)
for i in range(1, 21):
    df['pval_X{}'.format(i)] = df['pval_X{}'.format(i)].apply(lambda x: np.clip(x/(20*3), 0, 1))
    df['thresh_pval_X{}'.format(i)] = df['pval_X{}'.format(i)].apply(lambda x: int(x < 0.05))


# In[16]:


prog = {
    0: [],
    1: [9],
    2: [9, 10],
    3: [6, 7, 8],
    4: [1, 2, 7, 8]
}

pred = {
    0: [17, 18, 19, 20],
    1: [13, 14, 15],
    2: [11, 12],
    3: [20],
    4: []

}


# In[17]:


# Power curve
sns.set_style("whitegrid", {'axes.grid' : False})
fig, axes = plt.subplots(1, 3,figsize=(15, 4))

for k in [0, 1, 2]:

    df_ = df[(df['prog'] == k) & (df['method'] == 'Univariate interaction')]

    for i in range(1, 21):
        if i in prog[k]:
            sns.lineplot(
                data=df_,
                x='arr',
                y='thresh_pval_X{}'.format(i),
                linestyle='dashdot',
                errorbar=None,
                ax=axes[k],
                label='X{}'.format(i)
            )
        elif i in pred[k]:
            sns.lineplot(
                data=df_,
                x='arr',
                y='thresh_pval_X{}'.format(i),
                linestyle='dotted',
                errorbar=None,
                ax=axes[k],
                label='X{}'.format(i)
            )
        else:
            sns.lineplot(
                data=df_,
                x='arr',
                y='thresh_pval_X{}'.format(i),
                errorbar=None,
                ax=axes[k]
            )
    axes[k].set_ylabel('')
    axes[k].set_xlabel('ARR', fontsize=15)
    axes[k].set_ylim(0, 1)
    axes[k].set_title('prog = {}'.format(k), fontsize=15)
    axes[k].set_ylim(0, 1.05)
    axes[k].legend()

axes[0].set_ylabel('Power', fontsize=15)

plt.tight_layout()
plt.savefig("./figures/main_paper/zoom_univariate1.pdf")
plt.show()


# In[18]:


# Power curve
sns.set_style("whitegrid", {'axes.grid' : False})
fig, axes = plt.subplots(1, 2,figsize=(9, 3.5))

for k in [3, 4]:

    df_ = df[(df['prog'] == k) & (df['method'] == 'Univariate interaction')]

    for i in range(1, 21):
        if i in prog[k]:
            sns.lineplot(
                data=df_,
                x='arr',
                y='thresh_pval_X{}'.format(i),
                linestyle='dashdot',
                errorbar=None,
                ax=axes[k-3],
                label='X{}'.format(i)
            )
        elif i in pred[k]:
            sns.lineplot(
                data=df_,
                x='arr',
                y='thresh_pval_X{}'.format(i),
                linestyle='dotted',
                errorbar=None,
                ax=axes[k-3],
                label='X{}'.format(i)
            )
        else:
            sns.lineplot(
                data=df_,
                x='arr',
                y='thresh_pval_X{}'.format(i),
                errorbar=None,
                ax=axes[k-3]
            )
    axes[k-3].set_ylabel('')
    axes[k-3].set_xlabel('ARR', fontsize=15)
    axes[k-3].set_ylim(0, 1)
    axes[k-3].set_title('prog = {}'.format(k), fontsize=15)
    axes[k-3].set_ylim(0, 1.05)
    axes[k-3].legend()

axes[0].set_ylabel('Power', fontsize=15)
plt.tight_layout()
plt.savefig("./figures/main_paper/zoom_univariate2.pdf")
plt.show()


# In[ ]:
