#!/usr/bin/env python
# coding: utf-8

# # Analysis of csv results

# ## Imports and colors

# In[2]:


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


# In[3]:


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


# In[4]:


dict_scale = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6'
}


# ## Accuracy

# To produce a table with the accuracy rate associated to a subgroup, you need to write, in the cell below:
# - the name of the subgroup,
# - the size of the training set
# - the dimension p

# In[11]:


sbg_name = "dim1000_pred4_prog0_balanced"
p = 1000
nb = 10  # 1 if typeI error, 10 otherwise
train_size = 250
test_size = 250


# In[12]:


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
        #df_['censor'].replace('0.1', '0.0')
        print(file.split('scale=')[1].split('_')[0])
        list_df.append(df_)
df = pd.concat([d for d in list_df], ignore_index=True)
table = (
    df[df['arr'].isin([df['arr'].unique()[i] for i in[0, 3, 6, 9]])][['arr', 'method', 'censor', 'Accuracy']]
    .groupby(['arr', 'method', 'censor'])
    .agg(lambda x: mean_confidence_interval(x))
    .unstack(level='censor')
)

table.columns = table.columns.droplevel(0)
table.to_csv("./averaged_results/rq3/[{}]_train={}_test={}.csv".format(sbg_name, train_size, test_size))
table


# In[13]:


latex_table = table.to_latex()
print(latex_table)


# ### Dim20 and 100

# To produce a dataframe with the accuracy associated to a subgroup, you need to write, in the cell below:
# - the name of the subgroup,
# - the size of the training set
# - the dimension p

# In[4]:


sbg_name = "dim100_pred4_prog0_balanced"
train_size = str(250)
test_size = str(250)
pdim = 100


# In[5]:


# list files in results folder
list_df = []
for elem in os.listdir('./processed_results/'):
    if elem[:3] != 'COM':
        if elem[-3:] == 'csv':
            if elem.split("'")[1][:len(sbg_name)] == sbg_name:
                if elem.split("train=")[1].split('_')[0] == train_size:
                    if elem.split("test=")[1].split('_')[0] == test_size:
                        if elem.split("_train")[0].split("nb=")[2] == "10":
                            if elem.split('scale=')[1].split('_')[0] in ['0.1', '1.0', '2.0', '3.0']:
                                print(elem)
                                df_ = pd.read_csv('./processed_results/'+elem, index_col=0)
                                df_['censor'] = elem.split('scale=')[1].split('_')[0]
                                list_df.append(df_)

df = pd.concat([d for d in list_df], ignore_index=True)


# In[6]:


table = df[(df['arr'].isin([df['arr'].unique()[i] for i in[0, 3, 6, 9]])) & (df['method'] != 'Oracle')][['arr', 'method', 'censor', 'Accuracy']].groupby(['arr', 'method', 'censor']).agg(lambda x: mean_confidence_interval(x)).unstack(level='censor')
table.columns = table.columns.droplevel(0)
table


# In[27]:


latex_table = table.to_latex()
#print(latex_table)


# ### Dim 1000

# For p=1000, need to rely on chunks to read the csv.
# - First, identify the csv file with the two first cells.
# - Then, read it using chunks.
# - Finally, look at the results

# In[8]:


sbg_name = "dim1000_pred4_prog0_balanced"
train_size = str(250)
test_size = str(250)
pdim = 1000


# In[10]:


# list files in results folder
list_df = []
for elem in os.listdir('./processed_results/'):
    if elem[:3] != 'COM':
        if elem[-3:] == 'csv':
            if elem.split("'")[1][:len(sbg_name)] == sbg_name:
                if elem.split("train=")[1][:len(train_size)] == train_size:
                    if elem.split("test=")[1][:len(test_size)] == test_size:
                        if elem.split("_train")[0].split("nb=")[2] == "10":
                            if elem.split('scale=')[1].split('_')[0] in ['0.1', '4.0', '5.0', '6.0']:
                                print(elem)
                                reader = pd.read_csv('./processed_results/'+elem, iterator=True, chunksize=1000)
                                chunks = []
                                k = 1
                                for chunk in reader:
                                    print(k)
                                    k += 1
                                    chunks.append(chunk)
                                df_per_file = pd.concat(chunks, ignore_index=True).drop(columns='Unnamed: 0')
                                df_per_file['censor'] = elem.split('scale=')[1].split('_')[0]
                                list_df.append(df_per_file)


# In[11]:


df = pd.concat([d for d in list_df], ignore_index=True)


# In[12]:


table = df[(df['arr'].isin([df['arr'].unique()[i] for i in[0, 3, 6, 9]])) & (df['method'] != 'Oracle')][['arr', 'method', 'censor', 'Accuracy']].groupby(['arr', 'method', 'censor']).agg(lambda x: mean_confidence_interval(x)).unstack(level='censor')
table.columns = table.columns.droplevel(0)
table


# In[28]:


latex_table = table.to_latex()
#print(latex_table)


# ## Accurary curve 20, 100, 1000 (main paper)

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


# Join the DataFrames and plot

# In[8]:


df_20['dim'] = 20
df_100['dim'] = 100
df_1000['dim'] = 1000
df = pd.concat([df_20, df_100, df_1000], ignore_index=True)


# In[9]:


# accuracy
fig, axes = plt.subplots(1, 3,figsize=(15, 4))
for i, dim in enumerate([20, 100, 1000]):
    for k, p in plot_dict.items():
        c, s, l = p
        if (k != 'Oracle'):
            sns.lineplot(
                data=df[(df['dim'] == dim) & (df['method'] == k)],
                x='arr',
                y='Accuracy',
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

axes[0].set_ylabel('Accuracy', fontsize=15)
axes[1].set_ylabel('')
axes[2].set_ylabel('')

axes[0].set_ylim(0.4, 1.05)
axes[1].set_ylim(0.4, 1.05)
axes[2].set_ylim(0.4, 1.05)

axes[0].set_title('p = 20', fontsize=15)
axes[1].set_title('p = 100', fontsize=15)
axes[2].set_title('p = 1000', fontsize=15)

fig.legend(ncol=5, labels=labels, bbox_to_anchor = (0.5, 1.15), loc='upper center', fontsize=13)
fig.savefig("./figures/main_paper/accuracy.pdf", bbox_inches='tight')
plt.show()


# In[ ]:
