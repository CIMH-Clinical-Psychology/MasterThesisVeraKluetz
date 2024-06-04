# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:19:04 2024

@author: simon.kern
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import settings
import utils

plt.rc('font', size= 16)

df_data = pd.concat([utils.load_exp_data(p) for p in settings.participants]
                    , ignore_index=True)


#%% First check overall correlation between valence and arousal

fig, axs = plt.subplots(2, 2, figsize=[10, 10])
# axs = axs.flatten()


df_corr = pd.DataFrame()

for i, name1 in enumerate(['valence', 'arousal']):
    for j, name2 in enumerate(['valence', 'arousal']):

        ax = axs[i, j]
        sns.stripplot(df_data, x=name1, y=f'emo_{name2}.rating', ax=ax, color=sns.color_palette()[i], orient='h', alpha=0.1)
        sns.regplot(df_data, x=name1, y=f'emo_{name2}.rating', ax=ax, color=sns.color_palette()[i], scatter=False)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_title(f'obj {name1} x subj {name2} rating correlation across all participants')
        r, p = pearsonr(df_data[name1], df_data[f'emo_{name2}.rating'])
        ax.text(6, 1.35, f'{r=:.2f}, {p=:.3f}', fontsize=16)
       
        
for i, name in enumerate(['valence', 'arousal']):

   for p, df_subj in df_data.groupby('participant_id'):
       r_subj, p_subj = pearsonr(df_subj[name], df_subj[f'emo_{name}.rating'])
       df_corr = pd.concat([df_corr, pd.DataFrame({'Participant': p,
                                                    'marker': name,
                                                    'r': r_subj},
                                                   index=[0])])

# plot correlation
fig, axs = plt.subplots(2, 1, figsize=[12, 6])

ax = axs[0]
df_corr.sort_values(['marker', 'r'], inplace=True)
sns.scatterplot(df_corr[df_corr.marker=='valence'], x='Participant', y='r', ax=ax)  
plt.sca(ax)
plt.xticks(rotation=90)
ax.set_title('Individual correlation values valence')

ax = axs[1]
sns.scatterplot(df_corr[df_corr.marker=='arousal'], x='Participant', y='r', ax=ax, color=sns.color_palette()[1])  
plt.sca(ax)
plt.xticks(rotation=90)
ax.set_title('Individual correlation values arousal')

fig.tight_layout()

#%% check individual levels

fig, axs = plt.subplots(1, 2, figsize=[8, 8])
sns.boxplot(data=df_data, x='valence_binary', y='emo_valence.rating', ax=axs[0])
axs[0].set_title('Overal valence ratings')
sns.boxplot(data=df_data, x='valence_binary', y='emo_arousal.rating', ax=axs[1])
axs[1].set_title('Overal arousal ratings')
plt.tight_layout()


#%% button presses

fig, ax = plt.subplots(1, 1, figsize=[8, 8])
sns.barplot(data=df_data, x='valence_binary', y='button_pressed', ax=ax)


button_presses = []
for subj, df_subj in df_data.groupby('participant_id'):
    x = df_subj.groupby('valence_binary').mean(True).reset_index()
    button_presses += [[x.button_pressed.values]]
button_presses = np.squeeze(button_presses)

fig, ax = plt.subplots(1, 1, figsize=[12, 2])
g = ax.imshow(np.squeeze(button_presses).T)
ax.set_xticks(np.arange(len(df_data.participant_id.unique())),df_data.participant_id.unique(), rotation=90)
ax.set_yticklabels(['', 'negative', 'positive'])
ax.set_ylabel('valence')
ax_c = fig.colorbar(g)
ax_c.set_label('ratio of\npressed buttons')

fig, ax = plt.subplots(1, 1, figsize=[8, 8])
diff = np.diff(button_presses, axis=-1).squeeze()
df_tmp = pd.DataFrame({'Participant': settings.participants,
                       'button bias': diff})
df_tmp.sort_values('button bias', inplace=True)
sns.scatterplot(df_tmp, x='Participant', y='button bias')
plt.xticks(rotation=90)
ax.set_ylabel('<-- more on negative GIFs    button press   more on positive GIFs-->', fontsize=12)
ax.set_title('Button press bias')

#%% button presses correlations

fig, ax = plt.subplots(1, 1, figsize=[8, 8])
sns.regplot(data=df_data, x='emo_arousal.rating', y='button_pressed')
sns.stripplot(data=df_data, x='emo_arousal.rating', y='button_pressed', alpha=0.01)
