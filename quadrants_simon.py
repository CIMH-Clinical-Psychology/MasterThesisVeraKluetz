#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:51:33 2024

@author: simon.kern
"""
import os
import ospath
import settings
import mne
from tqdm import tqdm
import utils
from joblib import Memory
import numpy as np
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

mem = Memory(settings.cachedir)

missing = [25, 28, 31]
participants = [str(i).zfill(2) for i in range(1, 36) if not i in missing]

@mem.cache
def load_resample(participant, tmin=-0.5, tmax=1, picks='meg'):
    file = os.path.join(settings.datadir, f'ERP-{participant}/ERP{participant}_initial_tsss_mc.fif')
    raw = mne.io.read_raw(file,  verbose='ERROR')
    raw.set_channel_types({'BIO002':'eog', 'BIO003':'eog'})
    events = mne.find_events(raw, min_duration=3/1000)
    raw.pick(picks)
    epochs = mne.Epochs(raw, events, event_id=[10],
                        tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
    epochs.resample(100, n_jobs=-1)
    data_x = epochs.get_data(picks=picks)
    gif_pos_chars = utils.load_exp_data(participant).gif_position
    # convert letter to number
    char_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    gif_pos = [char_to_num[i] for i in gif_pos_chars]
    data_y = np.array(gif_pos)
    return epochs.times, data_x, data_y


fig, axs, ax_bottom = utils.make_fig(n_axs=len(participants), n_bottom=[0, 0, 1])

n_splits = 5
tmin = -2.5
tmax = 2.5
picks = 'eog'

df_all = pd.DataFrame()

# base_clf = RandomForestClassifier(500)
base_clf = LogisticRegression(solver="liblinear", max_iter=100)  # liblinear is faster than lbfgs

for i, participant in enumerate(tqdm(participants)):
    times, data_x, data_y = load_resample(participant, tmin=tmin, tmax=tmax, picks=picks)

    # np.random.shuffle(data_y)  # shuffle data for checking randomness

    pipeline = make_pipeline(
        StandardScaler(),
        base_clf,
    )

    clf = SlidingEstimator(pipeline, scoring='accuracy', n_jobs=-1)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_multiscore(clf, data_x, data_y, cv=cv, n_jobs=-1)
    df_subj = pd.DataFrame({'participant': participant,
                            'timepoint': np.hstack([times]*n_splits),
                            'accuracy': scores.ravel()})
    df_all = pd.concat([df_all, df_subj])

    ax = axs[i]  # select axis of this participant
    sns.lineplot(data=df_subj, x='timepoint', y='accuracy', ax=ax)
    ax.hlines(0.25, min(times), max(times), linestyle='--', color='gray')  # draw random chance line
    ax.set_title(f'{participant=}')
    # then plot a summary of all participant into the big plot
    ax_bottom.clear()  # clear axis from previous line
    sns.lineplot(data=df_all, x='timepoint', y='accuracy', ax=ax_bottom)
    ax_bottom.hlines(0.25, min(times), max(times), linestyle='--', color='gray')  # draw random chance line
    ax_bottom.set_title(f'Mean of {len(df_all.participant.unique())} participants')
    utils.normalize_lims(axs)
    plt.pause(0.1)  # necessary for plotting to update
    fig.suptitle('Decoding on {picks=}')
