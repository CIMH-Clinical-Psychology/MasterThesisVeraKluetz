#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vera.klütz
"""

import pandas as pd
import os
from joblib import Parallel, delayed
import time
import settings
import utils
import functions
import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

os.nice(1)  # make sure we're not clogging the CPU
plt.ion()

# measure code execution
start_time = time.time()

missing = [25, 28, 31]
participants = [str(i).zfill(2) for i in range(1, 36) if not i in missing]

# small plots for individual participants and one bottom plot for a summary
fig, axs, ax_bottom = utils.make_fig(n_axs=len(participants),
                                     n_bottom=[0, 1],
                                     figsize=[14, 14])

# create list to store the amount of epochs per participant
list_num_epochs = []

# -------------------- read data ------------------------------------------------
# loop through each participants number from 01 to 35

df_all = pd.DataFrame()  # save results of the calculations in a dataframe
for p, participant in enumerate(
        participants):  # (6, 7)]: # for testing purposes we might use only 1 participant, so 2 instead of 36

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'This is participant number {participant}')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    epochs, labels = utils.get_quadrant_data(participant)
    #epochs, labels = utils.get_valence_data(participant)
    #epochs, labels = utils.get_nonsubj_valence_data(participant)
    if epochs is None or labels is None:
        continue

    # count epochs per participant
    list_num_epochs.append(len(epochs))

    # -------------------- loop through each timepoint to train and test the model---------
    # epochs.resample(100, n_jobs=-1, verbose='WARNING')  # for now resample to 100 to speed up computation
    data_x = epochs.get_data(copy=False)

    # if there are less than 20 epochs, skip this participant
    if len(data_x) < 20:
        axs[p].text(0.1, 0.4, f'{participant=} \n {len(data_x)} epochs, skip')
        continue  # some participants have very few usable epochs

    df_subj = pd.DataFrame()  # save results for this participant temporarily in a df


    # calculate all the timepoints in parallel massively speeds up calculation
    n_splits = 3
    tqdm_loop = tqdm(range(len(epochs.times)), total=len(epochs.times), desc='calculating timepoints')
    try:
        res = Parallel(-1)(delayed(functions.run_cv)(settings.pipe, data_x[:, :, t],
                                                     labels, n_splits=n_splits) for t in tqdm_loop)
    except:
        warnings.warn(
            f"There was an error with participant number {participant}. Maybe there were too few epochs for cross validaton.")
        continue

    # save result of the folds in a dataframe.
    # the unravelling of the res object can be a bit confusing.
    # res is a list, each entry has 5 accuracy values, for each fold one.
    # we need to now make sure that in the dataframe each row has one
    # accuracy value and it's assigned timepoint, and also an indicator of the
    # fold number
    df_subj = pd.DataFrame({'participant': participant,
                            'timepoint': np.repeat(epochs.times, n_splits),
                            'accuracy': np.ravel(res),
                            'split': list(range(n_splits)) * len(res)
                            })

    # append to dataframe holding all data
    df_all = pd.concat([df_all, df_subj])

    # update figure with subject
    functions.plot_subj_into_big_figure(df_all, participant, axs[p], ax_bottom, random_chance=0.2)

plot_filename = os.path.join(settings.plot_folderpath,
                             #f"quadrant_decoding_{settings.classifier_name}_event_id{settings.event_id_selection}_tmin{settings.tmin}_tmax{settings.tmax}{settings.fileending}.png")
                                f"emo_decoding_{settings.classifier_name}_event_id{settings.event_id_selection}_tmin{settings.tmin}_tmax{settings.tmax}{settings.fileending}.png")
fig.savefig(plot_filename)

end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.3f} seconds")

# create and save a plot that shows how many epochs there are per participant
#functions.plot_epochs_per_participant(participants, list_num_epochs)

