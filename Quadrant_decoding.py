

import pandas as pd
import os
from joblib import Memory
from joblib import Parallel, delayed
import time
import settings
import utils
import functions

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

os.nice(1)  # make sure we're not clogging the CPU
plt.ion()

# -------------------- user specifics ----------------------------------------
# folderpath, where the epochs are stored
epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")
plot_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Plots/")
# set if you want a plot to be shown and saved of how many epochs there are per participant
plot_epochs_per_participant = True

# take the following parameters from the stored filenames!

event_id_selection = 10
tmin = -2.5
tmax = 1
# for the fileending, choose between the following:
# ""  "_noIcaEogRejection"   "_minimalPreprocessing"   "_EOG-only"
fileending = ""
# either choose "RandomForest" or "LogisticRegression" as classifier
classifier = "LogisticRegression"
# ---------------------- end of user specifics --------------------------


# measure code execution
start_time = time.time()

#%%
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
for p, participant in enumerate(participants):  # (6, 7)]: # for testing purposes we might use only 1 participant, so 2 instead of 36

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'This is participant number {participant}')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    filename_epoch = f'participant{participant}_event_id_selection{event_id_selection}_tmin{tmin}_tmax{tmax}{fileending}'
    full_filename_fif = os.path.join(epochs_folderpath, f"{filename_epoch}-epo.fif")

    # read the epochs
    try:
        epochs = functions.read_epoch_cached_fif(full_filename_fif)
    except:
        print(f"Epochs: There is no epochs file for participant number {participant}. \n "
              f"If you expected the file to exist, please check the parameters given for the filename creation. \n "
              f"Proceeding with next participant.\n")
        continue
    # count epochs per participant
    list_num_epochs.append(len(epochs))


    # read the "solution"/target, in which quadrant it was shown
    try:
        df_subj = utils.load_exp_data(participant)
    except:
        print(f"Quadrants: There is no quadrant information for participant number {participant}. \n "
              f"If you expected the file to exist, check in the EMO_REACT_PRESTUDY in the participants_data folder if the csv file exists.\n "
              f"Make sure that the file is not currently opened by another program!!\n "
              f"Proceeding with next participant.\n")
        continue

    # -------------------- create target containing the gif positions --------------
    # only select the targets, that belong to the epochs that have not been rejected
    df_subj_gif = df_subj['gif_position']

    # all_poss_epoch_idx = np.arange(start=2, stop=144 * 10, step=10)
    lowest_epoch_idx = epochs.selection[0]
    lowest_possible_epoch_idx = lowest_epoch_idx % 10
    all_poss_epoch_idx = np.arange(start=lowest_possible_epoch_idx, stop=144 * 10, step=10)

    true_epoch_idx = epochs.selection

    gif_pos_chars = []
    for i in np.arange(144):
        if all_poss_epoch_idx[i] in true_epoch_idx:
            gif_pos_chars.append(df_subj_gif[i])

    # convert letter to number
    char_to_num = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    #char_to_num = {'A': 0, 'B': 0.33, 'C': 0.66, 'D': 1}
    gif_pos = [char_to_num[i] for i in gif_pos_chars]
    gif_pos = np.array(gif_pos)
    # todo: Does the char to num even make a difference?

    # -------------------- loop through each timepoint to train and test the model---------
    #epochs.resample(100, n_jobs=-1, verbose='WARNING')  # for now resample to 100 to speed up computation
    data_x = epochs.get_data()

    if len(data_x)<20:
        axs[p].text(0.1, 0.4, f'{participant=} \n {len(data_x)} epochs, skip')
        continue  # some participants have very few usable epochs

    df_subj = pd.DataFrame()  # save results for this participant temporarily in a df

    # could also use RandomForest, as it's more robust, should always work out of the box
    # C parameter is important to set regularization, might overregularize else
    if classifier == "LogisticRegression":
        clf = LogisticRegression(C=10, max_iter=1000, random_state=99)
    elif classifier == "RandomForest":
        clf = RandomForestClassifier(random_state=99)
    else:
        print("No valid classifier was selected")
        exit()

    pipe = Pipeline(steps=[('scaler', StandardScaler()),
                    ('classifier', clf)])
    # calculate all the timepoints in parallel massively speeds up calculation
    n_splits = 5
    tqdm_loop = tqdm(range(len(epochs.times)), total=len(epochs.times), desc='calculating timepoints')
    try:
        res = Parallel(-1)(delayed(functions.run_cv)(pipe, data_x[:, :, t],
                                    gif_pos, n_splits=n_splits) for t in tqdm_loop)
    except:
        print(f"There was an error with participant number {participant}. Maybe there were too few epochs for cross validaton.")
        continue

    # save result of the folds in a dataframe.
    # the unravelling of the res object can be a bit confusion.
    # res is a list, each entry has 5 accuracy values, for each fold one.
    # we need to now make sure that in the dataframe each row has one
    # accuracy value and it's assigned timepoint, and also an indicator of the
    # fold number
    df_subj = pd.DataFrame({'participant': participant,
                            'timepoint': np.repeat(epochs.times, n_splits),
                            'accuracy': np.ravel(res),
                            'split': list(range(n_splits))*len(res)
                            })


    # append to dataframe holding all data
    df_all = pd.concat([df_all, df_subj])

    # first plot this participant into the small axis
    ax = axs[p]  # select axis of this participant
    sns.lineplot(data=df_subj, x='timepoint', y='accuracy', ax=ax)
    ax.hlines(0.25, min(epochs.times), max(epochs.times), linestyle='--', color='gray')  # draw random chance line
    ax.set_title(f'{participant=}')
    # then plot a summary of all participant into the big plot
    ax_bottom.clear()  # clear axis from previous line
    sns.lineplot(data=df_all, x='timepoint', y='accuracy', ax=ax_bottom)
    ax_bottom.hlines(0.25, min(epochs.times), max(epochs.times), linestyle='--', color='gray')  # draw random chance line
    ax_bottom.set_title(f'Mean of {len(df_all.participant.unique())} participants')
    fig.tight_layout()
    plt.pause(0.1)  # necessary for plotting to update


plot_filename = os.path.join(plot_folderpath, f"quadrant_decoding_{classifier}_{event_id_selection=}_{tmin=}_{tmax=}{fileending}.png")
fig.savefig(plot_filename)


if plot_epochs_per_participant:
    plt.figure(figsize=(10,5))
    plt.bar(x=participants, height=list_num_epochs, width = 0.7)

    plot_filename = os.path.join(plot_folderpath, f"Epochs_per_participant_{event_id_selection=}_{tmin=}_{tmax=}{fileending}.png")
    plt.savefig(plot_filename)
    plt.show()

end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.3f} seconds")
