# some code taken and altered from https://www.geeksforgeeks.org/ml-logistic-regression-using-python/

import pandas as pd
import os
from joblib import Memory
from joblib import Parallel, delayed
import time
import settings
import utils
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mne
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

os.nice(1)  # make sure we're not clogging the CPU
plt.ion()

# -------------------- user specifics ----------------------------------------
# folderpath, where the epochs are stored
# take the following parameters from the stored filenames!
epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")
event_id_selection = 20
tmin = -0.5
tmax = 1

# -------------------- cached functions --------------------------------------
# measure code execution
start_time = time.time()

mem = Memory(settings.cachedir)


@mem.cache
def read_epoch_cached_csv(full_filename):
    df = pd.read_csv(full_filename)
    return df


@mem.cache
def read_epoch_cached_fif(full_filename):
    epochs = mne.read_epochs(full_filename)
    return epochs


@mem.cache
def load_exp_data_cached(participant):
    df_subj = utils.load_exp_data(int(participant))
    return df_subj

def run_cv(clf, data_x_t, gif_pos, n_splits=5):
    """outsourced crossvalidation function to run on a single timepoint,
    this way the function can be parallelized

    Parameters
    ----------
    clf : sklearn.Estimator
        any object having a .fit and a .predict function (Pipeline, Classifier).
    data_x_t : np.ndarray
        numpy array with shape [examples, features].
    gif_pos : np.ndarray, list
        list or array of target variables.
    n_splits : int, optional
        number of splits. The default is 5.

    Returns
    -------
    accs : list
        list of accuracies, for each fold one.
    """

    cv = StratifiedKFold(n_splits=n_splits)

    accs = []
    # Loop over each fold
    for k, (train_idx, test_idx) in enumerate(cv.split(data_x_t, gif_pos)):
        x_train, x_test = data_x_t[train_idx], data_x_t[test_idx]
        y_train, y_test = gif_pos[train_idx], gif_pos[test_idx]

        # clf can also be a pipe object
        clf.fit(x_train, y_train)

        #model = StandardScaler().fit(x_train, y_train)

        preds = clf.predict(x_test)
        #preds = model.predict(x_test)

        # accuracy = mean of binary predictions
        acc = np.mean((preds == y_test))
        accs.append(acc)
    return accs


#%%
missing = [25, 28, 31]
participants = [str(i).zfill(2) for i in range(1, 36) if not i in missing]

 # small plots for individual participants and one bottom plot for a summary
fig, axs, ax_bottom = utils.make_fig(n_axs=len(participants), n_bottom=1)

# -------------------- read data ------------------------------------------------
# loop through each participants number from 01 to 35

df_all = pd.DataFrame()  # save results of the calculations in a dataframe

for p, participant in enumerate(participants):  # (6, 7)]: # for testing purposes we might use only 1 participant, so 2 instead of 36

    if participant in ():  # these are missing
        continue

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'This is participant number {participant}')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    filename_epoch = f'participant{participant}_event_id_selection{event_id_selection}_tmin{tmin}_tmax{tmax}'
    full_filename_csv = os.path.join(epochs_folderpath, f"{filename_epoch}-epo.csv")
    full_filename_fif = os.path.join(epochs_folderpath, f"{filename_epoch}-epo.fif")

    # read the epochs
    try:
        epochs = read_epoch_cached_fif(full_filename_fif)
    except:
        print(f"Epochs: There is no epochs file for participant number {participant}. \n "
              f"If you expected the file to exist, please check the parameters given for the filename creation. \n "
              f"Proceeding with next participant.\n")
        continue

    # read the "solution"/target, in which quadrant it was shown
    try:
        df_subj = load_exp_data_cached(participant)
    except:
        print(f"Quadrants: There is no quadrant information for participant number {participant}. \n "
              f"If you expected the file to exist, check in the EMO_REACT_PRESTUDY in the participants_data folder if the csv file exists.\n "
              f"Make sure that the file is not currently opened by another program!!\n "
              f"Proceeding with next participant.\n")
        continue

    # -------------------- create target containing the gif positions --------------
    # only select the targets, that belong to the epochs that have not been rejected
    df_subj_gif = df_subj['gif_position']

    all_poss_epoch_idx = np.arange(start=2, stop=144 * 10, step=10)
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
    epochs.resample(100, n_jobs=-1, verbose='WARNING')  # for now resample to 100 to speed up computation
    data_x = epochs.get_data()

    if len(data_x)<5:
        axs[p].text(0.5, 0.5, f'{participant=} has only {len(data_x)} epochs, skip')
        continue  # some participants have very few usable epochs

    df_subj = pd.DataFrame()  # save results for this participant temporarily in a df

    # could also use RandomForest, as it's more robust, should always work out of the box
    # C parameter is important to set regularization, might overregularize else
    clf = LogisticRegression(C=10000, max_iter=1000)
    pipe = Pipeline(steps=[('scaler', StandardScaler()),
                    ('classifier', clf)])
    # calculate all the timepoints in parallel massively speeds up calculation
    n_splits = 5
    tqdm_loop = tqdm(range(len(epochs.times)), total=len(epochs.times), desc='calculating timepoints')
    res = Parallel(-1)(delayed(run_cv)(pipe, data_x[:, :, t],
                                       gif_pos, n_splits=n_splits) for t in tqdm_loop)

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
    ax_bottom.set_title(f'Mean of {len(df_all.participants.unique())} participants')
    plt.pause(0.1)  # necessary for plotting to update

    # print('one participant over')
    # print(f'highest number: {np.max(accuracy_number)}')
    # print(f'lowest number: {np.min(accuracy_number)}')
    # print(f'average number: {np.mean(accuracy_number)}')
    # print('hi')  # hello!

    # kf = KFold(n_splits=5, shuffle=True, random_state=99)

        # split data into training and testing sets
        #x_train, x_test, y_train, y_test = train_test_split(data_x_t, gif_pos, test_size=0.2, random_state=99)
#
        ## standardize features
        ## converts MEG values to a 0 mean and a standard deviation of 1
        #scaler = StandardScaler()
        #x_train = scaler.fit_transform(x_train)
        #x_test = scaler.transform(x_test)
#
        ## train the Logistic Regression model/classifier
        #model = LogisticRegression()
        #model.fit(x_train, y_train)
#
        ## evaluate the model
        #y_pred = model.predict(x_test)
        #accuracy = accuracy_score(y_test, y_pred)
        #print("Accuracy: {:.2f}%".format(accuracy * 100))
        #accs += [accuracy]



end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.3f} seconds")
