# some code taken and altered from https://www.geeksforgeeks.org/ml-logistic-regression-using-python/

import pandas as pd
import os
from joblib import Memory
import time
import settings
import utils
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mne
import numpy as np

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


# -------------------- read data ------------------------------------------------
# loop through each participants number from 01 to 35
for participant in [str(i).zfill(2) for i in
                    range(34, 36)]:  # (6, 7)]: # for testing purposes we might use only 1 participant, so 2 instead of 36

    if participant in ('25', '28', '31'):  # these are missing
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
    epochs = epochs.get_data()

    # shape (31, 306, 1501)

    #model = LogisticRegression()
    #pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe = Pipeline(steps=[('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(max_iter=1000))])
    accuracy_number = []

    n_timepoints = epochs.shape[2]
    for t in range(n_timepoints):
        data_x_t = epochs[:, :, t]
        # todo: find right way to normalize the input data
        # todo: maybe try MinMaxScaler from sklearn
        data_x_t = normalize(data_x_t)
        # initialize cross validator
        cv = StratifiedKFold(n_splits=5)

        accs = []
        # Loop over each fold
        for train_idx, test_idx in cv.split(data_x_t, gif_pos):
            x_train, x_test = data_x_t[train_idx], data_x_t[test_idx]
            y_train, y_test = gif_pos[train_idx], gif_pos[test_idx]



            pipe.fit(x_train, y_train)
            
            #model = StandardScaler().fit(x_train, y_train)

            preds = pipe.predict(x_test)
            #preds = model.predict(x_test)

            acc = (preds == y_test)

            accs.append(acc)
        accuracy_number += [np.count_nonzero(accs) / len(accs)]
    
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

    print('one participant over')
    print(f'highest number: {np.max(accuracy_number)}')
    print(f'lowest number: {np.min(accuracy_number)}')
    print(f'average number: {np.mean(accuracy_number)}')
    print('hi')

end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.3f} seconds")
