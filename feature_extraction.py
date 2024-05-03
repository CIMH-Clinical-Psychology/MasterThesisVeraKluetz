

import pandas as pd
import os
from joblib import Memory
from joblib import Parallel, delayed
import time
import settings
import utils
import functions
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mne
import matplotlib.pyplot as plt
import scipy
import numpy as np
import seaborn as sns
from tqdm import tqdm
import warnings
from mne_features.feature_extraction import extract_features

os.nice(1)  # make sure we're not clogging the CPU
plt.ion()

# -------------------- user specifics ----------------------------------------
# folderpath, where the epochs are stored
# take the following parameters from the stored filenames!
epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")
plot_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Plots/")
event_id_selection = 10
tmin = -2.5
tmax = 1

fileending = ""

# either choose "RandomForest" or "LogisticRegression" as classifier
classifier = "LogisticRegression"
#-------------------- end of user specifics



# measure code execution
start_time = time.time()

# loop through each participants number from 01 to 35

missing = [25, 28, 31]
participants = [str(i).zfill(2) for i in range(1, 36) if not i in missing]

for p, participant in enumerate(participants):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'This is participant number {participant}')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    filename_epoch = f'participant{participant}_event_id_selection{event_id_selection}_tmin{tmin}_tmax{tmax}{fileending}'
    full_filename_fif = os.path.join(epochs_folderpath, f"{filename_epoch}-epo.fif")

    # read the epochs
    try:
        epochs_fif = functions.read_epoch_cached_fif(full_filename_fif)
    except:
        print(f"Epochs: There is no epochs file for participant number {participant}. \n "
              f"If you expected the file to exist, please check the parameters given for the filename creation. \n "
              f"Proceeding with next participant.\n")
        continue

    sfreq=epochs_fif.info['sfreq']

    epochs = epochs_fif.get_data() #shape(144, 306, 3501)
    windows = utils.extract_windows(epochs, sfreq, win_size=0.5, step_size=0.25) #todo: is step size to big? only 50% overlap
    #shape(144, 306, 13, 500)

    windows_power = []

    # loop through windows
    for i in tqdm(range(windows.shape[2])):
        # this loop can for sure be paralellized and also should be a function with parameters and not a loop
        win = windows[:,:,i,:]
        # convert to frequency domain via rfft (we don't need the imaginary part)
        w = scipy.fftpack.rfft(win, axis=-1)
        freqs = scipy.fftpack.rfftfreq(w.shape[-1], d=1 / sfreq)
        # w.shape = (144, 306, 500)
        # freqs = [0, 2, 4, ..., 500] freqs belonging to each index of w
        power = np.abs(w) ** 2 / (len(w[-1]) * sfreq)  # convert to spectral power
        # e.g. w[0, 4, 2] = power of epoch 0, channel 4 for frequency bin 4 Hz
        # now you can use these to calculate brain bands, e.g.
        alpha = [8, 14]
        alpha_idx1 = np.argmax(freqs > alpha[0])
        alpha_idx2 = np.argmax(freqs > alpha[1])
        alpha_power = power[:, :, alpha_idx1:alpha_idx2].mean(-1)
        # alpha_power .shape = (144, 306),
        # for each epoch, for each channel one alpha power value
        windows_power += [alpha_power]  # add alpha power values of this window to list

    # just for the fun of it, plot mean alpha power over time for each channel
    # there should a streak of occipital channels showing higher alpha
    plt.imshow(np.mean(windows_power, axis=1).T, aspect='auto')
    plt.xlabel('timestep of window')
    plt.ylabel('channel number')



    print('hi')

