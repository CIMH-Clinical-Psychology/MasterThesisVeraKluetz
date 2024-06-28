#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:56 2024

This file contains all functions

@author: vera klütz
"""
import os
import sys
import settings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import warnings
import mne
import seaborn as sns
import autoreject
from joblib import Memory
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from tqdm import tqdm
import scipy



if sys.platform!='win32': # does not work on windows
    os.nice(1)

# define cached functions for faster execution
mem = Memory(settings.cachedir if settings.caching else None)  # only enable caching if wanted
# cached_func = mne.cache(mne.io.read_raw)


@mem.cache
def read_raw_filtered_cached(fif_filepath, highpass, lowpass, notch):
    raw = mne.io.read_raw_fif(fif_filepath, preload=True, verbose='INFO')
    raw.filter(highpass, lowpass, n_jobs=-1)
    if notch is not None:
        raw.notch_filter(notch)
    return raw


@mem.cache
def fit_apply_ica_cached(mne_obj, mne_obj_meg, ica_def, ica_ecg, ica_eog):
    """contains all ICA calculations,
    mne_obj contains EOG and ECG data + MEG/EEG
    mne_obj_meg contains only MEG channels (usually 306)
    expects a raw or epochs item
    returns the same object but with EOG and ECG components removed"""
    print('###  running ica')
    # work on a copy of the data, as some operations are in place
    mne_obj_meg = mne_obj_meg.copy()
    # manually save ICA results and hash them
    assert isinstance(ica_def, dict), 'ica_def must be dictionary with arguments'
    # define ICA within the function
    ica = mne.preprocessing.ICA(**ica_def)

    # there seems to be a problem here! the filter length is too large for the
    # epochs :-/ we might need to filter the raw object and then create epochs
    raw_ica = mne_obj_meg.copy().filter(1, None, n_jobs=-1)  # should be quite fast
    ica.fit(raw_ica, picks='meg')
    remove_components = []
    if ica_ecg == True:
        idx_ecg, scores_ecg = ica.find_bads_ecg(mne_obj) #, ch_name='BIO001')
        remove_components.extend(idx_ecg)
    if ica_eog == True:
        idx_eog, scores_eog = ica.find_bads_eog(mne_obj)
        remove_components.extend(idx_eog)

    print('removing the following components: {remove_components}')
    # apply ICA to the raw that only contains MEG data
    ica.apply(mne_obj_meg, exclude=remove_components)
    return mne_obj_meg, ica


def ignore_warnings():
    """surpress warning that naming convention is not met"""
    warnings.filterwarnings("ignore",
                            message=".*does not conform to MNE naming conventions. All raw files should end with raw.fif, raw_sss.fif, raw_tsss.fif*",
                            category=RuntimeWarning, module="mne")
    warnings.filterwarnings("ignore",
                            message=".*use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead*",
                            category=FutureWarning, module="seaborn")


def valid_filename(string):
    """strips a string of all characters that might be problems for filenames
    """
    #  strip all special characters from the filename, else it causes problems
    # all characters that are not alphanumeric will be replaced by _
    allowed = ['_', '-', '.', '/', '\\']
    filtered = ''
    for i, x in enumerate(string):
        if x.isalnum():  # is alphanumeric
            filtered += x
        elif x in allowed:  # is allowed character
            filtered += x
        elif i == 1 and x == ':':  # allow Windows drive names: eg. 'c:/'
            filtered += x
        else:
            filtered += ''
    return filtered


def autoreject_fit_cached(epochs):
    print('###  running autoreject')
    epochs_hash = hash(epochs)
    autoreject_cache_file = valid_filename(f'{settings.cachedir}/{epochs_hash}.autoreject')

    # if already computed, get the solution for this epochs object
    if os.path.exists(autoreject_cache_file):
        return autoreject.read_auto_reject(autoreject_cache_file)

    # else compute it
    ar = autoreject.AutoReject(n_jobs=-1, verbose=False, picks='meg',
                               random_state=99)
    ar.fit(epochs)  # weirdly enough need to pick here again
    ar.save(autoreject_cache_file, overwrite=True)
    return ar




def loop_through_participants(tmin, tmax, event_id_selection, highpass = 0.1, lowpass=50, notch = np.arange(50, 251, 50), picks = 'meg', fileending="", autoreject = True, ica_ecg = True, ica_eog = True):
    """Loops through all participants, finds stimulus events, creates epochs and saves them. Depending on the input of
    the function, the data also gets higpass-, lowpass-, and notch-filtered; also ica including eog and ecg rejection
    can be included as well as autoreject of bad epochs"""

    # creates a list of all participant numbers from 01 to 35 so that we can loop through them
    par_numbers = [str(i).zfill(2) for i in
                   range(1, 36)]  # for testing purposes we might use only 1 participant, so 2 instead of 36

    fileending = (f'_{fileending}' if fileending != "" else fileending)

    # loop through each participant's data
    for participant in par_numbers:

        if participant in ('25', '28', '31'):  # these are missing
            continue

        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(f'This is participant number {participant}')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        fif_filepath = settings.datadir + f"ERP-{participant}/ERP{participant}_initial_tsss_mc.fif"
        
        # read raw data
        try:
            print('###  loading and filtering data')
            # filter data already while loading :)
            raw = read_raw_filtered_cached(fif_filepath, highpass, lowpass, notch)
        except:
            warnings.warn(
                f"There have been problems with reading or filtering the raw data in the fif file of Participant number {participant}. Proceeding with next participant.")
            continue


        # check if we have more than 25 Minutes of Recordings included
        assert len(raw) > 60 * 25 * raw.info['sfreq']  # 50 sec * 25 min * sampling frequency

        # Set EOG/ECG channel
        ch_types = {'BIO001': 'ecg', 'BIO002': 'eog', 'BIO003': 'eog'}
        raw.set_channel_types({**ch_types})

        # -------------------- find events ---------------------------------------

        events = mne.find_events(raw, stim_channel='STI101', min_duration=3 / raw.info['sfreq'])
        assert len(events) == 1440, 'sanity check failed'

        # -------------------- create epochs -------------------------------------

        # creating epochs is instantaneous, so does not need to be cached :)
        # using `preload` prevents having to call "load_data" :)
        # the events=events needs all event markers, and with the event_id you
        # can subselecet which events to load
        epochs_orig = mne.Epochs(raw, events=events, event_id=event_id_selection,
                                 tmin=tmin, tmax=tmax, preload=True,
                                 baseline=None)  # don't subtract baseline

        epochs_meg = epochs_orig.copy().pick(picks)  # create copy that only contains MEG or only EOG data

        assert len(epochs_meg) == 144, f'sanity check failed, more or less than 144 epochs {len(epochs_meg)=}'
        print(f"Bad channels: {epochs_meg.info['bads']}")

        # Plot epochs
        # epochs.plot(show=False)

        # reject bad epochs automatically
        if autoreject== True:
            ar = autoreject_fit_cached(epochs_meg)
            # call to ar.transform should be relatively fast, no caching needed
            # there seems to be some bug that you can only enter channel data
            # of MAG and GRAD, weird! I think this is a bug with autoreject.
            epochs_meg = ar.transform(epochs_meg)

        # create downsampled epochs
        # sampling_rate = 200
        # epochs_resampled = epochs.resample(sampling_rate, npad="auto")

        # -------------------- independent component analysis ---------------------
        if (ica_ecg or ica_eog) == True:
            ica_method = 'fastica'
            n_components = 40  # todo: try with 50 maybe?
            random_state = 99
            # store definition in dictionary
            ica_def = dict(n_components=n_components, method=ica_method, random_state=random_state)
            epochs_meg, ica = fit_apply_ica_cached(epochs_orig, epochs_meg, ica_def, ica_ecg, ica_eog)

            # check ICA solution
            explained_var_ratio = ica.get_explained_variance_ratio(epochs_meg)
            for channel_type, ratio in explained_var_ratio.items():
                print(
                    f'Fraction of {channel_type} variance explained by all components: '
                    f'{ratio}'
                )

        # --------------------- save epochs ---------------------------------------
        if epochs_meg.times.flags['WRITEABLE']:
            # Set times array to read-only
            epochs_meg.times.setflags(write=False)

        print('###  saving epochs')
        filename_epoch = f'{participant=}_{event_id_selection=}_{tmin=}_{tmax=}{fileending}'
        filename_epoch = valid_filename(filename_epoch)
        epoch_file_path = os.path.join(settings.epochs_folderpath, f"{filename_epoch}-epo.fif")
        epochs_meg.save(epoch_file_path, fmt='double', overwrite=True)






@mem.cache
def read_epoch_cached_fif(full_filename):
    epochs = mne.read_epochs(full_filename)
    return epochs


def run_cv(clf, data_x_t, labels, n_splits=5):
    """outsourced crossvalidation function to run on a single timepoint,
    this way the function can be parallelized

    Parameters
    ----------
    clf : sklearn.Estimator
        any object having a .fit and a .predict function (Pipeline, Classifier).
    data_x_t : np.ndarray
        numpy array with shape [examples, features].
    labels : np.ndarray, list
        list or array of target variables.
    n_splits : int, optional
        number of splits. The default is 5.

    Returns
    -------
    accs or f1s : list
        list of accuracies or f1 scores, for each fold one.
    """
    warnings.filterwarnings("error", category=UserWarning)

    cv = StratifiedKFold(n_splits=n_splits)
    accs_f1s = []
    # Loop over each fold
    try:
        for k, (train_idx, test_idx) in enumerate(cv.split(data_x_t, labels)):
            x_train, x_test = data_x_t[train_idx], data_x_t[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # clf can also be a pipe object
            clf.fit(x_train, y_train)

            #model = StandardScaler().fit(x_train, y_train)

            preds = clf.predict(x_test)
            #preds = model.predict(x_test)

            if settings.output_metric == 'f1_score':
                f1 = f1_score(y_test, preds)
                accs_f1s.append(f1)
            else:
                # accuracy = mean of binary predictions
                acc = np.mean((preds == y_test))
                accs_f1s.append(acc)



    except UserWarning as e:
        if "The least populated class in y has only" in str(e):
            print(f"Skipping participant due to insufficient class members. {e} {y_test=} {y_train=}")
            return None
    return accs_f1s



def plot_epochs_per_participant(participants, list_num_epochs):
    '''create a 10x5 figure that shows the participant number on the x-axis and the amount of epochs per participant on the y-axis.
    The plot is saved in the plot folder and displayed immediately'''
    plt.figure(figsize=(10, 5))
    plt.bar(x=participants, height=list_num_epochs, width=0.7)

    plot_filename = os.path.join(settings.plot_folderpath,
                                 f"Epochs_per_participant_event_id{settings.event_id_selection}_tmin{settings.tmin}_tmax{settings.tmax}{settings.fileending}.png")
    plt.savefig(plot_filename)
    plt.show()



def plot_subj_into_big_figure(df_all, participant, ax, ax_bottom, random_chance=0.25):
    fig = ax.figure
    df_subj = df_all[df_all.participant==participant]
    times = df_subj.timepoint

    sns.lineplot(data=df_subj, x='timepoint', y=settings.output_metric, ax=ax)
    ax.hlines(random_chance, min(times), max(times), linestyle='--', color='gray')  # draw random chance line
    ax.set_title(f'{participant=}')
    # then plot a summary of all participant into the big plot
    ax_bottom.clear()  # clear axis from previous line
    sns.lineplot(data=df_all, x='timepoint', y=settings.output_metric, ax=ax_bottom)
    ax_bottom.hlines(random_chance, min(times), max(times), linestyle='--',
                     color='gray')  # draw random chance line
    ax_bottom.set_title(f'Mean of {len(df_all.participant.unique())} participants')
    fig.tight_layout()
    plt.pause(0.1)  # necessary for plotting to update


#def old_plot_subj_into_big_figure(fig, axs, ax_bottom, p, participant, epochs, df_subj, df_all):
#    '''plots a subject into the small axis and then updates the summary of all participants in the lower right plot'''
#    # first plot this participant into the small axis
#    ax = axs[p]  # select axis of this participant
#    sns.lineplot(data=df_subj, x='timepoint', y=settings.output_metric, ax=ax)
#    ax.hlines(0.25, min(epochs.times), max(epochs.times), linestyle='--', color='gray')  # draw random chance line
#    ax.set_title(f'{participant=}')
#    # then plot a summary of all participant into the big plot
#    ax_bottom.clear()  # clear axis from previous line
#    sns.lineplot(data=df_all, x='timepoint', y=settings.output_metric, ax=ax_bottom)
#    ax_bottom.hlines(0.25, min(epochs.times), max(epochs.times), linestyle='--',
#                     color='gray')  # draw random chance line
#    ax_bottom.set_title(f'Mean of {len(df_all.participant.unique())} participants')
#    fig.tight_layout()
#    plt.pause(0.1)  # necessary for plotting to update


def get_windows_power(windows, sfreq, axis=-1):
    ##todo: check once in a while if taking all windows at once instead of looping through them takes up too much memory
    w = scipy.fftpack.rfft(windows, axis=axis)
    freqs = scipy.fftpack.rfftfreq(w.shape[-1], d=1 / sfreq)
    # w.shape = (144, 306, 500)
    # freqs = [0, 2, 4, ..., 500] freqs belonging to each index of w
    power = np.abs(w) ** 2 / (len(w[-1]) * sfreq)  # convert to spectral power
    return power, freqs

    # previous get_windows_power code
    #'''returns a list of lists of e.g. alpha power. Each alhpa power has the shape (epochs x channel)'''
    #windows_power = []
    ## convert to frequency domain via rfft (we don't need the imaginary part)
    #w = scipy.fftpack.rfft(windows, axis=-1)
    #freqs = scipy.fftpack.rfftfreq(w.shape[-1], d=1 / sfreq)
    ## w.shape = (144, 306, 16, 500)
    ## freqs = [0, 2, 4, ..., 500] freqs belonging to each index of w
    #power = np.abs(w) ** 2 / (len(w[-1]) * sfreq)  # convert to spectral power
    ## e.g. w[0, 4, 2] = power of epoch 0, channel 4 for frequency bin 4 Hz
    ## now you can use these to calculate brain bands, e.g.
    #alpha = [8, 14]
    #alpha_idx1 = np.argmax(freqs > alpha[0])
    #alpha_idx2 = np.argmax(freqs > alpha[1])
    #alpha_power = power[:, :, :, alpha_idx1:alpha_idx2].mean(-1)
    ## alpha_power .shape = (144, 306, 6)
    ## for each epoch, for each channel, for each window, one alpha power value
    #windows_power += [alpha_power[:,:,i] for i in np.arange(windows.shape[2])]  # add alpha power values of each window to list
#
    #return windows_power


    ## even more old get_windwos_power code for looping through each window individually
    #windows_power = []
    ## loop through windows
    #for i in tqdm(range(windows.shape[2])):
    #    # this loop can for sure be paralellized and also should be a function with parameters and not a loop
    #    win = windows[:, :, i, :]
    #    # convert to frequency domain via rfft (we don't need the imaginary part)
    #    w = scipy.fftpack.rfft(win, axis=-1)
    #    freqs = scipy.fftpack.rfftfreq(w.shape[-1], d=1 / sfreq)
    #    # w.shape = (144, 306, 500)
    #    # freqs = [0, 2, 4, ..., 500] freqs belonging to each index of w
    #    power = np.abs(w) ** 2 / (len(w[-1]) * sfreq)  # convert to spectral power
    #    # e.g. w[0, 4, 2] = power of epoch 0, channel 4 for frequency bin 4 Hz
    #    # now you can use these to calculate brain bands, e.g.
    #    alpha = [8, 14]
    #    alpha_idx1 = np.argmax(freqs > alpha[0])
    #    alpha_idx2 = np.argmax(freqs > alpha[1])
    #    alpha_power = power[:, :, alpha_idx1:alpha_idx2].mean(-1)
    #    # alpha_power .shape = (144, 306),
    #    # for each epoch, for each channel one alpha power value
    #    windows_power += [alpha_power]  # add alpha power values of this window to list

    #return windows_power




def get_bands_power(windows, sfreq, bands, axis=-1):
    '''returns an array of shape (n_bands x n_epochs x n_channels x n_windows)'''
    power, freqs = get_windows_power(windows, sfreq, axis=axis)
    bands_power = []
    for min_freq, max_freq in bands:
        idx1 = np.argmax(freqs >= min_freq)  # not completely sure if >= or >
        idx2 = np.argmax(freqs > max_freq)  # not completely sure if >= or >
        mean_power = power.take(indices=range(idx1, idx2), axis=axis).mean(axis)
        bands_power.append(mean_power)
    return np.array(bands_power)



def decode_features(windows_power, labels, participant, pipe, timepoints, n_splits=5, n_jobs=-1):
    '''
    performs cross validation with a classifier set in the settings and the StandardScaler

    input:
    windows_power: shape(epochs, bands*channels, windows)
    labels: 1D with the target values
    participant: string with participant number

    returns: pandas DataFrame for one subject with the attributes: participant, timepoint, accuracy/f1 score, split
    '''

    print('Decoding starts')

    df_subj = pd.DataFrame()  # save results for this participant temporarily in a df

    # Access and change the random_state parameter for the new classifier
    pipe.set_params(classifier__random_state=99)

    # calculate all the timepoints in parallel massively speeds up calculation
    tqdm_loop = tqdm(np.arange(windows_power.shape[2]), desc='calculating timepoints')

    res=[]
    #for n_window in range(13):
    #    res = run_cv(pipe, windows_power[:, :, n_window], labels, n_splits=n_splits)
    try:
        res = Parallel(n_jobs)(delayed(run_cv)(pipe, windows_power[:,:,n_window], labels, n_splits=n_splits) for n_window in tqdm_loop)
        # res will return a list for each job. If there are too few class members in a class, None will be returned
        if None in res:
            return None
    except Exception as e:
        warnings.warn(
            f"There was an error with participant number {participant}. Maybe there were too few epochs for cross validaton. {e}")
        return None


    timepoints = np.linspace(timepoints[0], timepoints[-1], windows_power.shape[2])
    timepoints = [f"{x:.1f}" for x in timepoints]
    timepoints = np.array(timepoints, dtype=float)
    # save result of the folds in a dataframe.
    # the unravelling of the res object can be a bit confusing.
    # res is a list, each entry has 5 accuracy/f1 score values, for each fold one.
    # we need to now make sure that in the dataframe each row has one
    # accuracy/ f1 score value and it's assigned timepoint, and also an indicator of the
    # fold number
    df_subj = pd.DataFrame({'participant': participant,
                            'timepoint': np.repeat(timepoints, n_splits),
                            settings.output_metric: np.ravel(res),
                            'split': list(range(n_splits)) * len(res)
                            })

    return df_subj


#def reshape_windows_power(windows_power, swap, reshape_param):
#    reshaped_windows_power = windows_power.swapaxes(*swap).reshape([windows_power.shape[d] if d!=-1 else -1 for d in reshape_param])
#
#
#    #'''reshape windows_power from ( bands, epochs, channels, windows) to (epochs, bands * channels, windows)'''
#    ## Get the original shape
#    #n_bands, n_epochs, n_channels, n_windows = windows_power.shape
#
#    ## Reshape to the desired shape
#    #reshaped_windows_power = windows_power.transpose(1, 0, 2, 3).reshape(n_epochs, n_bands * n_channels, n_windows)
#
#    return reshaped_windows_power


def pca_fit_transform(data, n_components=200):
    pca = PCA(n_components, random_state=99)
    data = pca.fit_transform(data)
    return data


def remove_button_not_pressed(data_x, labels, buttons):
    idx_to_remove = []
    for i in range(len(labels)):
        if buttons[i] == False:
            idx_to_remove.append(i)
    data_x = np.delete(data_x, idx_to_remove, axis=0)
    labels = np.delete(labels, idx_to_remove)
    return data_x, labels


#def reshape_windows_power_for_pca(windows_power):
#    '''input: shape(epochs, bands*channels, windows)
#    output: shape(epochs*windows, bands*channels)
#    '''
#
#    n_epochs, n_bands_channels, n_windows = windows_power.shape
#    reshaped_windows_power = windows_power.transpose(0,2,1).reshape(n_epochs * n_windows, n_bands_channels)
#
#    return reshaped_windows_power


#def reshape_windows_power_after_pca(pca_windows_power, n_windows):
#    '''input: windows_power with shape (epochs*windows, bands*channels)
#    output: shape(epochs, bands*channels, windows)'''
#    n_epochs_windows, n_bands_channels = pca_windows_power.shape
#    n_epochs = int(n_epochs_windows/n_windows)
#
#    #windows_power = pca_windows_power.reshape(epochs, bands_channels, n_windows)
#    windows_power = pca_windows_power.reshape(n_epochs, n_windows, n_bands_channels).transpose(0, 2, 1)
#
#    return windows_power