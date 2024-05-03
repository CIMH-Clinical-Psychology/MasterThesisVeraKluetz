#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:56 2024

This file contains all functions

@author: vera klütz
"""
import os
import settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import mne
import seaborn as sns
import time
import autoreject
from joblib import Memory
from sklearn.model_selection import StratifiedKFold

os.nice(1)

# define cached functions for faster execution
mem = Memory(settings.cachedir if settings.caching else None)  # only enable caching if wanted
# cached_func = mne.cache(mne.io.read_raw)


@mem.cache
def read_raw_filtered_cached(fif_filepath, highpass, lowpass, notch):
    raw = mne.io.read_raw_fif(fif_filepath, preload=True, verbose='INFO')
    raw.filter(highpass, lowpass, n_jobs=-1)
    if notch.all() != None:
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




def loop_through_participants(tmin, tmax, event_id_selection, highpass = 0.1, lowpass=50, notch = np.arange(50, 251, 50), picks = 'meg', fileending=None, autoreject = True, ica_ecg = True, ica_eog = True):
    """Loops through all participants, finds stimulus events, creates epochs and saves them. Depending on the input of
    the function, the data also gets higpass-, lowpass-, and notch-filtered; also ica including eog and ecg rejection
    can be included as well as autoreject of bad epochs"""

    # creates a list of all participant numbers from 01 to 35 so that we can loop through them
    par_numbers = [str(i).zfill(2) for i in
                   range(1, 36)]  # for testing purposes we might use only 1 participant, so 2 instead of 36

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
        print('###  saving epochs')
        fileending = (f'_{fileending}' if fileending != None else fileending)
        filename_epoch = f'{participant=}_{event_id_selection=}_{tmin=}_{tmax=}{fileending}'
        filename_epoch = valid_filename(filename_epoch)
        epoch_file_path = os.path.join(settings.epochs_folderpath, f"{filename_epoch}-epo.fif")
        epochs_meg.save(epoch_file_path, fmt='double', overwrite=True)






@mem.cache
def read_epoch_cached_fif(full_filename):
    epochs = mne.read_epochs(full_filename)
    return epochs


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




