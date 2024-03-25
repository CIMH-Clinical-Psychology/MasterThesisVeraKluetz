### code taken and altered from https://github.com/natmegsweden/meeg_course_mne/blob/master/tutorial_01a_preprocessing.md
### as well as from Alper Koelgesiz, on GitHub MEGInternshipAlper/EEG_Many_Alper/Preprocessing.py
### 22.02.2024
### vera.kluetz@zi-mannheim.de

import os
import mne
import time
from joblib import Memory
import warnings
import matplotlib.pyplot as plt
import autoreject
import os
import pandas as pd
import numpy as np
import settings

# increase the niceness of the process on linux, i.e. give away  cpu resources
# to other important processes if they need them (giving itself lower priority)
os.nice(1)


# -------------------- user specifics ----------------------------------------

# set cache directory for faster execution with joblib Memory package

# set file path, where the data can be found

#set folderpath, where the resulting epochs should be stored
epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")
#(f"/home/vera.kluetz/epochs/")

# -------------------- initial setup and function definition -----------------

# Use the 'TkAgg' backend for interactive plotting
#plt.switch_backend('TkAgg')

# measure code execution
start_time = time.time()

# surpress warning that naming convention is not met
warnings.filterwarnings("ignore",
                        message=".*does not conform to MNE naming conventions. All raw files should end with raw.fif, raw_sss.fif, raw_tsss.fif*",
                        category=RuntimeWarning, module="mne")

# define cached functions for faster execution
mem = Memory(settings.cachedir if settings.caching else None)  # only enable caching if wanted
# cached_func = mne.cache(mne.io.read_raw)
@mem.cache
def read_raw_filtered_cached(fif_filepath, highpass=0.1, lowpass=49):
    raw = mne.io.read_raw_fif(fif_filepath, preload=True, verbose='INFO')
    raw.filter(highpass, lowpass, n_jobs=-1)
    return raw


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
        elif i==1 and x==':':  # allow Windows drive names: eg. 'c:/'
            filtered += x
        else:
            filtered += ''
    return filtered

@mem.cache
def fit_apply_ica_cached(mne_obj, mne_obj_meg, ica_def):
    """contains all ICA calculations,
    mne_obj contains EOG and ECG data + MEG/EEG
    mne_obj_meg contains only EOG and ECG data
    expects a raw or epochs item
    returns the same object but with EOG and ECG components removed"""
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
    idx_ecg, scores_ecg = ica.find_bads_ecg(mne_obj, ch_name='BIO001')
    idx_eog, scores_eog = ica.find_bads_eog(mne_obj)
    remove_components = idx_ecg + idx_eog

    print('removing the following components: {remove_components}')
    # apply ICA to the raw that only contains MEG data
    ica.apply(mne_obj_meg, exclude=remove_components)
    return mne_obj_meg, ica

def autoreject_fit_cached(epochs):
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


# -------------------- load data ---------------------------------------------

# creates a list of all participant numbers from 01 to 35 so that we can loop through them
par_numbers = [str(i).zfill(2) for i in
               range(1, 36)]  # for testing purposes we might use only 1 participant, so 2 instead of 36

# loop through each participant's data
for participant in par_numbers:

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'This is participant number {participant}')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    fif_filepath = settings.datadir + f"ERP-{participant}/ERP{participant}_initial_tsss_mc.fif"

    # read raw data
    try:
        print('###  loading and filtering data')
        # filter data already while loading :)
        raw = read_raw_filtered_cached(fif_filepath)
    except:
        print(
            f"Participant number {participant} does not exist or there has been problems with reading it's file. Proceeding with next participant.")
        continue
    # raw_copy = raw.copy()

    # check if we have more than 25 Minutes of Recordings included
    assert len(raw) > 60 * 25 * raw.info['sfreq']  # 50 sec * 25 min * sampling frequency

    # Set EOG/ECG channel
    ch_types = {'BIO001':'ecg', 'BIO002': 'eog', 'BIO003': 'eog'}
    raw.set_channel_types({**ch_types})

    # -------------------- filtering -----------------------------------------

    # low- and highpass data
    # raw.filter(0.1, 50)  # , method='fir')
    # notch filter
    # raw.notch_filter(np.arange(50, 251, 50))
    # todo: is this clever?
    # raw.filter(0.1, 49)

    # -------------------- find events ---------------------------------------

    events = mne.find_events(raw, stim_channel='STI101', min_duration=3 / raw.info['sfreq'])

    # set event IDs
    event_id = {'trigger_preimage': 10,
                'trigger_gif_onset': 20,
                'trigger_gif_offset': 30,
                'trigger_fixation': 99,
                'trigger_valence_start': 101,
                'trigger_arousal_start': 102,
                'trigger_flanker_start': 104}

    # only select events with a certain trigger
    event_id_selection = event_id['trigger_gif_onset']
    # events = eve[eve[:, 2] == 20]

    assert len(events)==1440, 'sanity check failed'

    # -------------------- create epochs -------------------------------------

    tmin, tmax = -0.5, 1
    # creating epochs is instantaneous, so does not need to be cached :)
    # using `preload` prevents having to call "load_data" :)
    # the events=events needs all event markers, and with the event_id you
    # can subselecet which events to load
    epochs_orig = mne.Epochs(raw, events=events, event_id=event_id_selection,
                        tmin=tmin, tmax=tmax, preload=True,
                        baseline=None)  # don't subtract baseline

    epochs_meg = epochs_orig.copy().pick('meg') # create copy that only contains MEG data

    assert len(epochs_meg)==144, f'sanity check failed, more or less than 144 epochs {len(epochs_meg)=}'
    print(f"Bad channels: {epochs_meg.info['bads']}")

    # Plot epochs
    #epochs.plot(show=False)

    # reject bad epochs automatically
    print('###  running autoreject')

    ar = autoreject_fit_cached(epochs_meg)

    # call to ar.transform should be relatively fast, no caching needed
    # there seems to be some bug that you can only enter channel data
    # of MAG and GRAD, weird! I think this is a bug with autoreject.
    epochs_meg = ar.transform(epochs_meg)

    # create downsampled epochs
    # sampling_rate = 200
    # epochs_resampled = epochs.resample(sampling_rate, npad="auto")

    # -------------------- idependent component analysis ---------------------
    print('###  running ica')
    ica_method = 'fastica'
    n_components = 40  # todo: try with 50 maybe?
    random_state = 99
    # store definition in dictionary
    ica_def = dict(n_components=n_components, method=ica_method, random_state=random_state)
    epochs_meg, ica = fit_apply_ica_cached(epochs_orig, epochs_meg, ica_def)

    # check ICA solution
    explained_var_ratio = ica.get_explained_variance_ratio(epochs_meg)  # todo: raw or raw_ICA_for_fitting?
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f'Fraction of {channel_type} variance explained by all components: '
            f'{ratio}'
        )

    #--------------------- save epochs ---------------------------------------
    print('###  saving epochs')
    filename_epoch = f'{participant=}_{event_id_selection=}_{tmin=}_{tmax=}-epo.fif'
    filename_epoch = valid_filename(filename_epoch)
    epoch_file_path = os.path.join(epochs_folderpath, f"{filename_epoch}-epo.fif")
    epochs_meg.save(epoch_file_path, fmt='double', overwrite=True)



end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.3f} seconds")
