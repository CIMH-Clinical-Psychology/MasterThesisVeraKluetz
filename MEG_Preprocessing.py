### code taken and altered from https://github.com/natmegsweden/meeg_course_mne/blob/master/tutorial_01a_preprocessing.md
### as well as from Alper Koelgesiz, on GitHub MEGInternshipAlper/EEG_Many_Alper/Preprocessing.py
### 22.02.2024
### vera.kluetz@zi-mannheim.de


import mne
import time
from joblib import Memory
import warnings
import matplotlib.pyplot as plt
import autoreject
import os
import numpy as np


# -------------------- user specifics ----------------------------------------

# set cache directory for faster execution with joblib Memory package
cachedir = '/zi/flstorage/group_klips/data/data/VeraK/joblib_cache' #'/home/vera.kluetz/joblib_cache'

# set file path, where the data can be found
folderpath = (f"/zi/flstorage/group_klips/data/data/Emo-React-Prestudy/participant_data/")

#set folderpath, where the resulting epochs should be stored
epochs_folderpath = (f"/home/vera.kluetz/epochs/")
#(f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")


# -------------------- initial setup and function definition -----------------

# Use the 'TkAgg' backend for interactive plotting
plt.switch_backend('TkAgg')

# measure code execution
start_time = time.time()

# surpress warning that naming convention is not met
warnings.filterwarnings("ignore",
                        message=".*does not conform to MNE naming conventions. All raw files should end with raw.fif, raw_sss.fif, raw_tsss.fif*",
                        category=RuntimeWarning, module="mne")

# define cached functions for faster execution
mem = Memory(cachedir)
# cached_func = mne.cache(mne.io.read_raw)
@mem.cache
def read_raw_cached(fif_filepath):
    raw = mne.io.read_raw_fif(fif_filepath, preload=True, verbose=False)
    return raw

@mem.cache
def create_epochs_cached(raw, eve, event_id, tmin, tmax):
    epochs = mne.Epochs(raw, events=eve, event_id=event_id, tmin=tmin, tmax=tmax, picks='meg')
    return epochs

@mem.cache
def filter_cached(data_in, lower_bound, higher_bound):
    data_out = data_in.filter(lower_bound, higher_bound)
    return data_out

@mem.cache
def ica_fit_cached(ica_in, data):
    ica_out = ica_in.fit(data, picks='meg')
    return ica_out

@mem.cache
def autoreject_fit_cached(ar_in, epochs):
    ar_out = ar_in.fit(epochs)
    return ar_out

@mem.cache
def ica_apply_cached(ica_in, epochs):
    ica_out = ica_in.apply(epochs)
    return ica_out


# -------------------- load data ---------------------------------------------

# creates a list of all participant numbers from 01 to 35 so that we can loop through them
par_numbers = [str(i).zfill(2) for i in
               range(1, 36)]  # for testing purposes we might use only 1 participant, so 2 instead of 36

# loop through each participant's data
for participant in par_numbers:

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'This is participant number {participant}')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    fif_filepath = folderpath + f"ERP-{participant}/ERP{participant}_initial_tsss_mc.fif"

    # read raw data
    try:
        raw = read_raw_cached(fif_filepath)
    except:
        print(
            f"Participant number {participant} does not exist or there has been problems with reading it's file. Proceeding with next participant.")
        continue
    raw_copy = raw.copy()

    # check if we have more than 25 Minutes of Recordings included
    assert len(raw) > 60 * 25 * raw.info['sfreq']  # 50 sec * 25 min * sampling frequency

    # Set EOG channel
    eogs = {'BIO002': 'eog', 'BIO003': 'eog'}
    raw.set_channel_types({**eogs})


    # -------------------- filtering -----------------------------------------

    # low- and highpass data
    # raw.filter(0.1, 50)  # , method='fir')
    # notch filter
    # raw.notch_filter(np.arange(50, 251, 50))
    # todo: is this clever?
    # raw.filter(0.1, 49)
    raw = filter_cached(raw, 0.1, 49)


    # -------------------- find events ---------------------------------------

    eve = mne.find_events(raw, stim_channel='STI101', min_duration=3 / raw.info['sfreq'])

    # set event IDs
    event_id = {'trigger_preimage': 10,
                'trigger_gif_onset': 20,
                'trigger_gif_offset': 30,
                'trigger_fixation': 99,
                'trigger_valence_start': 101,
                'trigger_arousal_start': 102,
                'trigger_flanker_start': 104}

    # only select events with a certain trigger
    eve_id = {'trigger_gif_onset': 20}
    events = eve[eve[:, 2] == 20]


    # -------------------- create epochs -------------------------------------

    tmin, tmax = -0.5, 1
    epochs = create_epochs_cached(raw, events, eve_id, tmin, tmax)
    epochs.load_data()

    print(f"Bad channels: {epochs.info['bads']}")

    # Plot epochs
    #epochs.plot(show=False)

    # reject bad epochs automatically
    ar = autoreject.AutoReject(n_jobs=-1, verbose=False)
    #todo: does the following substitution by cached function work properly?
    #ar.fit(epochs)
    ar = autoreject_fit_cached(ar, epochs)

    # create downsampled epochs
    # sampling_rate = 200
    # epochs_resampled = epochs.resample(sampling_rate, npad="auto")


    # -------------------- idependent component analysis ---------------------

    raw_ICA_for_fitting = filter_cached(raw.copy(), 1, None)

    ica_method = 'fastica'
    n_components = 40  # todo: try with 50 maybe?
    random_state = 99
    ica_def = mne.preprocessing.ICA(n_components=n_components, method=ica_method, random_state=random_state)
    ica = ica_fit_cached(ica_def, raw_ICA_for_fitting)

    # check ICA solution
    explained_var_ratio = ica.get_explained_variance_ratio(raw)  # todo: raw or raw_ICA_for_fitting?
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f'Fraction of {channel_type} variance explained by all components: '
            f'{ratio}'
        )


    # -------------------- reject EOG and ECG components ---------------------

    # find bad ecg and eog channels
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='BIO001')

    # Apply ICA and exclude bad EOG components
    # todo: eog is two arrays in one array, and only one indice, so does it really exclude the correct component?
    ica.exclude = eog_indices
    ica.exclude.extend(eog_indices)
    ica.exclude.extend(ecg_indices)
    ica = ica_apply_cached(ica, epochs)


    #--------------------- save epochs ---------------------------------------
    #todo: what happens if the file already exists?
    #final_epochs = epochs.copy()
    #filename_epoch = ("par" + str(participant) + "_" + str(next(iter(eve_id))) +"_"+ str(tmin) +"_"+ str(tmax) +"_"+ str(n_components))
    #epoch_file_path = os.path.join(epochs_folderpath, f'{filename_epoch}-epo.fif')
    #final_epochs.save(epoch_file_path, fmt='double', overwrite=True)


end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.3f} seconds")
