### code taken and altered from https://github.com/natmegsweden/meeg_course_mne/blob/master/tutorial_01a_preprocessing.md
### 22.02.2024
### vera.kluetz@zi-mannheim.de


import mne
import time
from joblib import Memory
import warnings
import matplotlib.pyplot as plt
import numpy as np


# --------- initial setup and function definition ---------
# Use the 'TkAgg' backend for interactive plotting
plt.switch_backend('TkAgg')

# measure code execution
start_time = time.time()
# set cache directory for faster execution with joblib Memory package
cachedir = '/home/vera.kluetz/joblib_cache'
mem = Memory(cachedir)

# surpress warning that naming convention is not met
warnings.filterwarnings("ignore",
                        message=".*does not conform to MNE naming conventions. All raw files should end with raw.fif, raw_sss.fif, raw_tsss.fif*",
                        category=RuntimeWarning, module="mne")


# define cache functions for faster execution
# cached_func = mne.cache(mne.io.read_raw)
@mem.cache
def read_raw_cached(fif_filepath):
    raw = mne.io.read_raw_fif(fif_filepath, preload=True, verbose=False)
    return raw

@mem.cache
def create_epochs_cached(raw, eve, event_id, tmin, tmax):
    epochs = mne.Epochs(raw, events=eve, event_id=event_id, tmin=tmin, tmax=tmax)
    return epochs


# --------- load data ---------
# set file path
# we want to use a file with this format:  /zi/flstorage/group_klips/data/data/Emo-React-Prestudy/participant_data/ERP-01/ERP01_initial_tsss_mc.fif
folderpath = (f"/zi/flstorage/group_klips/data/data/Emo-React-Prestudy/participant_data/")

# creates a list of all participant numbers from 01 to 35 so that we can loop through them
par_numbers = [str(i).zfill(2) for i in
               range(1, 2)]  # for testing purposes we might use only  participants, so 2 instead of 36

# loop through each participant's data
for participant in par_numbers:
    fif_filepath = folderpath + f"ERP-{participant}/ERP{participant}_initial_tsss_mc.fif"

    # show metadata
    # info = mne.io.read_info(fif_filepath, 'WARNING')
    # print(info.keys())
    # print(info['ch_names'])
    # print(info)

    # read raw data
    try:
        raw = read_raw_cached(fif_filepath)
    except:
        print(
            f"Participant number {participant} does not exist or there has been problems with reading it's file. Proceeding with next participant.")
        continue
    raw_copy = raw.copy()
    # print(raw.info)

    # check if we have more than 25 Minutes of Recordings included
    assert len(raw) > 60 * 25 * raw.info['sfreq']  # 50 sec * 25 min * sampling frequency

    # explore which channel types are there
    #channel_types = raw.get_channel_types()
    #print("Available Channel Types:", channel_types)


    # --------- filtering ---------
    # low- and highpass data
    raw.filter(0.1, 50)  # , method='fir')
    # notch filter
    raw.notch_filter(np.arange(50, 251, 50))


    # --------- idependent component analysis ---------
    raw_ICA_for_fitting = raw.copy().filter(1, None)

    ica_method = 'fastica'
    n_components = 40
    random_state = 99
    ica = mne.preprocessing.ICA(n_components=n_components, method=ica_method, random_state=random_state)
    # todo: does 'meg' really has an effect, because as channel types there only are meg and grad and other types but not meg
    ica.fit(raw_ICA_for_fitting, picks='meg')

    # check ICA solution
    explained_var_ratio = ica.get_explained_variance_ratio(raw)
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f'Fraction of {channel_type} variance explained by all components: '
            f'{ratio}'
        )

    # plot ICA
    # todo: look at plots: most of them lie outside the scalp: how many do we want to reject? Does it really show the heratbeat and eyeblinks?
    ica.plot_sources(raw)  # right click the component name to view its properties
    ica.plot_components(inst=raw)  # click the components to view its properties
    # or
    ica.plot_properties(raw, picks=[0, 1])

    # todo: related to the one above, check if these two are the ones that we want to exclude
    # Exclude components
    ica.exclude = [0, 1]


    # todo: perform some sanity checks for EOG ECG rejection, like plotting, since they have not been tested yet
    # todo: code taken from Alper, compare with the one from MEG Github guy
    # --------- reject EOG and ECG components ---------
    #eog_indices, eog_scores = ica.find_bads_eog(raw)
    #ecg_indices, ecg_scores = ica.find_bads_ecg(raw)

    # Apply ICA and exclude bad EOG components
    # todo: add ecg_indices to excluded
    #ica.exclude = eog_indices
    #ica.apply(raw)





    # --------- find events ---------
    eve = mne.find_events(raw, stim_channel='STI101', min_duration=3 / raw.info['sfreq'])

    # See unique events
    # print(np.unique(eve[:, -1]))
    # np.unique(eve[:, -1], return_counts=True)

    # print amount of each event
    # trigger_ids, counts = np.unique(eve[:, -1], return_counts=True)
    # for trigger_ids, counts in zip(trigger_ids,counts):
    #    print(f"Trigger ID {trigger_ids}: Count {counts}")

    # set event IDs
    event_id = {'trigger_preimage': 10,
                'trigger_gif_onset': 20,
                'trigger_gif_offset': 30,
                'trigger_fixation': 99,
                'trigger_valence_start': 101,
                'trigger_arousal_start': 102,
                'trigger_flanker_start': 104}
    # trigger_reset = 0
    # trigger_blank_post_feedback = 103
    # 8 =
    # 16 =
    # 64 =
    # 96 =

    # only select events with trigger_gif_onset, ID 20
    id_gif_onset = {'trigger_gif_onset': 20}
    eve_gif_onset = eve[eve[:, 2] == 20]

    # inspect events and triggers in raw data
    # todo: why are events anywhere but not related at all with triggers???
    raw.plot(events=eve_gif_onset, event_id=id_gif_onset, show=False)

    # plot events
    #fig = mne.viz.plot_events(eve, event_id=event_id)


    # --------- create epochs ---------
    tmin, tmax = -0.5, 2
    # epochs = create_epochs_cached(raw_filt, eve, event_id, tmin, tmax)
    # only gif_onset trigger id

    epochs_gif_onset = create_epochs_cached(raw, eve_gif_onset, id_gif_onset, tmin, tmax)
    #epochs_gif_onset = mne.Epochs(raw, events=eve_gif_onset, event_id=id_gif_onset, tmin=tmin, tmax=tmax)

    epochs = epochs_gif_onset
    epochs.load_data()

    # Print the channel types
    # channel_types = epochs.info['ch_names']
    # print("Channel Types:", channel_types)
    # butterflyplot todo: what does the plot tell me?
    # fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    # epochs.average().plot(spatial_colors=True, axes=ax)
    # plt.tight_layout()

    # todo: what does this plot tell me?
    # fig = epochs.plot_image(picks='mag')[0]

    print(f"Bad channels: {epochs.info['bads']}")

    # Plot epochs
    epochs.plot(show=False)

    # create downsampled epochs
    #sampling_rate = 200
    #epochs_resampled = epochs.resample(sampling_rate, npad="auto")





    # save epoch files

end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.3f} seconds")
