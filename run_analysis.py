import os
import time
import settings
import utils
import matplotlib.pyplot as plt
import numpy as np
import functions
import pandas as pd


os.nice(1)  # make sure we're not clogging the CPU
plt.ion()



n_components_pca = 544
bands_selection = ['delta', 'theta', 'alpha', 'beta']

bands_dict = {'delta': [1, 4],
                   'theta': [4, 8],
                   'alpha': [8, 12],
                   'beta': [13, 30]}



# measure code execution
start_time = time.time()


# loop through each participants number from 01 to 35
missing = [25, 28, 31]
participants = [str(i).zfill(2) for i in range(1, 36) if not i in missing]   #todo: set to 1

# small plots for individual participants and one bottom plot for a summary
fig, axs, ax_bottom = utils.make_fig(n_axs=len(participants), n_bottom=[0, 1], figsize=[14, 14])
df_all = pd.DataFrame()  # save results of the calculations in a dataframe

for p, participant in enumerate(participants):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'This is participant number {participant}')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    #epochs, labels = utils.get_quadrant_data(participant)
    epochs, labels = utils.get_valence_data(participant)
    if epochs is None or labels is None:
        continue

    # get sampling frequency
    sfreq=epochs.info['sfreq']
    timepoints = epochs.times
    # extract data stored in epochs
    data_x = epochs.get_data(copy=False) #shape(144, 306, 3501)

    # if there are less than 20 epochs, skip this participant
    if len(data_x) < 20:
        axs[p].text(0.1, 0.4, f'{participant=} \n {len(data_x)} epochs, skip')
        continue  # some participants have very few usable epochs

    windows = utils.extract_windows(data_x, sfreq, win_size=0.5, step_size=0.2) #todo: is step size 0.25 to big? only 50% overlap
    #shape(144, 306, 16, 500)

    bands = [bands_dict[bands_selection[i]] for i in range(len(bands_selection))]

    windows_power = functions.get_bands_power(windows, sfreq, bands)
    # shape (2,34,306,16)

    for i in range(windows_power.shape[0]):
        values= windows_power[i,:,:,:]
        values = np.mean(values, axis=2)
        values = np.mean(values, axis=0)
        fig_head, ax_head = utils.plot_sensors(values)
        filename = f"head_plot_{bands_selection[i]}_power_event_id{settings.event_id_selection}_tmin{settings.tmin}_tmax{settings.tmax}{settings.fileending}.png"
        plot_filename = os.path.join(settings.plot_folderpath, filename)
        fig_head.savefig(plot_filename)

    #uncomment this for saving a picture of alpha and or theta waves, note that is happens for every participant and overwrites the previous one
    #for i in range(windows_power.shape[0]):
    #    # just for the fun of it, plot e.g. mean alpha power over time for each channel
    #    # there should be a streak of occipital channels showing higher alpha
    #    fig_pow = plt.figure()
    #    ax_pow = fig_pow.subplots(1,1)
    #    ax_pow.imshow(np.mean(windows_power[i,:,:,:], axis=0), aspect='auto') #interpolation = none
    #    ax_pow.set_xlabel('timestep of window')
    #    ax_pow.set_ylabel('channel number')
    #    filename = f"Feature_{bands_selection[i]}_power_event_id{settings.event_id_selection}_tmin{settings.tmin}_tmax{settings.tmax}{settings.fileending}.png"
    #    plot_filename = os.path.join(settings.plot_folderpath, filename)
    #    fig_pow.savefig(plot_filename)
        #fig_pow.show()


    windows_power = functions.reshape_windows_power(windows_power)
    # shape (34, 2 * 306, 16)


    #todo: apply PCA for dimensionality reduction, see how many features explain how much of the variance to determine the right amount of components
    windows_power = functions.reshape_windows_power_for_pca(windows_power)
    pca_windows_power = functions.pca_fit_transform(windows_power, n_components_pca)
    pca_windows_power = functions.reshape_windows_power_after_pca(pca_windows_power, windows.shape[2])

    df_subj = functions.decode_features(pca_windows_power, labels, participant, settings.pipe, timepoints, n_splits=3)
    if df_subj is None:
        continue

    # append to dataframe holding all data
    df_all = pd.concat([df_all, df_subj])

    # update figure with subject
    functions.plot_subj_into_big_figure(df_all, participant, axs[p], ax_bottom, 0.2)



bands_string = result = '-'.join(bands_selection)
plot_filename = os.path.join(settings.plot_folderpath,
                             f"feature_decoding_{bands_string}_{settings.classifier_name}_event_id{settings.event_id_selection}_tmin{settings.tmin}_tmax{settings.tmax}_pca{n_components_pca}{settings.fileending}.png")

fig.savefig(plot_filename)

end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.3f} seconds")


# Run_analysis
#
#
#
# for participant in paticipants
#     data = get_your_data(participant)
#     feat = get_alpha(data)
#     ... do classification
#     accuracy = classify(data, feats)
#     plot_data(accuracy)
#
#
#
#
# quadrant_decoding.py
#
# feat = get_alpha(epochs)
#     epochs, labels = get_gif_onset_data(target='valence')
#     load_epochs(subj, event_id, tmin, tmax, )
#
#
#
# run_analysis.py
#
# def get_brain_bands()
# def get_alpha(data, sf):
#     get_fft(data, sf)
#     alpha =
#     returns alpha value per channel for this window
