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

    epochs, labels = utils.get_quadrant_data(participant)
    if (epochs or labels) == None:
        continue

    # get sampling frequency
    sfreq=epochs.info['sfreq']
    # extract data stored in epochs
    data_x = epochs.get_data() #shape(144, 306, 3501)

    # if there are less than 20 epochs, skip this participant
    if len(data_x) < 20:
        axs[p].text(0.1, 0.4, f'{participant=} \n {len(data_x)} epochs, skip')
        continue  # some participants have very few usable epochs

    windows = utils.extract_windows(data_x, sfreq, win_size=0.5, step_size=0.25) #todo: is step size to big? only 50% overlap
    #shape(144, 306, 13, 500)


    #todo: 1. extract beta, gamma, delta, theta values
    # 2. add them all on the channel dimension, so that the window with x features has shape (144, x*306, 13, 500)
    # 3. apply PCA for dimensionality reduction, see how many features explain how much of the variance to determine the right amount of components
    # 4. normal PCA works with 2D arrays, so take 306*(144*13)
    #np.hstack([144, 306, 13].T) -> 306x(144x13).T
    #PCA(200).fit?transform([2200x1555])

    windows_power = functions.get_windows_power(windows, sfreq)

    # just for the fun of it, plot mean alpha power over time for each channel
    # there should be a streak of occipital channels showing higher alpha
    plt.imshow(np.mean(windows_power, axis=1).T, aspect='auto') #interpolation = none
    plt.xlabel('timestep of window')
    plt.ylabel('channel number')
    filename = f"Feature_alpha_power_event_id_selection{settings.event_id_selection}_tmin{settings.tmin}_tmax{settings.tmax}{settings.fileending}.png"
    plot_filename = os.path.join(settings.plot_folderpath, filename)
    plt.savefig(plot_filename)


    df_subj = functions.decode_features(windows_power, labels, participant)
    if df_subj is None:
        continue

    # append to dataframe holding all data
    df_all = pd.concat([df_all, df_subj])

    # update figure with subject
    functions.plot_subj_into_big_figure(fig, axs, ax_bottom, p, participant, epochs, df_subj, df_all)


# todo: take alpha out of hard coded description
plot_filename = os.path.join(settings.plot_folderpath,
                             f"feature_decoding_alpha{settings.classifier}_event_id_selection{settings.event_id_selection}_tmin{settings.tmin}_tmax{settings.tmax}{settings.fileending}.png")
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
