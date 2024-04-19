import pandas as pd
import utils
import settings
import os
import mne
from joblib import Memory
import seaborn as sns
import matplotlib.pyplot as plt

os.nice(1)  # make sure we're not clogging the CPU

# -------------------- user specifics ----------------------------------------
# folderpath, where the epochs are stored
# take the following parameters from the stored filenames!
epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")
plot_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Plots/")
event_id_selection = 10
tmin = -2.5
tmax = 1


mem = Memory(settings.cachedir)


@mem.cache
def read_epoch_cached_fif(full_filename):
    epochs = mne.read_epochs(full_filename)
    return epochs



missing = [25, 28, 31]
participants = [str(i).zfill(2) for i in range(1, 36) if not i in missing] #36

 # small plots for individual participants and one bottom plot for a summary
#fig, axs, ax_bottom = utils.make_fig(n_axs=len(participants), n_bottom=1)

# -------------------- read data ------------------------------------------------
# loop through each participants number from 01 to 35

list_num_epochs = []

for participant in participants:  # (6, 7)]: # for testing purposes we might use only 1 participant, so 2 instead of 36


    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'This is participant number {participant}')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    filename_epoch = f'participant{participant}_event_id_selection{event_id_selection}_tmin{tmin}_tmax{tmax}'
    full_filename_fif = os.path.join(epochs_folderpath, f"{filename_epoch}-epo.fif")

    # read the epochs
    try:
        epochs = read_epoch_cached_fif(full_filename_fif)
    except:
        print(f"Epochs: There is no epochs file for participant number {participant}. \n "
              f"If you expected the file to exist, please check the parameters given for the filename creation. \n "
              f"Proceeding with next participant.\n")
        continue
    list_num_epochs.append(len(epochs))

#sns.scatterplot(x=participants, y=list_num_epochs)
myplot = sns.barplot(x=participants, y=list_num_epochs)
plt.show()

plot_filename = os.path.join(plot_folderpath, f"Epochs_per_pariticpant_{event_id_selection=}_{tmin=}_{tmax=}.png")
fig = myplot.get_figure()
fig.savefig(plot_filename)