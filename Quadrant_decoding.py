import pandas as pd
import os
from joblib import Memory
import time
import numpy as np


# -------------------- user specifics ----------------------------------------
# folderpath, where the epochs are stored
# take the following parameters from the stored filenames!!
epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")
event_id = "trigger_gif_onset"
tmin = -0.5
tmax = 1
n_components = 40

# set cache directory for faster execution with joblib Memory package
cachedir = '/zi/flstorage/group_klips/data/data/VeraK/joblib_cache' #'/home/vera.kluetz/joblib_cache'


# -------------------- cached functions --------------------------------------
# measure code execution
start_time = time.time()

mem = Memory(cachedir)
@mem.cache
def read_epoch_cached(full_filename):
    df = pd.read_csv(full_filename)
    return df


# -------------------- read data ---------------------------------------------
# loop through each participants number from 01 to 35
for participant in [str(i).zfill(2) for i in range(1, 36)]:

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'This is participant number {participant}')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    filename_epoch = ("par" + str(participant) + "_" + str(event_id) + "_" + str(tmin) + "_" + str(tmax) + "_" + str(n_components))
    full_filename = os.path.join(epochs_folderpath, f"{filename_epoch}-epo.csv")
    try:
        df = read_epoch_cached(full_filename)
    except:
        print(f"Participant number {participant} does not exist or there is no file for this participant. \n "
              f"If you expected it to exist, please check the parameters given for the filename creation. \n "
              f"Proceeding with next participant.\n")
        continue







end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.3f} seconds")




