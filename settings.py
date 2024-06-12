#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:11:07 2024

@author: simon.kern
"""
import os
import getpass
import platform
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#%%##########
# SETTINGS
#############

caching = True # enable or disable caching
# ... put some global settings here if you want


# folderpath, where the epochs are stored
epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")
plot_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Plots/")

# Regarding the following parameters:
#  - if you are before preprocessing, choose the parameters as you like
#  - if you want to use already prepocessed data, take the following parameters from the stored filenames you want to use!

event_id_selection = 30
#event_id = {'trigger_preimage': 10,
#            'trigger_gif_onset': 20,
#            'trigger_gif_offset': 30,
#            'trigger_fixation': 99,
#            'trigger_valence_start': 101,
#            'trigger_arousal_start': 102,
#            'trigger_flanker_start': 104}
tmin = -4
tmax = 0
# for the fileending, choose between the following:
# ""  "_noIcaEogRejection"   "_minimalPreprocessing"   "_EOG-only"
fileending = "_minimalPreprocessing"

# --- select classifier ----
clf = LogisticRegression(C=10, max_iter=1000, random_state=99) # C parameter is important to set regularization, might overregularize else
#clf = RandomForestClassifier(n_estimators=100, random_state=99) # could also use RandomForest, as it's more robust, should always work out of the box
classifier_name = clf.__class__.__name__

pipe = Pipeline(steps=[('scaler', StandardScaler()),
                           ('classifier', clf)])


# loop through each participants number from 01 to 35
_missing = [25, 28, 31]
participants = [str(i).zfill(2) for i in range(1, 36) if not i in _missing]   #todo: set to 1


#%%###################
# USER-SPECIFIC PATHS
######################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')

cachedir = None  # folder with fast i/o for caching files via joblib
datadir = None   # folder with the MEG data (i.e. FIF files)
#epochs_folderpath = None  # folder where the resulting epochs should be stored

if username == 'vera.kluetz' and host=='zilxap29':  # klipscalc host
    cachedir = '/zi/flstorage/group_klips/data/data/VeraK/joblib_cache'
    datadir = "/zi/flstorage/group_klips/data/data/Emo-React-Prestudy/participant_data/"
    epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")
    # Use the 'TkAgg' backend for interactive plotting
    #plt.switch_backend('TkAgg')

elif username == 'vera.kluetz' and host=='zislrds0035.zi.local': # simons VM
    cachedir = '/zi/flstorage/group_klips/data/data/VeraK/joblib_cache'
    datadir = "/zi/flstorage/group_klips/data/data/Emo-React-Prestudy/participant_data/"
    epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")

elif username == 'simon.kern' and host == 'zislrds0035.zi.local':  # simons VM
    cachedir = f'{home}/Desktop/joblib/'
    datadir = "/zi/flstorage/group_klips/data/data/Emo-React-Prestudy/participant_data/"
    epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")

elif username == 'simon' and host == 'kubuntu':  # simons home computer
    cachedir = f'{home}/Desktop/joblib/'
    datadir = None  # I don't have the data on my local computer
    epochs_folderpath = (f"/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/")

elif username == 'simon.kern' and host == '5cd320lfh8':  # simons work laptop
    cachedir = f'{home}/Desktop/joblib/'
    datadir = 'W:/group_klips/data/data/Emo-React-Prestudy/participant_data' 
    epochs_folderpath = "W:/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/"

# ENTER YOUR DIRS USING THE FOLLOWING TEMPLATE
# elif username == 'your.username (lowercase)' and host=='yourhostmachine':
#     datadir = '/hobbes/Klips/EMO_REACT-prestudy/participant_data/'
#     cachedir = f'{home}/Desktop/joblib-cache/'
#     epochs_folderpath = f'/zi/flstorage/group_klips/data/data/VeraK/Prestudy_preprocessed_epochs/'

else:
    raise Exception(f'No profile found for {username} @ {host}. Please enter in settings.py')

for foldername in {'cachedir', 'datadir'}:
    folder = locals()[foldername]  # hacky, don't use this generally
    if folder is None:
        warnings.warn(f'"{foldername}" is not defined for current user (see settings.py)')
    elif not os.path.exists(folder):
        warnings.warn(f'"{foldername}": {folder} is defined but does not exist (yet) on current machine.')
