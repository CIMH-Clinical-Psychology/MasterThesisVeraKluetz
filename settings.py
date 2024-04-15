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
import matplotlib.pyplot as plt
#%%##########
# SETTINGS
#############

caching = True # enable or disable caching
# ... put some global settings here if you want

#%%###################
# USER-SPECIFIC PATHS
######################
username = getpass.getuser().lower()  # your login name
host     = platform.node().lower()    # the name of this computer
system   = platform.system().lower()  # linux, windows or mac.
home = os.path.expanduser('~')

cachedir = None  # folder with fast i/o for caching files via joblib
datadir = None   # folder with the MEG data (i.e. FIF files)

if username == 'vera.kluetz' and host=='zilxap29':  # klipscalc host
    cachedir = '/zi/flstorage/group_klips/data/data/VeraK/joblib_cache'
    datadir = "/zi/flstorage/group_klips/data/data/Emo-React-Prestudy/participant_data/"
    # Use the 'TkAgg' backend for interactive plotting
    #plt.switch_backend('TkAgg')

elif username == 'simon.kern' and host=='zislrds0035.zi.local':  # simons VM
    cachedir = f'{home}/Desktop/joblib/'
    datadir = "/zi/flstorage/group_klips/data/data/Emo-React-Prestudy/participant_data/"

elif username == 'simon' and host=='kubuntu':  # simons home computer
    cachedir = f'{home}/Desktop/joblib/'
    datadir = None  # I don't have the data on my local computer

# ENTER YOUR DIRS USING THE FOLLOWING TEMPLATE
# elif username == 'your.username (lowercase)' and host=='yourhostmachine':
#     datadir = '/hobbes/Klips/EMO_REACT-prestudy/participant_data/'
#     cachedir = f'{home}/Desktop/joblib-cache/'

else:
    raise Exception(f'No profile found for {username} @ {host}. Please enter in settings.py')

for foldername in {'cachedir', 'datadir'}:
    folder = locals()[foldername]  # hacky, don't use this generally
    if folder is None:
        warnings.warn(f'"{foldername}" is not defined for current user (see settings.py)')
    elif not os.path.exists(folder):
        warnings.warn(f'"{foldername}": {folder} is defined but does not exist (yet) on current machine.')
