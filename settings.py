#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:11:07 2024

@author: simon.kern
"""
import os
import getpass
import platform
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

cachedir = None
datadir = None

if username == 'vera.kluetz' and host=='zilxap29':  # klipscalc host
    cachedir = '/zi/flstorage/group_klips/data/data/VeraK/joblib_cache'
    datadir = "/zi/flstorage/group_klips/data/data/Emo-React-Prestudy/participant_data/"
    # Use the 'TkAgg' backend for interactive plotting
    plt.switch_backend('TkAgg')

if username == 'simon.kern' and host=='zislrds0035.zi.local':  # simons VM
    cachedir = f'{home}/Desktop/joblib/'  # this folder is likely local, so much faster
    datadir = "/zi/flstorage/group_klips/data/data/Emo-React-Prestudy/participant_data/"

# ENTER YOUR DIRS USING THE FOLLOWING TEMPLATE
# if username == 'your.username (lowercase)' and host=='yourhostmachine':
#     datadir = '/hobbes/Klips/EMO_REACT-prestudy/participant_data/'
#     cachedir = f'{home}/Desktop/joblib-cache/'

#todo: include else statement
#else:
#    raise Exception(f'No profile found for {username} @ {host}. Please enter in settings.py')
