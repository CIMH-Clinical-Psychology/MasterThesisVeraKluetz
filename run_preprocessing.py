#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:29 2024

This file contains all code to run different preprocessings

@author: vera klütz
"""
import os
import functions
import time
import numpy as np


os.nice(1)

# measure code execution
start_time = time.time()

# set tmin and tmax for epoch lengths
tmin, tmax = -4, 0

# set event IDs
event_id = {'trigger_preimage': 10,
            'trigger_gif_onset': 20,
            'trigger_gif_offset': 30,
            'trigger_fixation': 99,
            'trigger_valence_start': 101,
            'trigger_arousal_start': 102,
            'trigger_flanker_start': 104}

# select events with a certain trigger
event_id_selection = event_id['trigger_gif_offset']

# ignore unnecessary warnings
functions.ignore_warnings()

# take all meg data and perform autoreject and ica
functions.loop_through_participants(tmin,
                                    tmax,
                                    event_id_selection,
                                    highpass=0.1,
                                    lowpass=50,
                                    notch=np.arange(50, 251, 50),
                                    picks='meg',
                                    fileending="",
                                    autoreject=True,
                                    ica_ecg=False, # this is set to False because we decided to not filter out ecg related components in the brain signal
     # because they might tell us something about arousal (hypothesis)
                                    ica_eog=True)

## take meg data and do not filter out eog components via ica
# functions.loop_through_participants(tmin,
#                                    tmax,
#                                    event_id_selection,
#                                    highpass=0.1,
#                                    lowpass=50,
#                                    notch=np.arange(50, 251, 50),
#                                    picks='meg',
#                                    fileending = 'noIcaEogRejection',
#                                    autoreject = True,
#                                    ica_ecg = True,
#                                    ica_eog = False)
#
# take eog data only
# functions.loop_through_participants(tmin,
#                                    tmax,
#                                    event_id_selection,
#                                    highpass=0.1,
#                                    lowpass=50,
#                                    notch=np.arange(50, 251, 50),
#                                    picks='eog',
#                                    fileending = 'EOG-only',
#                                    autoreject = False,
#                                    ica_ecg = False,
#                                    ica_eog = False)
#
# take meg data and process as minimally as possible
#functions.loop_through_participants(tmin,
#                                    tmax,
#                                    event_id_selection,
#                                    highpass=0.1,
#                                    lowpass=None,
#                                    notch=None,
#                                    picks='meg',
#                                    fileending = 'minimalPreprocessing',
#                                    autoreject = False,
#                                    ica_ecg = False,
#                                    ica_eog = False)

end_time = time.time()
print(f"Elapsed time: {(end_time - start_time):.3f} seconds")
