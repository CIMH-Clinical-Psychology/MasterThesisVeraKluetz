# MasterThesisVeraKluetz


This repository has the following structure:

- for preprocessing the .fif files from the MEG, execute the 'run_preprocessing.py'
- for decoding in which corner of the screen the GIF was shown, use the 'run_quadrant_decoding.py'. For reference, there is also the file 'quadrants_simon' which is in principal the same file written by simon, but it has not been kept up to date and is only there for reference
- for decoding emotional content, use the 'run_analysis.py' file

Many helper functions have been outsorced to 'functions.py' and 'utils.py', there is no clear distinction when to choose which file of them for adding a function.  
The 'settings.py' is an important central file which contains all user settings, like which exact classifier to choose, which pre-processing version, which timepoints for the epochs, which event id,...

The behavioural results of the MEG measurement have been evaluated with the 'run_behavioural_analysis.py'. For example, it creates plots showing how often people pressed the button, correlation values,..






///


For reference, here is the trial workflow:

```
blank screen 
(500 ms)
fixation cross
trigger 99
(2000 ms)
pre-image
trigger 10
(250 ms)
gif onset
trigger 20
(X seconds)
gif offset or button press
trigger 30
blank screen
trigger 103
(300 ms)
valence selection
trigger 101
(X seconds)
blank screen
trigger 103
(300 ms)
arousal selection
trigger 102
(X seconds)
blank screen
trigger 103
(300 ms)
flanker
trigger 104
(3 seconds max)
flanker
trigger 104
(3 seconds max)
flanker
trigger 104
(3 seconds max)
flanker
trigger 104
(3 seconds max)

repeat
```
