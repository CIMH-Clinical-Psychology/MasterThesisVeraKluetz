#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:35:53 2024

This file contains various functions for helping with the workflow,
e.g. loading of responses from participants

@author: simon kern
"""
import os
import settings
import pandas as pd

def load_exp_data(subj):
    """
    load experiment data from psychopy csv log file

    only contains data of GIF and ratings, not of flanker trial task
    There might be additional information in the CSV that is not loaded here.

    Parameters
    ----------
    subj : int
        integer participant number.

    Returns
    -------
    df : pd.DataFrame
        dataframe with the following columns:
           'mp4_filename',                  GIF filename
           'valence',                       mean valence rating of Keltner db
           'emotion',                       emotion with highest mean rating
           'gif_position',                  position on screen of GIF
           'emo_arousal.rating',            subjective arousal rating
           'emo_valence.rating',            subjective valence rating
           'emo_gifs_key.rt',               reaction time of button press (GIF)
           'feedback_valence_key.rt',       reaction time of valence selection
           'feedback_arousal_key.rt'        reaction time of arousal selection
           'button_pressed'                 if a button has been pressed at GIF
           'gif_shown_duration'             seconds the GIF was shown (roughly)

    """
    assert isinstance(subj, int)
    subj_dir = settings.datadir + f'/ERP-{subj:02d}'

    # load all files saved in participant's folder
    files = os.listdir(subj_dir)

    # select csv files
    csv_files = [f for f in files if f.endswith('csv')]

    # should be only one file, but maybe there are multiple, then
    # the wrong ones should be moved to a separate folder manually
    assert len(csv_files)>0, f'no csv found for subj ERP{subj:02d} at {subj_dir}'
    assert len(csv_files)==1, f'more than one csv found for subj ERP{subj:02d} in {subj_dir}'

    df = pd.read_csv(f'{subj_dir}/{csv_files[0]}')
    assert (dflen:=len(df))==604, f'should be 604 but csv has only {dflen} rows'

    # subselect trials without the learning trials in the beginning
    df = df[~df.mp4_filename.isna()]
    assert (df.Participant_ID==subj).all()

    # create some new columns
    # gif duration minu 330 ms inter trial interval
    df['gif_shown_duration'] = (df['feedback_valence_text.started'] -
                                df['emo_gifs.started'] - 0.33)
    df['button_pressed'] = ~df['emo_gifs_key.rt'].isna()

    # drop all columns that are not relevant. Some might still be usefule and
    # could be added later, for example the flanker task items
    keep = ['mp4_filename', 'valence', 'emotion', 'gif_position',
            'emo_arousal.rating', 'emo_valence.rating', 'emo_gifs_key.rt',
            'feedback_valence_key.rt', 'feedback_arousal_key.rt',
            'gif_shown_duration', 'button_pressed']
    df.drop(columns=[c for c in df.columns if not c in keep], inplace=True)

    # remove redundant rows
    df = df.iloc[::4].reset_index(drop=True)
    df.index.name = 'trial_nr'
    assert (n_trials:=len(df))==144, f'more or less than 144 trials in file {n_trials=}'
    return df

csv_file = '/home/simon/Nextcloud/ZI/2023.05 EMO-REACT-prestudy/data/40_EMO_REACT_prestudy_2023-09-20_20h57.56.068.csv'

if __name__=='__main__':
    # by running this file as a main script you can run tests that check
    # if all data can be loaded
    for subj in range(1,35):
        if subj in (25, 28, 31):  # these are missing
            continue
        df_subj = load_exp_data(subj)
