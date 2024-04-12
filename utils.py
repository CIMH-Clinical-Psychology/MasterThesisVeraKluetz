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
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from joblib import Memory

mem = Memory(settings.cachedir)

@mem.cache
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
    if isinstance(subj, str) and subj.isdigit():
        subj = int(subj)
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


def make_fig(n_axs, n_bottom=2, no_ticks=False, suptitle='',
             xlabel='', ylabel='', figsize=None, despine=True, subplot_kws={}):
    """
    to create a grid space with RxC rows and a large row with axis on the bottom
    that are slightly larger (e.g. for summary plots)
    will automatically try to put axis in a square layout, but return a
    flattened axs list

    Parameters
    ----------
    n_axs : int
        number of axis the plot should contain for individual plots.
    n_bottom : int, list, optional
        how many summary plots at the bottom should be contained
        can also be a list of format [bool, bool, ....] indicating positions
        that should be filled, e.g. [0, 1, 1] will create a bottom
        plot with two of three positions filled. The default is 2.
    no_ticks : bool, optional
        remove x/y ticks. The default is False.
    suptitle : str, optional
        super title of the plot. The default is ''.
    xlabel : str, optional
        xlabel. The default is ''.
    ylabel : str, optional
        ylabel. The default is ''.
    figsize : list, optional
        [w, h] of figure, same as plt.Figure. The default is None.
    despine : bool, optional
        call sns.despine(). The default is True.
    subplot_kws : dict, optional
        additional plt.subplot keywords. The default is {}.

    Returns
    -------
    fig, axs, *bottom_axis
        figure object
        all axis in a flattened array
        bottom axis of the summary plot

    """


    COL_MULT = 10 # to accomodate also too large axis
    # some heuristic for finding optimal rows and columns
    for columns in [2, 4, 6, 8]:
        rows = np.ceil(n_axs/columns).astype(int)
        if columns>=rows:
            break
    assert columns*rows>=n_axs

    if isinstance(n_bottom, int):
        n_bottom = [1 for _ in range(n_bottom)]

    COL_MULT = 1
    if len(n_bottom)>0:
        for COL_MULT in range(1, 12):
            if (columns*COL_MULT)%len(n_bottom)==0:
                break
        if not (columns*COL_MULT)%len(n_bottom)==0:
            warnings.warn(f'{columns} cols cannot be evenly divided by {len(n_bottom)} bottom plots')
    fig = plt.figure(dpi=75, constrained_layout=True, figsize=figsize)
    # assuming maximum 30 participants
    gs = fig.add_gridspec((rows+2*(len(n_bottom)>0)), columns*COL_MULT) # two more for larger summary plots
    axs = []

    # first the individual plot axis for each participant
    for x in range(rows):
        for y in range(columns):
            ax = fig.add_subplot(gs[x, y*COL_MULT:(y+1)*COL_MULT],
                                 **subplot_kws)
            if no_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            axs.append(ax)

    fig.suptitle(suptitle)

    if len(n_bottom)==0:
        return fig, axs

    # second the two graphs with all data combined/meaned
    axs_bottom = []
    step = np.ceil(columns*COL_MULT//len(n_bottom)).astype(int)
    for b, i in enumerate(range(0, columns*COL_MULT, step)):
        if n_bottom[b]==0: continue # do not draw* this plot
        ax_bottom = fig.add_subplot(gs[rows:, i:(i+step)], **subplot_kws)
        if xlabel: ax_bottom.set_xlabel(xlabel)
        if ylabel: ax_bottom.set_ylabel(ylabel)
        if i>0 and no_ticks: # remove yticks on righter plots
            ax_bottom.set_yticks([])
        axs_bottom.append(ax_bottom)
    if despine:
        sns.despine(fig)
    return fig, axs, *axs_bottom

def normalize_lims(axs, which='both'):
    """for all axes in axs: set axis limits to min/max of all other axs
    this way all axes have same xlim/ylim

    Parameters
    ----------
    axs : list
        list of axes to normalize.
    which : string, optional
        Which axis to normalize. Can be 'x', 'y', 'xy' oder 'both'.

    """
    if which=='both':
        which='xy'
    for w in which:
        ylims = [getattr(ax, f'get_{w}lim')() for ax in axs]
        ymin = min([x[0] for x in ylims])
        ymax = max([x[1] for x in ylims])
        for ax in axs:
            getattr(ax, f'set_{w}lim')([ymin, ymax])

def _window_view(a, window, step = None, axis = None, readonly = True):
    """
    Create a windowed view over `n`-dimensional input that uses an 
    `m`-dimensional window, with `m <= n`

    Parameters
    -------------
    a : Array-like
        The array to create the view on

    window : tuple or int
        If int, the size of the window in `axis`, or in all dimensions if 
        `axis == None`

        If tuple, the shape of the desired window.  `window.size` must be:
            equal to `len(axis)` if `axis != None`, else 
            equal to `len(a.shape)`, or 
            1

    step : tuple, int or None
        The offset between consecutive windows in desired dimension
        If None, offset is one in all dimensions
        If int, the offset for all windows over `axis`
        If tuple, the step along each `axis`.  
            `len(step)` must me equal to `len(axis)`

    axis : tuple, int or None
        The axes over which to apply the window
        If None, apply over all dimensions
        if tuple or int, the dimensions over which to apply the window

    generator : boolean
        Creates a generator over the windows 
        If False, it will be an array with 
            `a.nidim + 1 <= a_view.ndim <= a.ndim *2`.  
        If True, generates one window per .next() call
    
    readonly: return array as readonly

    Returns
    -------

    a_view : ndarray
        A windowed view on the input array `a`, or a generator over the windows   

    """
    ashp = np.array(a.shape)
    if axis != None:
        axs = np.array(axis, ndmin = 1)
        assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
    else:
        axs = np.arange(ashp.size)

    window = np.array(window, ndmin = 1)
    assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
    wshp = ashp.copy()
    wshp[axs] = window
    assert np.all(wshp <= ashp), "Window is bigger than input array in axes"

    stp = np.ones_like(ashp)
    if step:
        step = np.array(step, ndmin = 1)
        assert np.all(step > 0), "Only positive step allowed"
        assert (step.size == axs.size) | (step.size == 1), "step and axes don't match"
        stp[axs] = step

    astr = np.array(a.strides)

    shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
    strides = tuple(astr * stp) + tuple(astr)

    as_strided = np.lib.stride_tricks.as_strided
    a_view = np.squeeze(as_strided(a, 
                                 shape = shape, 
                                 strides = strides, writeable=not readonly))
    
    return a_view


def extract_windows(signal, sfreq, winsize, stepsize=None, axis=-1):
    """ 
    Extract windows from a signal of a given window size with striding step
    
    :param sfreq:    the sampling frequency of the signal
    :param winsize:  the size of the window when sliding through signal
                     in seconds
    :param stepsize: stepize of the window extraction in seconds. 
                     If None, defaults to 1
    :param axis:     on which axis to extract the windows
    """ 
    if stepsize is None: stepsize = 1
    stepsize *= sfreq  # convert to sample space
    winsize *= sfreq  # convert to sample space
    assert signal.shape[axis]>=winsize, 'signal is shorter than window size'
    n_step = signal.shape[axis]//stepsize
    
    if axis<0:
        axis = np.arange(signal.ndim)[axis]
    windows = _window_view(signal, window=winsize, step=stepsize,
                           axis=axis, readonly=True)
    return windows



csv_file = '/home/simon/Nextcloud/ZI/2023.05 EMO-REACT-prestudy/data/40_EMO_REACT_prestudy_2023-09-20_20h57.56.068.csv'

if __name__=='__main__':
    # by running this file as a main script you can run tests that check
    # if all data can be loaded
    for subj in range(1,36):
        if subj in (25, 28, 31):  # these are missing
            continue
        df_subj = load_exp_data(subj)
