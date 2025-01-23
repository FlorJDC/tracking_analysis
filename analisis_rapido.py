#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:05:32 2025

@author: azelcer
"""
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

# Todos los tiempos en PS

_PERIOD = 50000
_PULSE_LIMITS = (0, 13200, 25400, 38500, np.inf)
_DATA_SHIFT = 16200
_NBINS = 500

# %% Data load
filename = Path("Dona_1_20250123-155021_.npy")
data = np.load(filename)
# data = np.stack((np.arange(10000), np.random.randint(0, _PERIOD, (10000,))), axis=-1)

# %% Data split
# FIXME: ojo 0 y 1
deltas = data[:, 0]
ttags = data[:, 1] - data[0, 0]


# %% Data shift
deltas = (deltas - _DATA_SHIFT) % _PERIOD


# %% Prepare Filters
filters = [(deltas > _PULSE_LIMITS[p]) & (deltas < _PULSE_LIMITS[p+1])
           for p in range(4)]


# %% TimeTraces
# tt_binned, tt_edges = np.histogram(ttags, 500)
plt.figure("TimeTrace")
# ttraces = plt.subplots(4)
binned_tt, bin_tt, _ = plt.hist(ttags, _NBINS, histtype='step')
for f in filters:
    plt.hist(ttags[f], bins=bin_tt, histtype='step')


# %% minflux binning
plt.figure("Pulses")
binned_pulses, bin_pulses, _ = plt.hist(deltas, _NBINS)
for pos in _PULSE_LIMITS[:-1]:
    plt.axvline(pos)
