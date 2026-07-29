#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import IPython as IP
    ipy = IP.get_ipython()
    if ipy is not None:
        ipy.run_line_magic('reset', '-sf')
except Exception:
    pass

# %% import modules and set default fonts and colors
"""
Default plot formatting code for Austin Downey's series of open source notes/
books. This common header is used to set the fonts and format.
Header file last updated May 16, 2024
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl



#%% plot


D = np.loadtxt('vibration_0.7.csv',skiprows=2,rows=1000)

plt.figure()
plt.plot(D[:,0],D[:,1])