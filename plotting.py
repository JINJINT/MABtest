import numpy as np
from numpy import sqrt, log, exp, mean, cumsum, sum, zeros, ones, argsort, argmin, argmax, array, maximum, concatenate
from numpy.random import randn, rand

import os
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from toimport import *
from MABprocess import *

np.set_printoptions(precision = 10)

## Plotting settings
matplotlib.use('Agg')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['xtick.labelsize']= 10
mpl.rcParams['ytick.labelsize']= 10
plt.switch_backend('agg')
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)

color_list = ['green', 'darkorange', 'royalblue','mediumslateblue','firebrick']






