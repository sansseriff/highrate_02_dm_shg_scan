import orjson
import matplotlib.pyplot as plt
from phd import viz
import numpy as np
from IPython.display import JSON
from enum import Enum
from structure import TomoCounts, Bin, Phase, create_matrix_row, print_matrix
from structure import shiftedColorMap
import matplotlib
from qutip import *
colors, swatches = viz.phd_style(jupyterStyle=True, grid=True)
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap

from custom_model_dm import PowerRamp



with open("../data/regular_scan/s_min_power_ramp.json", 'rb') as f:
    data_min = orjson.loads(f.read())

with open("../data/regular_scan/s_d90_power_ramp.json", 'rb') as f:
    data_90 = orjson.loads(f.read())

with open("../data/regular_scan/s_max_derived_power_ramp.json", 'rb') as f:
    data_max_derived = orjson.loads(f.read())



ramp_min = PowerRamp(**data_min)
ramp_max = PowerRamp(**data_max_derived)
ramp_90 = PowerRamp(**data_90)



step_scan_90 = ramp_90.results[0]
step_scan_90.results[0].derived_90_voltage
