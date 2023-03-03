# -*- coding: utf-8 -*-
"""
    File name: punto1_taller1.py
    Author: Maria Paula Rey
    Date created: 5/08/2021
    Date last modified: 6/08/2021
    Python Version: 3.8
"""

import numpy as np

from py_pol import degrees
from py_pol.jones_vector import Jones_vector
from py_pol.jones_matrix import Jones_matrix
import random

# Defining the incident unpolarized light wave
alpha_wave = np.random.rand(1) * 90*degrees
delay_wave = np.random.rand(1) * 180*degrees
E0 = Jones_vector("Incident light")
E0.general_charac_angles(alpha=alpha_wave, delay=delay_wave)
print(E0)

# ------------------------------------First polarizer----------------------------------------------------
P0 = Jones_matrix('Polarizer 1')

# Two different rotation directions (left-handed or right-handed) for circular polarization
number_list = [45, -45]
angle = random.choice(number_list)
#print(angle)

P0.diattenuator_perfect(azimuth= angle * degrees)
print(P0)

# Checking linear polarization
S_linear = P0 * E0
S_linear.name = 'Output wave of first polarizer'
#print(S_linear)

dlp = S_linear.parameters.degree_linear_polarization(verbose=True)


#------------------------------------Quarter wave plate---------------------------------------------------
P1 = Jones_matrix('Polarizer 2')
P1.quarter_waveplate(azimuth = 0 * degrees);
print(P1)

E_final = P1 * P0 * E0
E_final.name = 'Final output wave'
print(E_final)

#Calculate degree of circular polarization
dcp = E_final.parameters.degree_circular_polarization(verbose=True)

#--------------------------------------------------------------------------------------------------------