# -*- coding: utf-8 -*-
"""
    File name: punto2_taller1.py
    Author: Maria Paula Rey
    Date created: 5/08/2021
    Date last modified: 9/08/2021
    Python Version: 3.8
"""

from raytracing import *
import matplotlib.pyplot as plt

# Optical system parameters [mm]
f1 = 1000        # Focal length of objective lens
d1 = 25          # Diameter of objective lens
f2 = 3000        # Focal length of eyepiece
d2 = 25          # Diameter of eyepiece
f3 = 22          # Typical focal length of human eye
d3 = 9           # Diameter of cristalline
d_obj = 2000     # Distance between first lens and object

# We build the optical path
path = OpticalPath()
path.label = "Keplerian Refracting Telescope"
 
path.fanAngle = 0.01
path.fanNumber = 10
path.rayNumber = 3

path.append(Space(d=d_obj))

# Objective lens
path.append(Lens(f=f1, diameter=d1, label='Objective lens'))

# Space between lenses (sum of focal lengths of the objective lens and the eyepiece)
path.append(Space(d=f1,n=1))
path.append(Space(d=f2,n=1))

# Eyepiece
path.append(Lens(f=f2, diameter=d2, label='Eyepiece'))
path.append(Space(d=f2))
path.append(Space(d=f3))

# Human cristalline
path.append(Lens(f=f3, diameter=d3, label='Eye'))
path.append(Space(d=f3))

path.display()


# Magnification calculation
M = f1/f2
print(M)