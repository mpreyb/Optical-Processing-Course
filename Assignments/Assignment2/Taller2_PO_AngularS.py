"""
    File name: Taller2_PO_AngularS.py
    Author: Maria Paula Rey, EAFIT University
    Date created: 12/08/2021
    Date last modified: 30/08/2021
    Python Version: 3.8
"""

import numpy as np
from matplotlib import pyplot as plt
from math import pi

#-------------------------------------------Angular Spectrum--------------------------------------------------------
def angularSpectrum(field, z, wavelength, dx, dy):
    '''
    # Function to diffract a complex field using the angular spectrum approach
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx/dy - sampling pitches
    '''
    M, N = field.shape      # Number of horizontal and vertical data points
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)
    
    field_spec = np.fft.fftshift(field)
    field_spec = np.fft.fft2(field_spec)
    field_spec = np.fft.fftshift(field_spec)
        
    phase = np.exp2(1j * z * pi * np.sqrt(np.power(1/wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))
	
    tmp = field_spec*phase
    
    out = np.fft.ifftshift(tmp)
    out = np.fft.ifft2(out)
    out = np.fft.ifftshift(out)
	
    return out

def imageShow (inp, title):
    '''
    # Function to display an image
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image        
    '''
    plt.imshow(inp, cmap='gray'), plt.title(title)  # image in gray scale
    plt.show()                                      # Show image

    return

#--------------------------------Intensity definition-------------------------------------------
def intensity (inp, log):
    '''
    # Function to calcule the intensity representation of a given complex field
    # Inputs:
    # inp - The input complex field
    # log - boolean variable to determine if a log representation is applied
    '''
    out = np.abs(inp)
    out = out*out
    if log == True:
        out = 20 * np.log(out)
    return out

#---------------------------------------Propagating-----------------------------------------------

M = N = 256

# Defining the pinhole (circular aperture)
x0 = M/2                    
y0 = N/2                    
radius = 25.6               # Radius = 1mm/(10mm/256) [px] Diameter = 2mm.
pinhole = np.zeros((M,N))   # Array filled with zeros        

for j in range (M):
    for i in range (N):
        # Defining = 1 for everything inside the radius (this is the circular aperture)
        if np.power(j-x0, 2) + np.power(i-y0, 2) < np.power(radius, 2):
            pinhole[i,j] = 1

#All units are calculated in mm
dx = 0.039                        # Input pitch               
dy = 0.039                        # Input pitch 
wavelength = 0.0006328            # Monochromatic light (red)
k =  (2 * pi ) / wavelength       # Wave number modulus

# Propagating distance (distance between the input and the output planes)
# 158.03mm (Fresnel Number = 10) or 143.66mm (Fresnel Number = 11)
z = 158.03   
#z = 143.66
#z = 300

# Pinhole illuminated by a plane wave
input_wave = np.exp2(1j*k*z)*pinhole

#Propagation from input plane (where the pinhole is) to output plane 
complexfield = angularSpectrum(input_wave, z, wavelength, dx, dx)

#This function calculates the amplitude representation of a given complex field
out = intensity(complexfield, False)

#Display a gray-value image with the given title
imageShow (out,'Output Image with Angular Spectrum' )

#Verifying if the propagation was executed properly
Fresnel_num = 1 / (wavelength * z)
print('Fresnel number: ',Fresnel_num)
round_Fresnel = round(Fresnel_num)
if (round_Fresnel % 2) == 0:  
   print("{0} is even. There should be a dark spot in the center of the diffraction pattern".format(round_Fresnel))  
else:  
   print("{0} is odd. There should be a bright spot in the center of the diffraction pattern".format(round_Fresnel))
