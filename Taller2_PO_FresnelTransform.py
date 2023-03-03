"""
    File name: Taller2_PO_FresnelTransform.py
    Author: Maria Paula Rey, EAFIT University
    Date created: 12/08/2021
    Date last modified: 30/08/2021
    Python Version: 3.8
"""
 
import numpy as np
from matplotlib import pyplot as plt
from math import pi

#-------------------------------------------Fresnel Transform--------------------------------------------------------
def fresnel(field, z, wavelength, dx, dy):
    """
    # Function to diffract a complex field using Fresnel approximation with
    # Fourier method
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx/dy - sampling pitches
    """
    
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dxout = (wavelength * z) / (M * dx)
    dyout = (wavelength * z) / (N * dy)
	
    k =  (2 * pi ) / wavelength
	
    z_phase = np.exp2((1j * k * z )/ (1j * wavelength * z))
    out_phase = np.exp2((1j * pi / (wavelength * z)) * (np.power(X * dxout, 2) + np.power(Y * dyout, 2)) )
    in_phase = np.exp2((1j * pi / (wavelength * z)) * (np.power(X * dx, 2) + np.power(Y * dy, 2)))

    tmp = (field * in_phase)
    tmp = np.fft.fftshift(tmp)
    tmp = np.fft.fft2(tmp)
    tmp = np.fft.fftshift(tmp)

    out = z_phase * out_phase * dx * dy * tmp
	
    return out 


def imageShow (inp, title):
    '''
    # Function to display an image
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image        
    '''
    plt.imshow(inp, cmap='gray'), plt.title(title)  # Image in gray scale
    plt.show()                                      # Show image

    return

#----------------------------------Amplitude definition-----------------------------------------

def amplitude (inp, log):
    '''
    # Function to calcule the amplitude representation of a given complex field
    # Inputs:
    # inp - The input complex field
    # log - boolean variable to determine if a log representation is applied    
    '''
    out = np.abs(inp)
    if log == True:
        out = 20 * np.log(out)
    return out

#-------------------------------------Propagating------------------------------------------------

M = N = 256

# Defining the pinhole (circular aperture)
x0 = M/2                    #M/2
y0 = N/2                    #N/2
radius = 51.4               #Radius = 2mm/(10mm/256) [px].  Diameter = 4mm
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
# 1264.22mm (Fresnel Number = 5) or 1053.52mm (Fresnel Number = 6)
z = 1264.22   
#z = 1053.52

# Pinhole illuminated by a plane wave
input_wave = np.exp2(1j*k*z)*pinhole

#Propagation from input plane (where the pinhole is) to output plane 
temp = fresnel(input_wave, wavelength, z, dx, dx)
out = amplitude(temp, False)

#Display an gray value image with the given title
imageShow (out , 'Output Image with Fresnel Transform')

#Verifying if the propagation was executed properly
Fresnel_num = 2**2 / (wavelength * z)
print('Fresnel number: ',Fresnel_num)
round_Fresnel = round(Fresnel_num)
if (round_Fresnel % 2) == 0:  
   print("{0} is even. There should be a dark spot in the center of the diffraction pattern".format(round_Fresnel))  
else:  
   print("{0} is odd. There should be a bright spot in the center of the diffraction pattern".format(round_Fresnel))  
