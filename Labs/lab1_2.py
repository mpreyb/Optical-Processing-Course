# -*- coding: utf-8 -*-
"""
    This code is part of the Frequency response of optical imaging systems report
    File name: average.py
    Author: Grupo B (Mateo Morales, Esteban Velásquez, Maria Paula Rey)
    Date last modified: 17/10/2021
    Python Version: 3.8
"""

# Necessary libraries
import cv2_imshow
import cv2
import scipy as sp
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Create 4 circles one in each corner centered in the corner to filter the fourier transformed image
def imageShow (inp, title):
    '''
    # Function to display an image
    # Inputs:
    # inp - The input complex field
    # title - The title of the displayed image        
    '''
    plt.figure(figsize = (30,15))
    plt.imshow(inp, cmap='gray'), plt.title(title)  # image in gray scale
    plt.show()  # show image
 
    return

# Function of a small aperture transmitance
def circ2D(size, radius, center=None): 
  '''
  Make a 2D circ function.
  Size is the length of the signal
  radius is the radius of the circ function
  '''
  if center is None:
      x0 = y0 = size // 2
  else:
      x0 = center[0]
      y0 = center[1]

  data = np.ones((size,size))
  for j in range (size):
      for i in range (size):
          if np.power(j-x0, 2) + np.power(i-y0, 2) < np.power(radius, 2):
              data[i,j] = 0

  return data

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

 
def Fouriertransf(Newimg1):
  imageShow(Newimg1,'input')
  Newimg1=np.clip(Newimg1, 0, 255)
  norm = Newimg1.max()
  Newimg1 = (Newimg1/norm)                  # Normalizing th input image
  imageShow(Newimg1,'input')
  field_spec = np.fft.fft2(Newimg1)         # Fourier transform
  field_spec = np.fft.fftshift(field_spec)
  plt.figure(figsize = (30,15))
  field_spec=amplitude(field_spec,False)    # Gettig the amplitude
  field_spec = field_spec.astype('uint8')   # Field uint
  plt.imshow(field_spec,cmap='gray')
  plt.title('Fourier Transform')
  plt.show()
  return field_spec

## Fourier transform PSFmono
img = cv2.imread("PSFmono.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ElipseFourier= Fouriertransf(imgGray)

## Fourier transform PSFpoly
img = cv2.imread("PSFpoli.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ElipseFourier2= Fouriertransf(imgGray)