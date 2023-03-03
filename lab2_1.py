# -*- coding: utf-8 -*-
"""
    This code is part of the Optical Processing – 4F System report
    File name: lab1_2.py
    Author: Grupo B (Mateo Morales, Esteban Velásquez, Maria Paula Rey)
    Date last modified: 17/10/2021
    Python Version: 3.8
"""

import cv2_imshow 
import cv2
import scipy as sp
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Showing all the images
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
def circ2D(size, radius, center=None): #Function of a small aperture transmitance
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
  Newimg1 = (Newimg1/norm)# normalizing th input image
  imageShow(Newimg1,'input')
  field_spec = np.fft.fft2(Newimg1)# Fourier transform
  field_spec = np.fft.fftshift(field_spec)
  plt.figure(figsize = (30,15))
  field_spec=amplitude(field_spec,False)# gettig the amplitude
  field_spec = field_spec.astype('uint8')#field uint
  plt.imshow(field_spec,cmap='gray')
  plt.title('Fourier Transform')
  plt.show()
  return field_spec

def Filter(imgGray,radiuspercentage):        # Iris filter
  x_center = imgGray.shape[0]/2
  y_center = imgGray.shape[1]/2
  a=circ2D(imgGray.shape[0],imgGray.shape[1], (radiuspercentage/100)*x_center, (x_center,y_center)) # 15 micrometers of radius
  imageShow (a, 'Input Signal')
  return a

def FilterHigh(imgGray,radiuspercentage):    # Spot filter
  x_center = imgGray.shape[0]/2
  y_center = imgGray.shape[1]/2
  a=circ2D(imgGray.shape[0],imgGray.shape[1], (radiuspercentage/100)*(x_center), (x_center,y_center)) # 15 micrometers of radius
  imageShow (a, 'Input Signal')
  return a

def colfilter(imgGray,percentage):           # Column filter 
  x_center = imgGray.shape[0]/2
  y_center = imgGray.shape[1]/2
  a=circ2D(imgGray.shape[0],imgGray.shape[1], percentage, (x_center,y_center))
  imageShow (a, 'Input Signal')
  return a 

def colfilter1(imgGray,percentage):          # Row filter
  x_center = imgGray.shape[0]/2
  y_center = imgGray.shape[1]/2
  a=circ2D(imgGray.shape[0],imgGray.shape[1], percentage, (x_center,y_center))
  imageShow (a, 'Input Signal') 
  return a

#--------------------------------------------------------------------------------------------------
# Fourier transformation and filtering in Fourier plane
def Filterandfourier(Newimg1,filter,percentage): 

  field_spec = np.fft.fftshift(Newimg1)
  field_spec = np.fft.fft2(field_spec)
  field_spec = np.fft.fftshift(field_spec)
  
  #Normalizing the image
  # norm = field_spec.max()
  # field_spec = (field_spec/norm)
  #Multiplying by the filter
  out=np.multiply(filter,field_spec)

  # Inverse
  out = np.fft.ifftshift(out)
  out = np.fft.ifft2(out)
  out = np.fft.ifftshift(out)

  # Getting them all in uint
  out=amplitude(out,False)
  out = out.astype('uint8')


  #Filtered in frequencies
  plt.show()
  plt.figure(figsize = (30,15))
  plt.imshow(out,cmap='gray')
  plt.title(f'Filtered Image {percentage} %')
  plt.show()

  return out
#--------------------------------------------------------------------------------------------------

#Fourier transformation of images 2f
# 
## Fourier 2f

img = cv2.imread("elipse.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ElipseFourier= Fouriertransf(imgGray)

img = cv2.imread("bandanegra.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ElipseFourier= Fouriertransf(imgGray)

img = cv2.imread("bandablanca.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ElipseFourier= Fouriertransf(imgGray)

#--------------------------------------------------------------------------------------------------

# Fourier filtering 4f

# Filtering of column
img = cv2.imread("frecmult.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGray=np.clip(imgGray, 0, 255)
plt.figure(figsize = (30,15))
plt.imshow(imgGray,cmap='gray')
plt.title('Original image')
filteredimage = Filterandfourier(imgGray,colfilter(imgGray,1.5),1.5) # 1.5, 7.5 y 15 % filtos columna
filteredimage = Filterandfourier(imgGray,colfilter(imgGray,7.5),7.5)
filteredimage = Filterandfourier(imgGray,colfilter(imgGray,15),15)

# Filtering rows
img = cv2.imread("frecmult.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize = (30,15))
plt.imshow(imgGray,cmap='gray')
plt.title('Original image')
filteredimage = Filterandfourier(imgGray,colfilter1(imgGray,1.5),1.5)
filteredimage = Filterandfourier(imgGray,colfilter1(imgGray,7.5),7.5)
filteredimage = Filterandfourier(imgGray,colfilter1(imgGray,15),15)

# Filtering Spots
img = cv2.imread("frecmult.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize = (30,15))
plt.imshow(imgGray,cmap='gray')
plt.title('Original image')
filteredimage = Filterandfourier(imgGray,FilterHigh(imgGray,1),1)
filteredimage = Filterandfourier(imgGray,FilterHigh(imgGray,5),5)
filteredimage = Filterandfourier(imgGray,FilterHigh(imgGray,10),10)

# Filtering Iris
img = cv2.imread("frecmult.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize = (30,15))
plt.imshow(imgGray,cmap='gray')
plt.title('Original image')
filteredimage = Filterandfourier(imgGray,Filter(imgGray,1.5),1.5)
filteredimage = Filterandfourier(imgGray,Filter(imgGray,5),5)
filteredimage = Filterandfourier(imgGray,Filter(imgGray,10),10)
