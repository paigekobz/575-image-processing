from skimage.color import rgb2gray
from skimage.io import imread
 
from math import log10, sqrt
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
import cv2 
plt.gray()

lena = rgb2gray(imread('lena.tiff')) *255
cameraman = imread('cameraman.tif').astype(np.float64)
tire = imread('tire.tif').astype(np.float64) / 255.0






def PSNR(f, g):
    mse = np.mean((f - g) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    PSNR_out = 20 * log10(max_pixel / sqrt(mse)) 
    return PSNR_out
 



def zoom(f):
    imagein = skimage.transform.downscale_local_mean(f, [4, 4], cval=0, clip=True)

    cv2.imshow('Original', imagein) 
    
    nni = cv2.resize(imagein, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

    bilinear = cv2.resize(imagein, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    bicubic = cv2.resize(imagein, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('Nearest Neighbour', nni)
    cv2.imshow('Bilinear', bilinear)    
    cv2.imshow('Bicubic', bicubic)

zoom(lena)