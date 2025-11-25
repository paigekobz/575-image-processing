import scipy.signal as signal
import matplotlib.pyplot as plt
import skimage.util
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
import matplotlib
from scipy.signal import convolve2d


plt.gray()
lena= rgb2gray(imread('lena.tiff'))
cameraman = imread('cameraman.tif').astype(np.float64)/255


def gaussian_filter(n_rows, n_cols, stdv):
    """
    Returns a 2d Gaussian image filter.
    """
    g_r = signal.windows.gaussian(n_rows, stdv)
    g_c = signal.windows.gaussian(n_cols, stdv)

    G = np.outer(g_r, g_c)

    return G/np.sum(G)

def PSNR(f,g):
    return 10*np.log10(1.0/ np.mean(np.square(f-g)))


h1 = (1/6)*np.ones((1,6))
h2 = h1.T
h3 = np.array([[-1, 1]])

matplotlib.pyplot.imshow(lena, cmap='gray', vmin=0, vmax=1,)
plt.show()


matplotlib.pyplot.imshow(convolve2d(lena, h1, mode='full', boundary='fill', fillvalue=0), cmap='gray', vmin=0, vmax=1,)
plt.show()

matplotlib.pyplot.imshow(convolve2d(lena, h2, mode='full', boundary='fill', fillvalue=0), cmap='gray', vmin=0, vmax=1,)
plt.show()

matplotlib.pyplot.imshow(convolve2d(lena, h3, mode='full', boundary='fill', fillvalue=0), cmap='gray', vmin=0, vmax=1,)
plt.show()

def imnoisespeckle(im, v):
    # im: input image
    # v: variance
    n = np.sqrt(v*12) * (np.random.rand(im.shape[0], im.shape[1]) - 0.5)
    return im + im * n


f = np.hstack([0.3*np.ones((200,100)), 0.7*np.ones((200,100))])

matplotlib.pyplot.imshow(f, cmap='gray', vmin=0, vmax=1,)
plt.show()
plt.hist(f)
plt.show()

matplotlib.pyplot.imshow(imnoisespeckle(f, 0.1), cmap='gray', vmin=0, vmax=1,)
plt.show()
plt.hist(imnoisespeckle(f, 0.1))
plt.show()


matplotlib.pyplot.imshow(skimage.util.random_noise(f, mode='gaussian', var=0.01), cmap='gray', vmin=0, vmax=1,)   
plt.show()
plt.hist(skimage.util.random_noise(f, mode='gaussian', var=0.01))
plt.show()

matplotlib.pyplot.imshow(skimage.util.random_noise(f, mode='s&p', amount=0.05), cmap='gray', vmin=0, vmax=1,)   
plt.show()
plt.hist(skimage.util.random_noise(f, mode='s&p', amount=0.05))
plt.show()


matplotlib.pyplot.imshow(skimage.util.random_noise(lena, mode='gaussian', var=0.002), cmap='gray', vmin=0, vmax=1,)   
plt.show()
plt.hist(skimage.util.random_noise(lena, mode='gaussian', var=0.002))
plt.show()

print(PSNR(lena, skimage.util.random_noise(lena, mode='gaussian', var=0.002)))

matplotlib.pyplot.imshow(np.ones((3,3))/(3.0*3.0), cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.show()
print((np.ones((3,3))/(3.0*3.0))) 

matplotlib.pyplot.imshow(ndimage.convolve(skimage.util.random_noise(lena, mode='gaussian', var=0.002), np.ones((3,3))/(3.0*3.0)), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.hist(ndimage.convolve(skimage.util.random_noise(lena, mode='gaussian', var=0.002), np.ones((3,3))/(3.0*3.0)))
plt.show()

print(PSNR(skimage.util.random_noise(lena, mode='gaussian', var=0.002), ndimage.convolve(skimage.util.random_noise(lena, mode='gaussian', var=0.002), np.ones((3,3))/(3.0*3.0))))

print(PSNR(lena, ndimage.convolve(skimage.util.random_noise(lena, mode='gaussian', var=0.002), np.ones((3,3))/(3.0*3.0))))


matplotlib.pyplot.imshow(np.ones((7,7))/(7.0*7.0), cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.show()
print((np.ones((7,7))/(7.0*7.0))) 

##q12
matplotlib.pyplot.imshow(ndimage.convolve(skimage.util.random_noise(lena, mode='gaussian', var=0.002), np.ones((7,7))/(7.0*7.0)), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.hist(ndimage.convolve(skimage.util.random_noise(lena, mode='gaussian', var=0.002), np.ones((7,7))/(7.0*7.0)))
plt.show()

print(PSNR(skimage.util.random_noise(lena, mode='gaussian', var=0.002), ndimage.convolve(skimage.util.random_noise(lena, mode='gaussian', var=0.002), np.ones((7,7))/(7.0*7.0))))

matplotlib.pyplot.imshow(gaussian_filter(7,7,1), cmap='gray', vmin=0, vmax=1)
plt.colorbar()
plt.show()

matplotlib.pyplot.imshow(ndimage.convolve(skimage.util.random_noise(lena, mode='gaussian', var=0.002), gaussian_filter(7,7,1)), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.hist(ndimage.convolve(skimage.util.random_noise(lena, mode='gaussian', var=0.002), gaussian_filter(7,7,1)))
plt.show()

matplotlib.pyplot.imshow(skimage.util.random_noise(lena, mode='s&p', amount=0.05), cmap='gray', vmin=0, vmax=1,)  
plt.show()
plt.hist(skimage.util.random_noise(lena, mode='s&p', amount=0.05))
plt.show()

matplotlib.pyplot.imshow(ndimage.convolve(skimage.util.random_noise(lena, mode='s&p', amount=0.05), np.ones((7,7))/(7.0*7.0)), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.hist(ndimage.convolve(skimage.util.random_noise(lena, mode='s&p', amount=0.05), np.ones((7,7))/(7.0*7.0)))  
plt.show()

matplotlib.pyplot.imshow(ndimage.convolve(skimage.util.random_noise(lena, mode='s&p', amount=0.05), gaussian_filter(7,7,1)), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.hist(ndimage.convolve(skimage.util.random_noise(lena, mode='s&p', amount=0.05), gaussian_filter(7,7,1)))
plt.show()

print(PSNR(lena, ndimage.convolve(skimage.util.random_noise(lena, mode='s&p', amount=0.05), np.ones((7,7))/(7.0*7.0))))
print(PSNR(lena, ndimage.convolve(skimage.util.random_noise(lena, mode='s&p', amount=0.05), gaussian_filter(7,7,1))))


matplotlib.pyplot.imshow(ndimage.median_filter(skimage.util.random_noise(lena, mode='s&p', amount=0.05), size=7), cmap='gray', vmin=0, vmax=1)
plt.show()
plt.hist(ndimage.median_filter(skimage.util.random_noise(lena, mode='s&p', amount=0.05), size=7))
plt.show()
print(PSNR(lena, ndimage.median_filter(skimage.util.random_noise(lena, mode='s&p', amount=0.05), size=7)))

matplotlib.pyplot.imshow(ndimage.convolve(cameraman, gaussian_filter(7,7,1)), cmap='gray', vmin=0, vmax=1)
plt.show()
matplotlib.pyplot.imshow(cameraman - ndimage.convolve(cameraman, gaussian_filter(7,7,1)), cmap='gray', vmin=0, vmax=1)
plt.show()

matplotlib.pyplot.imshow(cameraman + (cameraman - ndimage.convolve(cameraman, gaussian_filter(7,7,1))), cmap='gray', vmin=0, vmax=1)
plt.show()

matplotlib.pyplot.imshow(cameraman + (0.5 * (cameraman - ndimage.convolve(cameraman, gaussian_filter(7,7,1)))), cmap='gray', vmin=0, vmax=1)
plt.show()