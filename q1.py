import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform, exposure
from math import log10, sqrt
# Load images
lena = color.rgb2gray(io.imread('lena.tiff') )*255
tire = io.imread('tire.tif') 
cameraman = io.imread('cameraman.tif').astype(np.float64)



def PSNR(f, g):
    f = f.astype(np.float64)
    g = g.astype(np.float64)
    mse = np.mean((f - g) ** 2)
    if mse == 0:  # MSE is zero means no noise, hence PSNR is infinite
        return float('inf')
    max_pixel = 255.0
    psnr = 10 * log10((max_pixel ** 2) /mse)
    return psnr

def downsample(image):
    return transform.resize(image, (image.shape[0] // 4, image.shape[1] // 4), anti_aliasing=True)

lena_down = downsample((lena))
cameraman_down = downsample(cameraman)

def zoom(image, method):
    return transform.resize(image, (image.shape[0] * 4, image.shape[1] * 4), order=method)

# Nearest Neighbor Interpolation (order=0)
lena_nn = zoom(lena_down, 0)
cameraman_nn = zoom(cameraman_down, 0)

# Bilinear Interpolation (order=1)
lena_bilinear = zoom(lena_down, 1)
cameraman_bilinear = zoom(cameraman_down, 1)

# Bicubic Interpolation (order=3)
lena_bicubic = zoom(lena_down, 3)
cameraman_bicubic = zoom(cameraman_down, 3)

def plot_images(original, downsampled, nn, bilinear, bicubic, title):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title(f'Original {title}')

    plt.subplot(2, 3, 2)
    plt.imshow(downsampled, cmap='gray')
    plt.title(f'Downsampled {title}')

    plt.subplot(2, 3, 3)
    plt.imshow(nn, cmap='gray')
    plt.title(f'{title} Nearest Neighbor')

    plt.subplot(2, 3, 4)
    plt.imshow(bilinear, cmap='gray')
    plt.title(f'{title} Bilinear Interpolation')

    plt.subplot(2, 3, 5)
    plt.imshow(bicubic, cmap='gray')
    plt.title(f'{title} Bicubic Interpolation')

    plt.tight_layout()
    plt.show()

plot_images((lena), lena_down, lena_nn, lena_bilinear, lena_bicubic, "Lena")
plot_images(cameraman, cameraman_down, cameraman_nn, cameraman_bilinear, cameraman_bicubic, "Cameraman")

print(cameraman.min(), cameraman.max())
print(cameraman_nn.min(), cameraman_nn.max())
print((lena).min(), (lena).max())

print(f"Lena PSNR (Nearest Neighbor): {PSNR((lena), lena_nn)}")
print(f"Lena PSNR (Bilinear): {PSNR((lena), lena_bilinear)}")
print(f"Lena PSNR (Bicubic): {PSNR((lena), lena_bicubic)}")

print(f"Cameraman PSNR (Nearest Neighbor): {PSNR(cameraman, cameraman_nn)}")
print(f"Cameraman PSNR (Bilinear): {PSNR(cameraman, cameraman_bilinear)}")
print(f"Cameraman PSNR (Bicubic): {PSNR(cameraman, cameraman_bicubic)}")

def negative(image):
    return 1.0 - image

tire_scaled = tire / 255.0
tire_negative = negative(tire_scaled)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(tire_scaled, cmap='gray')
plt.title('Original Tire Image')

plt.subplot(1, 2, 2)
plt.imshow(tire_negative, cmap='gray')
plt.title('Negative Tire Image')
plt.show()

def power_law(image, gamma):
    return np.power(image, gamma)

gamma1 = 0.5
gamma2 = 1.3
tire_gamma_05 = power_law(tire_scaled, gamma1)
tire_gamma_13 = power_law(tire_scaled, gamma2)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(tire_gamma_05, cmap='gray')
plt.title('Gamma 0.5')

plt.subplot(1, 3, 2)
plt.imshow(tire_gamma_13, cmap='gray')
plt.title('Gamma 1.3')

plt.subplot(1, 3, 3)
plt.imshow(tire_scaled, cmap='gray')
plt.title('Original Tire Image')
plt.show()

tire_equalized = exposure.equalize_hist(tire_scaled)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(tire_equalized, cmap='gray')
plt.title('Equalized Tire Image')

plt.subplot(1, 2, 2)
plt.hist(tire_equalized.flatten(), bins=256, range=[0,1])
plt.title('Equalized Histogram')
plt.show()
