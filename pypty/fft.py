try:
    import cupy as cp
    import cupy.fft as sf
except:
    import numpy as cp
    import scipy.fft as sf
### 2D FFT
def fftfreq(length, sampling=1):
    return sf.fftfreq(length, sampling)
def ifftfreq(length, sampling=1):
    return sf.ifftfreq(length, sampling)
def fftshift(array):
    return sf.fftshift(array)
def ifftshift(array):
    return sf.ifftshift(array)
def shift_fft2(arr, axes=(0,1), overwrite_x=False):
    y=sf.fftshift(sf.fft2(arr, axes=axes, overwrite_x=overwrite_x), axes=axes)
    return y
def fft2(arr, axes=(0,1), overwrite_x=False):
    return sf.fft2(arr, axes=axes, overwrite_x=overwrite_x)
def ifft2(arr, axes=(0,1), overwrite_x=False):
    '''ifft2'''
    return sf.ifft2(arr, axes=axes, overwrite_x=overwrite_x)
def ifft2_ishift(arr, axes=(0,1), overwrite_x=False):
    return sf.ifft2(sf.ifftshift(arr, axes=axes), axes=axes, overwrite_x=overwrite_x)
### 3D FFT
def shift_fftn(arr, axes=(0,1,2)):
    y=sf.fftshift(sf.fftn(arr, axes=axes), axes=axes)
    return y
def ifftn_ishift(arr, axes=(0,1,2)):
    y=sf.ifftn(sf.ifftshift(arr, axes=axes), axes=axes)
    return y
def ifftn(arr, axes=(0,1,2)):
    y=sf.ifftn(arr, axes=axes)
    return y
def fftn(arr, axes=(0,1,2)):
    y=sf.fftn(arr, axes=axes)
    return y
