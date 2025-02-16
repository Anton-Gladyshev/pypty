try:
    import cupy as cp
    import cupyx.scipy.fft as sf
except:
    import numpy as cp
    import scipy.fft as sf
### 2D FFT
def fftfreq(length, sampling=1):
    """
    Wrapper for fftfreq function
    """
    return sf.fftfreq(length, sampling)
def ifftfreq(length, sampling=1):
    """
    Wrapper for ifftfreq function
    """
    return sf.ifftfreq(length, sampling)
def fftshift(array):
    """
    Wrapper for fftshift function
    """
    return sf.fftshift(array)
def ifftshift(array):
    """
    Wrapper for ifftshift function
    """
    return sf.ifftshift(array)
def shift_fft2(arr, axes=(0,1), overwrite_x=False):
    """
    Wrapper for xp.fft.fftshift(xp.fft.fft2(...)) functions
    """
    y=sf.fftshift(sf.fft2(arr, axes=axes, overwrite_x=overwrite_x), axes=axes)
    return y
def fft2(arr, axes=(0,1), overwrite_x=False):
    """
    Wrapper for fft2 function
    """
    return sf.fft2(arr, axes=axes, overwrite_x=overwrite_x)
def ifft2(arr, axes=(0,1), overwrite_x=False):
    """
    Wrapper for ifft2 function
    """
    '''ifft2'''
    return sf.ifft2(arr, axes=axes, overwrite_x=overwrite_x)
def ifft2_ishift(arr, axes=(0,1), overwrite_x=False):
    """
    Wrapper for xp.fft.ifft2(xp.fft.ifftshift(...)) function
    """
    return sf.ifft2(sf.ifftshift(arr, axes=axes), axes=axes, overwrite_x=overwrite_x)
### 3D FFT
def shift_fftn(arr, axes=(0,1,2)):
    """
    Wrapper for xp.fft.fftshift(xp.fft.fftn(...)) function
    """
    y=sf.fftshift(sf.fftn(arr, axes=axes), axes=axes)
    return y
def ifftn_ishift(arr, axes=(0,1,2)):
    """
    Wrapper for xp.fft.ifftn(xp.fft.ifftshift(...)) function
    """
    y=sf.ifftn(sf.ifftshift(arr, axes=axes), axes=axes)
    return y
def ifftn(arr, axes=(0,1,2)):
    """
    Wrapper for xp.fft.ifftn function
    """
    y=sf.ifftn(arr, axes=axes)
    return y
def fftn(arr, axes=(0,1,2)):
    """
    Wrapper for fftn function
    """
    y=sf.fftn(arr, axes=axes)
    return y
