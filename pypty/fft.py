try:
    import cupy as cp
    import cupyx.scipy.fft as sf
except:
    import numpy as cp
    import scipy.fft as sf
    
cp.fftback=sf
import numpy as np
import scipy.fft as sff
np.fftback=sff
    
    
### 2D FFT
def fftfreq(length, sampling=1, xp=cp):
    """
    Wrapper for fftfreq function.

    Parameters
    ----------
    length : int
        Length of the output array.
    sampling : float, optional
        Sample spacing (default is 1).
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        Discrete Fourier Transform sample frequencies.
    """
    return xp.fftback.fftfreq(length, sampling)

def ifftfreq(length, sampling=1, xp=cp):
    """
    Wrapper for ifftfreq function.

    Parameters
    ----------
    length : int
        Length of the output array.
    sampling : float, optional
        Sample spacing (default is 1).
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        Discrete inverse Fourier Transform sample frequencies.
    """
    return xp.fftback.ifftfreq(length, sampling)

def fftshift(array, xp=cp):
    """
    Wrapper for fftshift function.

    Parameters
    ----------
    array : ndarray
        Input array to be shifted.
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        Shifted array.
    """
    return xp.fftback.fftshift(array)

def ifftshift(array, xp=cp):
    """
    Wrapper for ifftshift function.

    Parameters
    ----------
    array : ndarray
        Input array to be inverse shifted.
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        Inverse shifted array.
    """
    return xp.fftback.ifftshift(array)

def shift_fft2(arr, axes=(0,1), overwrite_x=False, xp=cp):
    """
    Wrapper for xp.fft.fftshift(xp.fft.fft2(...)) functions.

    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the FFT (default is (0, 1)).
    overwrite_x : bool, optional
        If True, allow overwriting the input array (default is False).
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        2D Fourier Transform of the input array, shifted.
    """
    y=xp.fftback.fftshift(xp.fftback.fft2(arr, axes=axes, overwrite_x=overwrite_x), axes=axes)
    return y

def fft2(arr, axes=(0,1), overwrite_x=False, xp=cp):
    """
    Wrapper for fft2 function.

    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the FFT (default is (0, 1)).
    overwrite_x : bool, optional
        If True, allow overwriting the input array (default is False).
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        2D Fourier Transform of the input array.
    """
    return xp.fftback.fft2(arr, axes=axes, overwrite_x=overwrite_x)

def ifft2(arr, axes=(0,1), overwrite_x=False, xp=cp):
    """
    Wrapper for ifft2 function.

    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the inverse FFT (default is (0, 1)).
    overwrite_x : bool, optional
        If True, allow overwriting the input array (default is False).
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        2D inverse Fourier Transform of the input array.
    """
    return xp.fftback.ifft2(arr, axes=axes, overwrite_x=overwrite_x)

def ifft2_ishift(arr, axes=(0,1), overwrite_x=False, xp=cp):
    """
    Wrapper for xp.fft.ifft2(xp.fft.ifftshift(...)) function.

    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the inverse FFT (default is (0, 1)).
    overwrite_x : bool, optional
        If True, allow overwriting the input array (default is False).
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        2D inverse Fourier Transform of the input array, after shifting.
    """
    return xp.fftback.ifft2(xp.fftback.ifftshift(arr, axes=axes), axes=axes, overwrite_x=overwrite_x)

### 3D FFT
def shift_fftn(arr, axes=(0,1,2), xp=cp):
    """
    Wrapper for xp.fft.fftshift(xp.fft.fftn(...)) function.

    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the FFT (default is (0, 1, 2)).
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        N-dimensional Fourier Transform of the input array, shifted.
    """
    y=xp.fftback.fftshift(xp.fftback.fftn(arr, axes=axes), axes=axes)
    return y

def ifftn_ishift(arr, axes=(0,1,2), xp=cp):
    """
    Wrapper for xp.fft.ifftn(xp.fft.ifftshift(...)) function.

    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the inverse FFT (default is (0, 1, 2)).
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        N-dimensional inverse Fourier Transform of the input array, after shifting.
    """
    y=xp.fftback.ifftn(xp.fftback.ifftshift(arr, axes=axes), axes=axes)
    return y

def ifftn(arr, axes=(0,1,2), xp=cp):
    """
    Wrapper for ifftn function.

    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the inverse FFT (default is (0, 1, 2)).
    xp: mudule
        numpy or cupy, cpu or gpu backend

    Returns
    -------
    ndarray
        N-dimensional inverse Fourier Transform of the input array.
    """
    y=xp.fftback.ifftn(arr, axes=axes)
    return y

def fftn(arr, axes=(0,1,2), xp=cp):
    """
    Wrapper for fftn function.

    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the FFT (default is (0, 1, 2)).
    xp: mudule
        numpy or cupy, cpu or gpu backend
        
    Returns
    -------
    ndarray
        N-dimensional Fourier Transform of the input array.
    """
    y=xp.fftback.fftn(arr, axes=axes)
    return y

