Module pypty.fft
================

Functions
---------

`fft2(arr, axes=(0, 1), overwrite_x=False)`
:   Wrapper for fft2 function.
    
    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the FFT (default is (0, 1)).
    overwrite_x : bool, optional
        If True, allow overwriting the input array (default is False).
    
    Returns
    -------
    ndarray
        2D Fourier Transform of the input array.

`fftfreq(length, sampling=1)`
:   Wrapper for fftfreq function.
    
    Parameters
    ----------
    length : int
        Length of the output array.
    sampling : float, optional
        Sample spacing (default is 1).
    
    Returns
    -------
    ndarray
        Discrete Fourier Transform sample frequencies.

`fftn(arr, axes=(0, 1, 2))`
:   Wrapper for fftn function.
    
    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the FFT (default is (0, 1, 2)).
    
    Returns
    -------
    ndarray
        N-dimensional Fourier Transform of the input array.

`fftshift(array)`
:   Wrapper for fftshift function.
    
    Parameters
    ----------
    array : ndarray
        Input array to be shifted.
    
    Returns
    -------
    ndarray
        Shifted array.

`ifft2(arr, axes=(0, 1), overwrite_x=False)`
:   Wrapper for ifft2 function.
    
    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the inverse FFT (default is (0, 1)).
    overwrite_x : bool, optional
        If True, allow overwriting the input array (default is False).
    
    Returns
    -------
    ndarray
        2D inverse Fourier Transform of the input array.

`ifft2_ishift(arr, axes=(0, 1), overwrite_x=False)`
:   Wrapper for xp.fft.ifft2(xp.fft.ifftshift(...)) function.
    
    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the inverse FFT (default is (0, 1)).
    overwrite_x : bool, optional
        If True, allow overwriting the input array (default is False).
    
    Returns
    -------
    ndarray
        2D inverse Fourier Transform of the input array, after shifting.

`ifftfreq(length, sampling=1)`
:   Wrapper for ifftfreq function.
    
    Parameters
    ----------
    length : int
        Length of the output array.
    sampling : float, optional
        Sample spacing (default is 1).
    
    Returns
    -------
    ndarray
        Discrete inverse Fourier Transform sample frequencies.

`ifftn(arr, axes=(0, 1, 2))`
:   Wrapper for ifftn function.
    
    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the inverse FFT (default is (0, 1, 2)).
    
    Returns
    -------
    ndarray
        N-dimensional inverse Fourier Transform of the input array.

`ifftn_ishift(arr, axes=(0, 1, 2))`
:   Wrapper for xp.fft.ifftn(xp.fft.ifftshift(...)) function.
    
    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the inverse FFT (default is (0, 1, 2)).
    
    Returns
    -------
    ndarray
        N-dimensional inverse Fourier Transform of the input array, after shifting.

`ifftshift(array)`
:   Wrapper for ifftshift function.
    
    Parameters
    ----------
    array : ndarray
        Input array to be inverse shifted.
    
    Returns
    -------
    ndarray
        Inverse shifted array.

`shift_fft2(arr, axes=(0, 1), overwrite_x=False)`
:   Wrapper for xp.fft.fftshift(xp.fft.fft2(...)) functions.
    
    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the FFT (default is (0, 1)).
    overwrite_x : bool, optional
        If True, allow overwriting the input array (default is False).
    
    Returns
    -------
    ndarray
        2D Fourier Transform of the input array, shifted.

`shift_fftn(arr, axes=(0, 1, 2))`
:   Wrapper for xp.fft.fftshift(xp.fft.fftn(...)) function.
    
    Parameters
    ----------
    arr : ndarray
        Input array to be transformed.
    axes : tuple of int, optional
        Axes over which to compute the FFT (default is (0, 1, 2)).
    
    Returns
    -------
    ndarray
        N-dimensional Fourier Transform of the input array, shifted.