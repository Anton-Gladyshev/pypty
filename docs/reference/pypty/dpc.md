Module pypty.dpc
================

Functions
---------

`GetPLRotation(dpcx, dpcy)`
:   Estimate rotation angle that minimizes the curl of the DPC signal.
    
    Parameters
    ----------
    dpcx : ndarray
        X-component of the DPC signal (2D array).
    dpcy : ndarray
        Y-component of the DPC signal (2D array).
    
    Returns
    -------
    float
        Optimal rotation angle in radians.

`fft_based_dpc(pypty_params, hpass=0, lpass=0, save=False, comx=None, comy=None, plot=False)`
:   FFT-based DPC phase reconstruction. If you setted up the pypty_params properly, you would only need to specify the hpass and lpass values, both are non-negative floats.
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary with dataset and calibration settings.
    hpass : float, optional
        High-pass filtering coefficient (default is 0).
    lpass : float, optional
        Low-pass filtering coefficient (default is 0).
    save : bool, optional
        Whether to save the reconstructed phase (default is False).
    comx : ndarray or None, optional
        Precomputed center-of-mass x-component.
    comy : ndarray or None, optional
        Precomputed center-of-mass y-component.
    plot : bool, optional
        If True, display the phase reconstruction.
    
    Returns
    -------
    pot : ndarray
        Reconstructed 2D phase image.
    pypty_params : dict
        Updated parameter dictionary with computed COM and rotation angle.

`get_curl(angle, dpcx, dpcy)`
:   Compute the standard deviation of the curl of a rotated DPC vector field. This is the objective function for minimization. This particular function was copied from a DPC plugin written by Jordan Hachtel.
    
    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    dpcx : ndarray
        X-component of the DPC signal.
    dpcy : ndarray
        Y-component of the DPC signal.
    
    Returns
    -------
    float
        Standard deviation of the curl after rotation.

`get_curl_derivative(angle, dpcx, dpcy)`
:   Compute the derivative of the curl-based objective function with respect to rotation angle.
    
    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    dpcx : ndarray
        X-component of the DPC signal.
    dpcy : ndarray
        Y-component of the DPC signal.
    
    Returns
    -------
    float
        Derivative of the curl-based objective function.

`iterative_dpc(pypty_params, num_iterations=100, beta=0.5, hpass=0, lpass=0, step_size=0.1, COMx=None, COMy=None, px_size=None, print_flag=False, save=False, select=None, plot=True, use_backtracking=True, pad_width=5)`
:   Iterative DPC phase reconstruction. If you setted up the pypty_params properly, you would only need to specify the hpass and lpass values, both are non-negative floats.
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    num_iterations : int, optional
        Number of gradient descent iterations (default is 100).
    beta : float, optional
        Step reduction factor for backtracking (default is 0.5).
    hpass : float, optional
        High-pass filtering coefficient (default is 0).
    lpass : float, optional
        Low-pass filtering coefficient (default is 0).
    step_size : float, optional
        Initial gradient descent step size (default is 0.1).
    COMx : ndarray or None
        X-component of COM map.
    COMy : ndarray or None
        Y-component of COM map.
    px_size : float or None
        Scan step size in Ångströms.
    print_flag : bool, optional
        Whether to print progress information (default is False).
    save : bool, optional
        Whether to save the result to disk (default is False).
    select : ndarray or None
        Optional binary mask to constrain reconstruction.
    plot : bool, optional
        If True, plot the reconstruction result.
    use_backtracking : bool, optional
        Whether to use backtracking line search (default is True).
    pad_width : int, optional
        Padding width to suppress FFT boundary artifacts (default is 5).
    
    Returns
    -------
    padded_phase : ndarray
        Reconstructed 2D phase image.

`iterative_poisson_solver(laplace, num_iterations=100, beta=0.5, hpass=0, lpass=0, select=None, px_size=1, print_flag=False, step_size=0.1, use_backtracking=True, pad_width=1, xp=<module 'numpy' from '/Users/anton/miniconda/lib/python3.10/site-packages/numpy/__init__.py'>)`
:   Iterative solver for Poisson equation given a Laplacian map.
    
    Parameters
    ----------
    laplace : ndarray
        Input 2D array representing the Laplacian of the desired phase.
    num_iterations : int, optional
        Number of iterations (default is 100).
    beta : float, optional
        Step size reduction factor (default is 0.5).
    hpass : float, optional
        High-pass filtering parameter (default is 0).
    lpass : float, optional
        Low-pass filtering parameter (default is 0).
    select : ndarray or None, optional
        Optional binary mask to restrict updates.
    px_size : float, optional
        Pixel size in Ångströms (default is 1).
    print_flag : bool, optional
        If True, print convergence status (default is False).
    step_size : float, optional
        Initial gradient descent step size (default is 0.1).
    use_backtracking : bool, optional
        Whether to use backtracking line search (default is True).
    pad_width : int, optional
        Number of pixels to pad around the solution (default is 1).
    xp : module, optional
        Backend array library (NumPy or CuPy, default is NumPy).
    
    Returns
    -------
    ndarray
        Reconstructed 2D phase from the input Laplacian.