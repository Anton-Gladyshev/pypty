Module pypty.direct_ptychography
================================

Functions
---------

`wdd(pypty_params, eps_wiener=0.001, thresh=None, save=0)`
:   Perform Wigner Distribution Deconvolution.
    
    This function applies Wigner Distribution Deconvolution to the provided data, allowing for enhanced reconstruction of complex objects and probes.
    
    Parameters
    ----------
    pypty_params : dict
        Dictionary containing calibrated parameters, including paths and settings for data processing.
    eps_wiener : float, optional
        Epsilon parameter for the Wiener filter (default is 1e-3).
    thresh : float, optional
        Threshold for an alternative deconvolution approach. If provided, `eps_wiener` is ignored, and denominator values below this threshold are set to 1 while the corresponding numerator values are set to 0.
    save : int, optional
        Flag indicating whether to save the output files (default is 0, which means False). Ignored if `save_preprocessing_files` is provided in `pypty_params`.
    
    Returns
    -------
    o : 2D complex ndarray
        The deconvolved complex object.
    probe : 2D complex ndarray
        The reconstructed complex beam.
    
    Notes
    -----
    - The function handles both GPU (via CuPy) and CPU (via NumPy) computations based on the availability of the CuPy library.
    - The `pypty_params` dictionary must be prepaired via initilize module