Module pypty.tcbf
=================

Functions
---------

`run_tcbf_alignment(params, binning_for_fit=[8], save=True, optimize_angle=True, aberrations=None, n_aberrations_to_fit=12, reference_type='bf', refine_box_dim=10, upsample=3, cross_corr_type='phase', cancel_large_shifts=None, pattern_blur_width=None, scan_pad=None, aperture=None, subscan_region=None, compensate_lowfreq_drift=False, append_lowfreq_shifts_to_params=True, interpolate_scan_factor=1, binning_cross_corr=1, phase_cross_corr_formula=False, f_scale_lsq=1, x_scale_lsq=1, loss_lsq='linear', tol_ctf=1e-08)`
:   Align and fit the beam contrast transfer function (CTF) using 4D-STEM data.
    
    This function estimates beam aberrations by aligning individual pixel images using 
    cross-correlation and fitting a CTF model. It supports iterative fitting with various 
    binning levels and options for low-frequency drift compensation.
    
    Parameters
    ----------
    params : dict
        Dictionary containing PyPTY experimental and reconstruction settings.
    binning_for_fit : list of int, optional
        List of binning factors for each iteration of the CTF fit.
    save : bool, optional
        Whether to save intermediate tcBF images and shift estimates.
    optimize_angle : bool, optional
        Whether to include probe rotation angle in the fit.
    aberrations : list or None, optional
        Initial guess for aberrations. If None, `n_aberrations_to_fit` zeros will be used.
    n_aberrations_to_fit : int, optional
        Number of aberrations to fit if no initial guess is provided.
    reference_type : str, optional
        Reference used for cross-correlation ("bf" or "zero").
    refine_box_dim : int, optional
        Radius (in pixels) of the interpolation box for sub-pixel shift refinement.
    upsample : int, optional
        Factor for refining cross-correlation to estimate sub-pixel shifts.
    cross_corr_type : str, optional
        Type of cross-correlation to use ("phase" or "classical").
    cancel_large_shifts : float or None, optional
        Threshold (0–1) to ignore large shift outliers in the fit.
    pattern_blur_width : int or None, optional
        Radius for optional circular blur mask applied to patterns.
    scan_pad : int or None, optional
        Number of scan pixels to pad around the dataset (auto if None).
    aperture : ndarray or None, optional
        Aperture mask. If None, attempts to extract from parameters.
    subscan_region : list or None, optional
        Subregion for CTF fitting: [left, top, right, bottom].
    compensate_lowfreq_drift : bool, optional
        Whether to compensate for pattern drift in large FOVs.
    append_lowfreq_shifts_to_params : bool, optional
        Whether to store low-frequency drift corrections in `params`.
    interpolate_scan_factor : int, optional
        Factor to upsample the scan via interpolation (experimental).
    binning_cross_corr : int, optional
        Binning factor before peak detection in cross-correlation.
    phase_cross_corr_formula : bool, optional
        Use analytical peak refinement formula for phase correlation.
    f_scale_lsq : float, optional
        Scaling factor for residuals in `least_squares`.
    x_scale_lsq : float, optional
        Scaling for initial step size in `least_squares`.
    loss_lsq : str, optional
        Loss function for `least_squares` optimization.
    tol_ctf : float, optional
        Tolerance (`ftol`) for stopping criterion in optimization.
    
    Returns
    -------
    pypty_params : dict
        Updated parameter dictionary with fitted aberrations, defocus, and potentially
        updated scan positions and rotation.
    Notes
    -----
    - Requires a scan dataset and optionally a precomputed aperture mask.
    - Intermediate results and diagnostics can be saved to disk if `save` is True.

`run_tcbf_compressed_alignment(params, num_iterations, save=True, optimize_angle=True, aberrations=None, n_aberrations_to_fit=12, reference_type='bf', refine_box_dim=10, upsample=3, cross_corr_type='phase', cancel_large_shifts=None, pattern_blur_width=None, scan_pad=None, aperture=None, subscan_region=None, compensate_lowfreq_drift=False, append_lowfreq_shifts_to_params=True, interpolate_scan_factor=1, binning_cross_corr=1, phase_cross_corr_formula=False, f_scale_lsq=1, x_scale_lsq=1, loss_lsq='linear', tol_ctf=1e-08)`
:   Perform a CTF alignment using compressed 4D-STEM data and masked bright-field regions.
    
    This function fits the beam CTF to the shifts between the individual pixel images of the 4d-stem dataset. It's the same as run_tcbf_alignment, but for compressed data. The shift estimation is done via cross-correaltion.
    
    Parameters
    ----------
    params : dict
        Dictionary containing experimental and reconstruction settings.
    num_iterations : int
        Number of fitting iterations to perform.
    save : bool, optional
        Whether to save intermediate tcBF images and shift maps. Default is True.
    optimize_angle : bool, optional
        Whether to include probe rotation angle in the CTF fit. Default is True.
    aberrations : list or None, optional
        Initial guess for the aberration coefficients. If None, it will be inferred or zero-initialized.
    n_aberrations_to_fit : int, optional
        Number of aberration coefficients to fit if `aberrations` is not provided. Default is 12.
    reference_type : str, optional
        "bf" to use the tcBF image as a reference, "zero" to use the central pixel. Default is "bf".
    refine_box_dim : int, optional
        Size of the cropped region around the cross-correlation peak for sub-pixel refinement. Default is 10.
    upsample : int, optional
        Upsampling factor for sub-pixel interpolation. Default is 3.
    cross_corr_type : str, optional
        Cross-correlation method: "phase" (recommended) or "classic". Default is "phase".
    cancel_large_shifts : float or None, optional
        Threshold to reject large shift outliers during fitting. Value between 0 and 1. Default is None.
    pattern_blur_width : int or None, optional
        Width of blur kernel for patterns prior to analysis. Default is None.
    scan_pad : int or None, optional
        Number of scan pixels to pad around the scan to prevent wrap-around. Default is auto.
    aperture : ndarray or None, optional
        Aperture mask defining pixels to analyze. If None, it will be loaded from `params`.
    subscan_region : list or None, optional
        Optional subregion [left, top, right, bottom] on which to perform the alignment. Default is None.
    compensate_lowfreq_drift : bool, optional
        Whether to compute and correct for slow drifting of the aperture over time. Default is False.
    append_lowfreq_shifts_to_params : bool, optional
        If True, saves the low-frequency correction back into `params`. Default is True.
    interpolate_scan_factor : int, optional
        Experimental: interpolate scan grid by this factor (e.g., 2 for 2x upsampled grid). Default is 1.
    binning_cross_corr : int, optional
        Binning factor applied to cross-correlation maps before refinement. Default is 1.
    phase_cross_corr_formula : bool, optional
        If True, uses analytical subpixel peak estimation for phase correlation. Default is False.
    f_scale_lsq : float, optional
        Scaling factor for least squares residuals (`f_scale`). Default is 1.
    x_scale_lsq : float, optional
        Initial step scaling (`x_scale`) for least squares. Default is 1.
    loss_lsq : str, optional
        Loss type for least squares optimizer. E.g., "linear", "huber". Default is "linear".
    tol_ctf : float, optional
        Tolerance for optimizer convergence (`ftol`). Default is 1e-8.
    
    Returns
    -------
    pypty_params : dict
        Updated dictionary of reconstruction parameters including fitted aberrations and scan rotation.
    
    Notes
    -----
    - Requires masks to define the compressed bright field regions.

`upsampled_tcbf(pypty_params, upsample=5, pad=10, compensate_lowfreq_drift=False, default_float=64, round_shifts=False, xp=<module 'numpy' from '/Users/anton/miniconda/lib/python3.10/site-packages/numpy/__init__.py'>, save=0, max_parallel_fft=100, bin_fac=1)`
:   Perform an upsampled tcBF (transmission coherent Bright Field) reconstruction.
    
    This function reconstructs a tcBF image on an upsampled scan grid from 4D-STEM data.
    It applies Fourier-based shifts to align the bright field pixel images and combines them into a final image.
    Prior to calling this function, it is recommended to run the tcBF alignment routine to update `pypty_params`.
    
    Parameters
    ----------
    pypty_params : dict
        Dictionary containing experimental parameters and reconstruction settings.
        This should include keys such as 'data_path', 'scan_size', 'aperture_mask', 'acc_voltage', etc.
    upsample : int, optional
        Upsampling factor for the scan grid. Default is 5.
    pad : int, optional
        Number of additional scan positions to pad on each side to avoid wrap-around artifacts.
        Default is 10.
    compensate_lowfreq_drift : bool, optional
        If True, compensates for low-frequency drift of the aperture.
        Requires that run_tcbf_alignment has been executed to provide drift corrections.
        Default is False.
    default_float : {64, 32}, optional
        Precision for floating point computations. Use 64 for higher precision or 32 for lower memory usage.
        Default is 64.
    round_shifts : bool, optional
        If True, shifts are rounded and alignment is performed using array roll operations.
        If False, FFT-based subpixel shifting is used. Default is False.
    xp : module, optional
        Backend array module (e.g., numpy or cupy). Default is cupy.
    save : bool or int, optional
        Flag to save the output image. If True, the image is saved to disk.
        Ignored if 'save_preprocessing_files' is set in `pypty_params`. Default is 0 (False).
    max_parallel_fft : int, optional
        Maximum number of FFTs to perform in parallel for vectorized processing.
        Default is 100.
    bin_fac : int, optional
        Binning factor for the data in reciprocal space. Default is 1 (no binning).
    
    Returns
    -------
    O_r : ndarray
        Real-valued tcBF image reconstructed on the upsampled grid.
    px_size_final : float
        Final pixel size in Ångströms after upsampling.