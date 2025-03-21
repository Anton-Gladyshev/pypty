Module pypty.utils
==================

Functions
---------

`apply_defocus_probe(probe, distance, acc_voltage, pixel_size_x_A, pixel_size_y_A, default_complex, default_float, xp)`
:   Apply a defocus phase shift to a probe in Fourier space.
    
    Parameters
    ----------
    probe : ndarray
        The input probe wavefunction.
    distance : float
        Defocus distance in meters.
    acc_voltage : float
        Acceleration voltage in kiloelectronvolts (keV).
    pixel_size_x_A : float
        Pixel size along x in angstroms.
    pixel_size_y_A : float
        Pixel size along y in angstroms.
    default_complex : dtype
        Complex data type for computation.
    default_float : dtype
        Float data type for computation.
    xp : module
        Numerical backend (NumPy or CuPy).
    
    Returns
    -------
    ndarray
        The defocused probe.

`apply_probe_modulation(probe, extra_probe_defocus, acc_voltage, pixel_size_x_A, pixel_size_y_A, aberrations, print_flag, beam_ctf, n_hermite_probe_modes, defocus_spread_modes, probe_marker, default_complex, default_float, xp)`
:   Apply defocus, aberrations, Hermite mode generation, and other modulations to the probe.
    
    Parameters
    ----------
    probe : ndarray
        Initial probe.
    extra_probe_defocus : float
        Defocus distance to apply.
    acc_voltage : float
        Accelerating voltage in keV.
    pixel_size_x_A : float
        Pixel size in x (Å).
    pixel_size_y_A : float
        Pixel size in y (Å).
    aberrations : list or ndarray
        List of aberration coefficients.
    print_flag : bool
        Whether to print info.
    beam_ctf : ndarray or None
        Optional beam CTF to apply.
    n_hermite_probe_modes : tuple or None
        Number of Hermite modes in (y, x).
    defocus_spread_modes : list or None
        Defocus values to generate additional modes.
    probe_marker : ndarray or None
        Probe assignment array for multi-scenario.
    default_complex : dtype
        Complex type.
    default_float : dtype
        Float type.
    xp : module
        Numerical backend.
    
    Returns
    -------
    ndarray
        Modulated probe array.

`complex_grad_to_mag_grad(grad, abs, phase)`
:   Calculate a magnitude gradient from a complex gradient and separate magnitude and phase arrays.
    
    Parameters
    ----------
    grad : array_like
        The complex gradient.
    abs : array_like
        The magnitude array.
    phase : array_like
        The phase array.
    
    Returns
    -------
    array_like
        The magnitude gradient.

`complex_grad_to_phase_abs_grad(grad, array)`
:   Compute the phase gradient and negative amplitude gradient from a complex gradient.
    
    Parameters
    ----------
    grad : array_like
        The Wirtinger derivative (dL/dz*).
    array : array_like
        The complex array (z = exp(-a + i*phase)).
    
    Returns
    -------
    tuple of array_like
        A tuple containing:
        - Phase gradient (dL/dp).
        - Negative amplitude gradient (dL/da).

`complex_grad_to_phase_grad(grad, array)`
:   Convert a Wirtinger derivative to the gradient with respect to the phase.
    
    Parameters
    ----------
    grad : array_like
        The Wirtinger derivative (dL/dz*).
    array : array_like
        The complex array (z = |z| exp(i*phase)).
    
    Returns
    -------
    array_like
        The phase gradient (dL/dp).

`construct_update_abs_proto_phase(object_grad, obj)`
:   Compute object updates projected along phase gradients.
    
    Parameters
    ----------
    object_grad : array_like
        The gradient of the object.
    obj : array_like
        The current object array.
    
    Returns
    -------
    array_like
        The computed update for the object.

`convert_num_to_nmab(num_abs)`
:   Convert a number of aberration terms to (n, m, ab) indices based on Krivanek notation.
    
    Parameters
    ----------
    num_abs : int
        Number of aberration coefficients.
    
    Returns
    -------
    tuple of lists
        Lists of n, m, and ab strings ('', 'a', or 'b') for each aberration mode.

`convert_to_nxs(folder_path, output_file)`
:   Convert saved PyPty reconstruction data to NeXus (.nxs) format.
    
    Parameters
    ----------
    folder_path : str
        Directory containing saved reconstruction files.
    output_file : str
        Path where the NeXus file will be saved.
    
    Returns
    -------
    None

`convert_to_string(dicti2, strip_dataset_from_params=True)`
:   Convert parameter dictionary to string format, including lambda serialization.
    
    Parameters
    ----------
    dicti2 : dict
        Original parameter dictionary.
    strip_dataset_from_params : bool, optional
        Whether to exclude 'dataset' key (default is True).
    
    Returns
    -------
    dict
        Dictionary with string values.

`create_probe_from_nothing(probe, data_pad, mean_pattern, aperture_mask, tilt_mode, tilts, dataset, estimate_aperture_based_on_binary, pixel_size_x_A, acc_voltage, data_multiplier, masks, data_shift_vector, data_bin, upsample_pattern, default_complex_cpu, print_flag, algorithm, measured_data_shape, n_obj_modes, probe_marker, recon_type, defocus_array, Cs)`
:   Generate an initial probe guess when no valid probe is provided.
    
    Depending on the input, this function either uses an aperture mask, computes a mean pattern
    from the dataset, or adjusts an existing mean pattern to generate a probe. It applies shifting,
    binning, padding, and scaling to produce a probe suitable for the specified reconstruction type.
    
    Parameters
    ----------
    probe : ndarray, str, or None
        Input probe. If set to "aperture", the aperture mask is used. If None, the probe is generated
        based on the mean pattern.
    data_pad : int
        Padding size applied to the data.
    mean_pattern : ndarray or None
        Mean pattern used to generate the probe if no probe is provided.
    aperture_mask : ndarray
        Aperture mask used when probe is set to "aperture".
    tilt_mode : bool
        Flag indicating if tilt mode is active.
    tilts : ndarray
        Tilt values.
    dataset : ndarray
        Measured dataset.
    estimate_aperture_based_on_binary : bool or float
        Factor used to threshold the dataset for aperture estimation.
    pixel_size_x_A : float
        Pixel size in the x-direction in angstroms.
    acc_voltage : float
        Acceleration voltage in keV.
    data_multiplier : float
        Factor to scale the data intensity.
    masks : ndarray or None
        Optional masks to apply to the mean pattern.
    data_shift_vector : list or tuple of int
        Vector indicating the shift to be applied to the data.
    data_bin : int
        Binning factor.
    upsample_pattern : int
        Upsampling factor applied to the pattern.
    default_complex_cpu : dtype
        Complex data type for CPU computations.
    print_flag : int
        Flag controlling verbosity.
    algorithm : str
        Identifier for the reconstruction algorithm.
    measured_data_shape : tuple
        Shape of the measured data.
    n_obj_modes : int
        Number of object modes.
    probe_marker : ndarray or None
        Marker array for probe scenarios.
    recon_type : str
        Type of reconstruction ("near_field" or "far_field").
    defocus_array : ndarray
        Array of defocus values.
    Cs : float
        Spherical aberration coefficient.
    
    Returns
    -------
    ndarray
        The generated probe as a complex array.

`create_spatial_frequencies(px, py, shape, damping_cutoff_multislice, smooth_rolloff, default_float)`
:   Generate spatial frequency grids and corresponding masks for multislice simulations.
    
    Parameters
    ----------
    px : float
        Pixel size in the x-direction.
    py : float
        Pixel size in the y-direction.
    shape : int
        Size of the grid.
    damping_cutoff_multislice : float
        Damping cutoff factor for multislice simulations.
    smooth_rolloff : float
        Smoothing rolloff parameter.
    default_float : data-type
        Data type for computations.
    
    Returns
    -------
    tuple
        Tuple containing:
        - q2: 2D array of squared spatial frequencies.
        - qx: 2D array of spatial frequencies in x.
        - qy: 2D array of spatial frequencies in y.
        - exclude_mask: Mask in Fourier space.
        - exclude_mask_ishift: Unshifted mask.

`create_static_background_from_nothing(static_background, probe, damping_cutoff_multislice, data_pad, upsample_pattern, default_float, recon_type)`
:   Generate an initial static background if none is provided.
    
    Parameters
    ----------
    static_background : float or ndarray
        Initial static background value or None.
    probe : ndarray
        Probe wavefunction.
    damping_cutoff_multislice : float
        Maximum spatial frequency used.
    data_pad : int
        Padding to be applied.
    upsample_pattern : int
        Upsampling factor used.
    default_float : dtype
        Data type for output.
    recon_type : str
        Type of reconstruction ('near_field' or 'far_field').
    
    Returns
    -------
    ndarray
        Initialized static background.

`delete_dataset_from_params(params_path)`
:   Delete the 'dataset' key from saved parameter file.
    
    Parameters
    ----------
    params_path : str
        Path to the pickled parameters file.
    
    Returns
    -------
    None

`downsample_something(something, upsample, xp)`
:   Downsample a 2D array.
    
    Parameters
    ----------
    something : ndarray
        The 2D array to be downsampled.
    upsample : int
        Downsampling factor.
    xp : module
        Array module, e.g., numpy or cupy.
    
    Returns
    -------
    ndarray
        The downsampled array.

`downsample_something_3d(something, upsample, xp)`
:   Downsample a 3D array along the last two axes.
    
    Parameters
    ----------
    something : ndarray
        The 3D array to be downsampled.
    upsample : int
        Downsampling factor.
    xp : module
        Array module, e.g., numpy or cupy.
    
    Returns
    -------
    ndarray
        The downsampled 3D array.

`fourier_clean(array, cutoff=0.66, mask=None, rolloff=0, default_float=numpy.float32, xp=<module 'numpy' from '/Users/anton/miniconda/lib/python3.10/site-packages/numpy/__init__.py'>)`
:   Apply a Fourier filter to the input array. Supports 2D or 3D arrays.
    
    Parameters
    ----------
    array : array_like
        Input array (2D or 3D) to be filtered.
    cutoff : float, optional
        Cutoff frequency (default is 0.66).
    mask : array_like or None, optional
        Predefined mask to apply. If None, a mask is generated.
    rolloff : float, optional
        Rolloff parameter for smoothing the mask (default is 0).
    default_float : data-type, optional
        Data type for computations (default is cp.float32).
    xp : module, optional
        Array module (default is cp).
    
    Returns
    -------
    array_like
        The filtered array after applying the Fourier filter.

`fourier_clean_3d(array, cutoff=0.66, mask=None, rolloff=0, default_float=numpy.float32, xp=<module 'numpy' from '/Users/anton/miniconda/lib/python3.10/site-packages/numpy/__init__.py'>)`
:   Apply a 3D Fourier filter to the input array.
    
    Parameters
    ----------
    array : array_like
        Input 3D array to be filtered.
    cutoff : float, optional
        Cutoff frequency (default is 0.66).
    mask : array_like or None, optional
        Predefined mask to apply. If None, a mask is generated.
    rolloff : float, optional
        Rolloff parameter for smoothing the mask (default is 0).
    default_float : data-type, optional
        Data type for computations (default is cp.float32).
    xp : module, optional
        Array module (default is cp).
    
    Returns
    -------
    array_like
        The filtered array after applying the Fourier filter.

`generate_hermite_modes(main_mode, n_herm_x, n_herm_y, default_complex, xp)`
:   Generate Hermite polynomial-based probe modes from a main mode.
    
    Parameters
    ----------
    main_mode : ndarray
        The main probe mode.
    n_herm_x : int
        Max Degree of Hermite polynomials in x.
    n_herm_y : int
        Max Degree of Hermite polynomials in y.
    default_complex : dtype
        Complex data type to use.
    xp : module
        Numerical backend.
    
    Returns
    -------
    ndarray
        Stack of Hermite-based probe modes.

`generate_mask_for_grad_from_pos(shapex, shapey, positions_list, shape_footprint_x, shape_footprint_y, shrink=0)`
:   Construct a binary mask from given positions and footprint dimensions.
    
    Parameters
    ----------
    shapex : int
        Width of the mask.
    shapey : int
        Height of the mask.
    positions_list : list of tuple
        List of (y, x) positions where the mask should be activated.
    shape_footprint_x : int
        Footprint width.
    shape_footprint_y : int
        Footprint height.
    shrink : int, optional
        Shrink factor to adjust the footprint (default is 0).
    
    Returns
    -------
    array_like
        The constructed binary mask.

`get_compute_batch(compute_batch, load_one_by_one, hist_size, measured_data_shape, memory_saturation, smart_memory, data_pad, obj_shape, probe_shape, dtype, propmethod, print_flag)`
:   Estimate the optimal compute batch size based on GPU memory usage.
    
    Parameters
    ----------
    compute_batch : int
        Initial guess or default.
    load_one_by_one : bool
        Whether data is streamed instead of fully loaded.
    hist_size : int
        History size for optimizers.
    measured_data_shape : tuple
        Shape of the input dataset.
    memory_saturation : float
        Proportion of GPU memory to use.
    smart_memory : callable or bool
        User-provided memory strategy.
    data_pad : int
        Padding applied to data.
    obj_shape : tuple
        Shape of the object array.
    probe_shape : tuple
        Shape of the probe array.
    dtype : str
        Data type string ('single' or 'double').
    propmethod : str
        Propagation method name.
    print_flag : int
        Verbosity.
    
    Returns
    -------
    tuple
        Suggested batch size, load_one_by_one flag, and memory strategy.

`get_ctf(aberrations, kx, ky, wavelength, angle_offset=0)`
:   Compute the scalar contrast transfer function (CTF) from aberrations.
    
    Parameters
    ----------
    aberrations : list or ndarray
        List of aberration coefficients.
    kx : ndarray
        Spatial frequency in x-direction.
    ky : ndarray
        Spatial frequency in y-direction.
    wavelength : float
        Electron wavelength.
    angle_offset : float, optional
        Additional rotation angle in radians (default is 0).
    
    Returns
    -------
    ndarray
        The computed CTF.

`get_ctf_derivatives(aberrations, kx, ky, wavelength, angle_offset=0)`
:   Compute spatial derivatives of the CTF with respect to kx and ky.
    
    Parameters
    ----------
    aberrations : list or ndarray
        List of aberration coefficients.
    kx : ndarray
        Spatial frequency in x-direction.
    ky : ndarray
        Spatial frequency in y-direction.
    wavelength : float
        Electron wavelength.
    angle_offset : float, optional
        Additional rotation angle (default is 0).
    
    Returns
    -------
    tuple of ndarray
        Derivatives of CTF with respect to kx and ky.

`get_ctf_gradient_rotation_angle(aberrations, kx, ky, wavelength, angle_offset=0)`
:   Compute the gradient of the phase with respect to rotation angle.
    
    Parameters
    ----------
    aberrations : list or ndarray
        List of aberration coefficients.
    kx : ndarray
        Spatial frequency in x-direction.
    ky : ndarray
        Spatial frequency in y-direction.
    wavelength : float
        Electron wavelength.
    angle_offset : float, optional
        Additional angular offset (default is 0).
    
    Returns
    -------
    tuple of ndarray
        Derivatives of the CTF gradient in x and y directions with respect to angular change.

`get_ctf_matrix(kx, ky, num_abs, wavelength, xp=<module 'numpy' from '/Users/anton/miniconda/lib/python3.10/site-packages/numpy/__init__.py'>)`
:   Generate a matrix of phase contributions for all aberration modes.
    
    Parameters
    ----------
    kx : ndarray
        Spatial frequency in x-direction.
    ky : ndarray
        Spatial frequency in y-direction.
    num_abs : int
        Number of aberration coefficients.
    wavelength : float
        Electron wavelength.
    xp : module, optional
        Array module (default is cupy).
    
    Returns
    -------
    ndarray
        list of Zernike polynomials (num_abs, height, width) with phase contributions.

`get_cupy_memory_usage()`
:   Print current CuPy GPU memory usage statistics.
    
    Returns
    -------
    None

`get_steps_epoch(steps, epoch, default_float)`
:   Evaluate step values for the current epoch.
    
    Parameters
    ----------
    steps : list
        List of (multiplier, callable) or fixed values.
    epoch : int
        Current training epoch.
    default_float : dtype
        Float precision type.
    
    Returns
    -------
    list
        List of step values.

`get_value_for_epoch(func_or_value, epoch, default_float)`
:   Evaluate a list of values or functions at the current epoch.
    
    Parameters
    ----------
    func_or_value : list
        List of fixed values or callables.
    epoch : int
        Current epoch number.
    default_float : dtype
        Float precision type.
    
    Returns
    -------
    list
        Evaluated values.

`get_window(shape, r0, r_max, inverted=True)`
:   Create a circular cosine-tapered window mask.
    
    Parameters
    ----------
    shape : int
        Size of the square window.
    r0 : float
        Inner radius where tapering begins (normalized).
    r_max : float
        Outer radius where mask falls to zero (normalized).
    inverted : bool, optional
        If True, returns 1 - mask (default is True).
    
    Returns
    -------
    ndarray
        A 2D mask array of the specified shape.

`lambda_to_string(f)`
:   Extract lambda function source as a string.
    
    Parameters
    ----------
    f : function
        Lambda function.
    
    Returns
    -------
    str
        Extracted string source of the lambda.

`load_nexus_params(path_nexus)`
:   Load reconstruction parameters from a NeXus (.nxs) HDF5 file.
    
    Parameters
    ----------
    path_nexus : str
        Path to the .nxs file.
    
    Returns
    -------
    dict
        Dictionary of extracted parameters.

`load_params(path)`
:   Load parameter dictionary from a .pkl file.
    
    Parameters
    ----------
    path : str
        Path to the .pkl parameter file.
    
    Returns
    -------
    dict
        Loaded parameters.

`nmab_to_strings(possible_n, possible_m, possible_ab)`
:   Convert aberration indices into string identifiers in Krivanek notation.
    
    Parameters
    ----------
    possible_n : list of int
        List of radial indices.
    possible_m : list of int
        List of azimuthal indices.
    possible_ab : list of str
        List of aberration mode types ('', 'a', 'b').
    
    Returns
    -------
    list of str
        List of formatted aberration identifiers like 'C30a', 'C11', etc.

`padfft(array, pad)`
:   Pad the input array in Fourier space by padding its FFT.
    
    Parameters
    ----------
    array : ndarray
        Input array to be padded.
    pad : int
        Number of pixels to pad on each side.
    
    Returns
    -------
    ndarray
        The padded array in spatial domain.

`padprobetodatafarfield(probe, measured_data_shape, data_pad, upsample_pattern)`
:   Pad or crop a probe in Fourier space to match far-field data dimensions.
    
    Parameters
    ----------
    probe : ndarray
        The probe wavefunction.
    measured_data_shape : tuple
        Shape of the measured data.
    data_pad : int
        Padding applied to the data.
    upsample_pattern : int
        Upsampling factor used in the reconstruction.
    
    Returns
    -------
    ndarray
        Adjusted probe wavefunction.

`padprobetodatanearfield(probe, measured_data_shape, data_pad, upsample_pattern)`
:   Pad or crop a probe for near-field reconstruction.
    
    This function adjusts the probe wavefunction by padding or cropping it to match the
    near-field measured data dimensions after upsampling and padding.
    
    Parameters
    ----------
    probe : ndarray
        The input probe wavefunction.
    measured_data_shape : tuple
        Shape of the measured data.
    data_pad : int
        Padding size applied to the data.
    upsample_pattern : int
        Upsampling factor applied to the measured data.
    
    Returns
    -------
    ndarray
        The adjusted probe wavefunction.

`phase_cross_corr_align(im_ref_fft, im_2_fft, refine_box_dim, upsample, x_real, y_real, shift_y=None, shift_x=None)`
:   Align two FFT-transformed images using phase cross-correlation.
    
    Parameters
    ----------
    im_ref_fft : ndarray
        Reference image FFT.
    im_2_fft : ndarray
        FFT of the image to be aligned.
    refine_box_dim : int
        Size of the interpolation box for sub-pixel alignment.
    upsample : int
        Upsampling factor for interpolation.
    x_real : ndarray
        Real space x grid.
    y_real : ndarray
        Real space y grid.
    shift_y : float or None
        Predefined shift in y (optional).
    shift_x : float or None
        Predefined shift in x (optional).
    
    Returns
    -------
    ndarray
        Shifted FFT of the second image.

`prepare_main_loop_params(algorithm, probe, obj, positions, tilts, measured_data_shape, acc_voltage, allow_subPixel_shift=True, sequence=None, use_full_FOV=False, print_flag=0, default_float_cpu=numpy.float64, default_complex_cpu=numpy.complex128, default_int_cpu=numpy.int64, probe_constraint_mask=None, aperture_mask=None, extra_space_on_side_px=0)`
:   Prepare main loop parameters for reconstruction.
    
    This function adjusts scan positions, pads the object if necessary, handles subpixel corrections,
    and computes the electron wavelength based on the accelerating voltage.
    
    Parameters
    ----------
    algorithm : any
        Identifier for the reconstruction algorithm.
    probe : ndarray
        The probe array.
    obj : ndarray
        The object array.
    positions : ndarray
        Array of scan positions.
    tilts : ndarray
        Array of tilt angles.
    measured_data_shape : tuple
        Shape of the measured data.
    acc_voltage : float
        Accelerating voltage in keV.
    allow_subPixel_shift : bool, optional
        If True, compute subpixel corrections (default is True).
    sequence : list or callable, optional
        Sequence of indices for positions (default is None, which uses full range).
    use_full_FOV : bool, optional
        If True, use full field-of-view adjustments (default is False).
    print_flag : int, optional
        Verbosity flag (default is 0).
    default_float_cpu : data-type, optional
        Float data type for CPU computations (default is np.float64).
    default_complex_cpu : data-type, optional
        Complex data type for CPU computations (default is np.complex128).
    default_int_cpu : data-type, optional
        Integer data type for CPU computations (default is np.int64).
    probe_constraint_mask : ndarray or None, optional
        Optional mask for probe constraints.
    aperture_mask : ndarray or None, optional
        Optional aperture mask.
    extra_space_on_side_px : int, optional
        Extra padding (in pixels) to add to scan positions (default is 0).
    
    Returns
    -------
    tuple
        A tuple containing:
        - obj : ndarray
            The padded object array.
        - positions : ndarray
            Adjusted (rounded) scan positions.
        - int
            A placeholder zero (reserved for future use).
        - sequence : list
            The sequence of indices used.
        - wavelength : float
            Computed electron wavelength in angstroms.
        - full_pos_correction : ndarray
            Subpixel corrections for scan positions.
        - tilts_correction : ndarray
            Array of zeros with same shape as tilts (tilt corrections).
        - aperture_mask : ndarray or None
            The probe constraint mask or aperture mask if provided.

`prepare_saving_stuff(output_folder, save_loss_log, epoch_prev)`
:   Prepare folder and loss CSV for saving training logs.
    
    Parameters
    ----------
    output_folder : str
        Directory for results.
    save_loss_log : bool
        Whether to save loss values.
    epoch_prev : int
        Previous epoch index.
    
    Returns
    -------
    None

`preprocess_dataset(dataset, load_one_by_one, algorithm_type, recon_type, data_shift_vector, data_bin, data_pad, upsample_pattern, data_multiplier, xp, force_pad)`
:   Apply preprocessing steps to the dataset including shifting, binning, padding, and scaling.
    
    Parameters
    ----------
    dataset : ndarray
        The input dataset.
    load_one_by_one : bool
        Whether data is loaded incrementally.
    algorithm_type : str
        Type of reconstruction algorithm.
    recon_type : str
        Type of reconstruction (e.g., near_field, far_field).
    data_shift_vector : list of int
        Vector indicating pixel shift in y and x.
    data_bin : int
        Binning factor.
    data_pad : int
        Padding size.
    upsample_pattern : int
        Upsampling factor for the pattern.
    data_multiplier : float
        Factor to scale data intensity.
    xp : module
        Array module, e.g., numpy or cupy.
    force_pad : bool
        If True, apply forced padding.
    
    Returns
    -------
    tuple
        Tuple containing:
        - preprocessed dataset
        - data_shift_vector
        - data_bin
        - data_pad
        - data_multiplier

`print_pypty_header(data_path, output_folder, save_loss_log)`
:   Print formatted header announcing start of reconstruction.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset.
    output_folder : str
        Directory where results are saved.
    save_loss_log : bool
        Whether loss logging is enabled.
    
    Returns
    -------
    None

`print_recon_state(t0, algorithm, epoch, current_loss, current_sse, current_obj_step, current_probe_step, current_probe_pos_step, current_tilts_step, current_static_background_step, current_aberrations_array_step, current_beam_current_step, current_hist_length, print_flag)`
:   Display current reconstruction progress including loss, optimization state, and updates.
    
    Parameters
    ----------
    t0 : float
        Start time of the epoch (Unix timestamp).
    algorithm : str
        Name of the loss or optimization algorithm used.
    epoch : int
        Current training epoch.
    current_loss : float
        Loss value at current epoch.
    current_sse : float
        Sum of squared errors.
    current_obj_step : bool
        Whether the object is being updated.
    current_probe_step : bool
        Whether the probe is being updated.
    current_probe_pos_step : bool
        Whether the scan grid is being updated.
    current_tilts_step : bool
        Whether tilt corrections are being updated.
    current_static_background_step : bool
        Whether static background is being updated.
    current_aberrations_array_step : bool
        Whether aberration coefficients are being updated.
    current_beam_current_step : bool
        Whether beam current is being updated.
    current_hist_length : int
        Optimizer memory length (0=GD, 1=CG, >1=BFGS).
    print_flag : int
        Verbosity flag: 0 = silent, 1 = single-line print, 2 = verbose.
    
    Returns
    -------
    None

`save_current_checkpoint_obj_probe(output_folder, obj, probe, tilts_correction, full_pos_correction, positions, tilts, static_background, current_probe_step, current_obj_step, current_probe_pos_step, current_tilts_step, current_static_background_step, current_aberrations_array_step, aberrations_array, beam_current, bcstep, xp)`
:   Save intermediate reconstruction data as checkpoints.
    
    This function saves the current state of the object, probe, tilt corrections, scan positions,
    static background, and aberrations to disk. It is intended to allow resuming reconstruction
    from the last checkpoint.
    
    Parameters
    ----------
    output_folder : str
        Directory where checkpoint files will be saved.
    obj : ndarray or GPU array
        The current object array.
    probe : ndarray or GPU array
        The current probe array.
    tilts_correction : ndarray
        Correction values for tilt angles.
    full_pos_correction : ndarray
        Sub-pixel correction values for scan positions.
    positions : ndarray
        Scan positions array.
    tilts : ndarray
        Tilt angles array.
    static_background : ndarray
        Static background array.
    current_probe_step : bool
        Flag indicating whether to save the probe.
    current_obj_step : bool
        Flag indicating whether to save the object.
    current_probe_pos_step : bool
        Flag indicating whether to save the scan positions.
    current_tilts_step : bool
        Flag indicating whether to save the tilt angles.
    current_static_background_step : bool
        Flag indicating whether to save the static background.
    current_aberrations_array_step : bool
        Flag indicating whether to save the aberrations array.
    aberrations_array : ndarray or GPU array
        The current aberrations array.
    beam_current : ndarray or GPU array or None
        The current beam current array.
    bcstep : bool
        Flag indicating whether to save the beam current.
    xp : module
        Numerical backend (e.g., numpy or cupy).
    
    Returns
    -------
    None

`save_params(params_path, params, strip_dataset_from_params)`
:   Save parameters to a .pkl file, optionally removing the dataset.
    
    Parameters
    ----------
    params_path : str
        Output path for the parameter file.
    params : dict
        Parameter dictionary to save.
    strip_dataset_from_params : bool
        If True, remove the dataset entry.
    
    Returns
    -------
    None

`save_updated_arrays(output_folder, epoch, current_probe_step, current_probe_pos_step, current_tilts_step, current_obj_step, obj, probe, tilts_correction, full_pos_correction, positions, tilts, static_background, current_aberrations_array_step, current_static_background_step, count, current_loss, current_sse, aberrations, beam_current, current_beam_current_step, save_flag, save_loss_log, constraint_contributions, actual_step, count_linesearch, d_value, new_d_value, current_update_step_bfgs, t0, xp, warnings)`
:   Save current reconstruction state and log loss metrics during training.
    
    This function saves checkpoints for object, probe, tilts, scan positions, static background, 
    aberration coefficients, and beam current if specified. It also logs loss and constraint 
    contributions in a CSV file if logging is enabled.
    
    Parameters
    ----------
    output_folder : str
        Directory where files will be saved.
    epoch : int
        Current epoch number.
    current_probe_step : bool
        Whether to save the current probe.
    current_probe_pos_step : bool
        Whether to save current scan positions.
    current_tilts_step : bool
        Whether to save current tilts.
    current_obj_step : bool
        Whether to save the current object.
    obj : ndarray or xp.ndarray
        Object array to save.
    probe : ndarray or xp.ndarray
        Probe array to save.
    tilts_correction : ndarray
        Tilt correction values.
    full_pos_correction : ndarray
        Sub-pixel scan position correction.
    positions : ndarray
        Integer scan positions.
    tilts : ndarray
        Original tilt values.
    static_background : ndarray or xp.ndarray
        Static background array.
    current_aberrations_array_step : bool
        Whether to save aberration array.
    current_static_background_step : bool
        Whether to save static background.
    count : int
        Not used inside the function.
    current_loss : float
        Current loss value.
    current_sse : float
        Current sum of squared errors.
    aberrations : ndarray or xp.ndarray
        Array of aberration coefficients.
    beam_current : ndarray or xp.ndarray
        Array of beam current values.
    current_beam_current_step : bool
        Whether to save beam current.
    save_flag : bool
        Whether to trigger checkpoint saving.
    save_loss_log : bool or int
        Whether to log loss. If set to 2, log full breakdown of constraints.
    constraint_contributions : list
        List of constraint term contributions to the loss.
    actual_step : float
        Step size applied in the optimizer.
    count_linesearch : int
        Number of line search iterations.
    d_value : float
        Initial directional derivative.
    new_d_value : float
        New directional derivative after the step.
    current_update_step_bfgs : float
        Step size suggested by BFGS or optimizer.
    t0 : float
        Start time of the epoch (used for timing).
    xp : module
        NumPy or CuPy module used for computation.
    warnings : str
        Warning string to be logged.
    
    Returns
    -------
    None

`shift_probe_fourier(probe, shift_px)`
:   Shift a probe in Fourier space by applying a phase ramp.
    
    Parameters
    ----------
    probe : array_like
        The input probe array.
    shift_px : tuple of float
        Shift in pixels (y, x).
    
    Returns
    -------
    tuple
        Tuple containing the shifted probe, the phase mask, the Fourier transform of the probe,
        and the spatial frequency grids (maskx, masky).

`string_params_to_usefull_params(params)`
:   Convert string-encoded lambdas in parameter dictionary back to callables.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary possibly containing lambda strings.
    
    Returns
    -------
    dict
        Updated dictionary with callables.

`string_to_lambda(lambda_string)`
:   Convert stringified lambda expression to a Python function.
    Parameters
    ----------
    lambda_string : str
        Lambda string to evaluate.
    
    Returns
    -------
    callable or str
        The resulting function or original string if evaluation fails.

`try_to_gpu(obj, probe, positions, full_pos_correction, tilts, tilts_correction, masks, defocus_array, slice_distances, aperture_mask, dataset, load_one_by_one, static_background, aberrations_array, beam_current, default_float, default_complex, default_int, xp)`
:   Convert all key reconstruction variables to GPU arrays if using CuPy.
    
    Parameters
    ----------
    obj : ndarray
        Object array.
    probe : ndarray
        Probe array.
    positions : ndarray
        Integer scan positions.
    full_pos_correction : ndarray
        Sub-pixel scan grid correction.
    tilts : ndarray
        Tilt values.
    tilts_correction : ndarray
        Tilt corrections.
    masks : ndarray or None
        Optional segmentation or region masks.
    defocus_array : ndarray
        Array of defocus values per position.
    slice_distances : ndarray
        Slice spacing in multislice simulations.
    aperture_mask : ndarray or None
        Probe aperture mask.
    dataset : ndarray
        Measured dataset.
    load_one_by_one : bool
        Whether dataset is streamed from disk.
    static_background : ndarray or None
        Static background array.
    aberrations_array : ndarray or None
        Array of aberration coefficients.
    beam_current : ndarray or None
        Beam current scaling factor.
    default_float : dtype
        Float precision dtype for casting.
    default_complex : dtype
        Complex precision dtype for casting.
    default_int : dtype
        Integer dtype for casting.
    xp : module
        Numerical backend (`numpy` or `cupy`).
    
    Returns
    -------
    tuple
        The same variables in GPU format (if using CuPy), with proper types.

`try_to_initialize_beam_current(beam_current, measured_data_shape, default_float, xp)`
:   Initialize beam current array or pad if it's too short.
    
    Parameters
    ----------
    beam_current : ndarray or None
        Existing beam current values.
    measured_data_shape : tuple
        Shape of measured dataset.
    default_float : dtype
        Float type for the array.
    xp : module
        NumPy or CuPy.
    
    Returns
    -------
    ndarray
        Initialized or padded beam current.

`upsample_something(something, upsample, scale=True, xp=<module 'numpy' from '/Users/anton/miniconda/lib/python3.10/site-packages/numpy/__init__.py'>)`
:   Upsample a 2D array.
    
    Parameters
    ----------
    something : ndarray
        The 2D array to be upsampled.
    upsample : int
        Upsampling factor.
    scale : bool, optional
        If True, scales the result to conserve total sum (default is True).
    xp : module, optional
        Array module (default is numpy).
    
    Returns
    -------
    ndarray
        The upsampled array.

`upsample_something_3d(something, upsample, scale=True, xp=<module 'numpy' from '/Users/anton/miniconda/lib/python3.10/site-packages/numpy/__init__.py'>)`
:   Upsample a 3D array along the last two axes.
    
    Parameters
    ----------
    something : ndarray
        The 3D array to be upsampled.
    upsample : int
        Upsampling factor.
    scale : bool, optional
        If True, scales the upsampled result to conserve total sum (default is True).
    xp : module, optional
        Array module, e.g., numpy or cupy (default is numpy).
    
    Returns
    -------
    ndarray
        The upsampled 3D array.

`wolfe_1(value, new_value, d_value, step, wolfe_c1=0.5)`
:   Check the Armijo condition (Wolfe condition 1) for line search.
    
    Parameters
    ----------
    value : float
        The current function value.
    new_value : float
        The function value after the step.
    d_value : float
        The directional derivative at the current point.
    step : float
        Step size.
    wolfe_c1 : float, optional
        Armijo condition constant (default is 0.5).
    
    Returns
    -------
    bool
        True if the condition is satisfied, False otherwise.

`wolfe_2(d_value, new_d_value, wolfe_c2=0.9)`
:   Check the curvature condition (Wolfe condition 2) for line search.
    
    Parameters
    ----------
    d_value : float
        The directional derivative at the current point.
    new_d_value : float
        The directional derivative after the step.
    wolfe_c2 : float, optional
        Curvature condition constant (default is 0.9).
    
    Returns
    -------
    bool
        True if the condition is satisfied, False otherwise.