Module pypty.initialize
=======================

Functions
---------

`append_aperture_to_params(pypty_params, mean_pattern)`
:   Append a measured aperture to the reconstruction parameters.
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    mean_pattern : ndarray
        Aperture image to be rescaled and added.
    
    Returns
    -------
    dict
        Updated dictionary with aperture.

`append_exp_params(experimental_params, pypty_params=None)`
:   Attach experimental parameters to a PyPty preset dictionary and callibrate an extisting PyPty preset to new data. 
    
    Parameters
    ----------
    experimental_params : dict
        Dictionary containing experimental metadata and setup for PyPty reconstruction.
    pypty_params : dict or str or None, optional
        Existing PyPty preset to update, a filepath to a preset, or None to create a new one.
    
    Returns
    -------
    dict
        Updated PyPty parameter dictionary.
    
    Notes
    -------
    experimental_params should contain following entries:
        -data_path - path to a PyPty-style 3d .h5 file [N_measurements, ky,kx] or .npy Nion-style 4d-stem dataset (or 3d .npy dataset)
        -masks - 3d numpy array or None. if data is compressed provide the virtual detectors (masks) shape should be [N_masks,ky,kx]
        -output_folder - path to an outputfolder where the results will be stored
        -path_json - path to a nion-style json file with metadata (optional)
        -acc_voltage - float, accelerating voltage in kV
        
        One or multiple of the following callibrations:
            -rez_pixel_size_A - reciprocal pixel size in Å^-1
            -rez_pixel_size_mrad - reciprocal pixel size in mrad
            
            -conv_semiangle_mrad - beam convergence semi-angle in mrad
            -aperture - (optional)- binary 2D mask
            -bright_threshold - threshold to estimate an aperture, everything above threshold times maximum value in a pacbed will be concidered as bright field disk.
        -data_pad - int, reciprocal space padding. If None (default), pading is 1/4 of the total width of a diffraction pattern
        -upsample_pattern - int, default 1 (no upsampling)
        
        -aberrations - list or 1d numpy array containing beam aberrations (in Å). Aberrations are stored in Krivanek notation, e.g. C10, C12a, C12b, C21a, C21b, C23a, C23b, C30 etc
        -defocus - float, default 0. Extra probe defocus besides the one contained in aberrations.
        
        -scan_size - tuple of two ints, number of scan points along slow (y) and fast (x) axes. Optional. If no scan step or position grid is provided, it will be used to get the scan step
        -scan_step_A - float, scan step (STEM pixel size) in Å.
        -fov_nm - float, FOV along the fast axis in nm.
        -special_postions_A - 2d numpy array, default None. If you acquiered a data on a special non-rectangular grid, please specify the positions in Å via this array for all measurements in a following form: [y_0,x_0],[y_1,x_1],....[y_n,x_n]]
        -transform_axis_matrix- 2x2 matrix for postions transformation
        -PLRotation_deg - float, rotation angle between scan and detector axes. Default None. If None, a DPC measurement will be exectuted to get this angle. !!!!!!! Note that negative PLRotation_deg values rotate scan counter clockwise and diffraction space clockwise !!!!!!!!!!!
        -flip_ky - boolean Flag. Default is False. If no PyPty-style h5 data was created, this flag will flip the y-axis of diffraction patterns.
        
        -total_thickness - total thickness of a sample in Å. Has no effect if num_slices is 1 and propagation method (pypty_params entry) is multislice 
        -num_slices - integer, number of slices, default is 1.
        
        -plot - boolean Flag, default is True 
        -print_flag - integer. Default is 1. If 0 nothing will be printed. 1 prints only thelatest state of the computation, 2 prints every state as a separate line. 3 prints the linesearch progress in iterative optimization. 4 prints everything that 3 does and if constraints are applied, it prints how they contribute so that a user can configure the weights properly.
        -save_preprocessing_files - Boolean Flag. Default is True.

`conjugate_beam(pypty_params)`
:   Apply beam conjugation (flip defocus and aberrations).
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    
    Returns
    -------
    dict
        Updated dictionary with conjugated probe and CTF.

`create_aberrations_chunks(pypty_params, chop_size, n_abs)`
:   Create chunks, i.e. multiple subscans with independent beam aberrations. Usefull for large fields of view where the beam is varyying. If applied, the iterative reconstruction will have the same beam in each subscan, but apply a different CTF in each of these regions.
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    chop_size : int
        Size of each subscan region (in scan points).
    n_abs : int
        Number of aberration coefficients per region.
    
    Returns
    -------
    dict
        Updated parameter dictionary with aberration array and marker.

`create_probe_marker_chunks(pypty_params, chop_size)`
:   Creates chunks, i.e. multiple subscans with independent beam aberrations. Usefull for large fields of view where the beam is varyying. If applied, the iterative reconstruction will have the a differenet beam in each of these subscans.
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    chop_size : int
        Size of each subscan region (in scan points).
    
    Returns
    -------
    dict
        Updated dictionary with probe marker.

`create_pypty_data(data, path_output, swap_axes=False, flip_ky=False, flip_kx=False, flip_y=False, flip_x=False, comcalc_len=1000, comx=None, comy=None, bin=1, crop_left=None, crop_right=None, crop_top=None, crop_bottom=None, normalize=False, cutoff_ratio=None, pad_k=0, data_dtype=numpy.float32, rescale=1, exist_ok=True)`
:   Create a PyPty-style `.h5` dataset from 4D-STEM data.
    
    Parameters
    ----------
    data : str or ndarray
        Path to `.h5` or `.npy` data file or a 4D numpy array [scan_y, scan_x, ky, kx].
    path_output : str
        Output file path for the PyPty `.h5` dataset.
    swap_axes : bool, optional
        Swap the last two axes (kx, ky). Default is False.
    flip_ky, flip_kx, flip_y, flip_x : bool, optional
        Flip the data along specific axes. Default is False.
    comcalc_len : int, optional
        Number of patterns to use to estimate center-of-mass. Default is 1000.
    comx, comy : int or None, optional
        Predefined center-of-mass. If None, it will be computed.
    bin : int, optional
        Spatial binning factor on the diffraction patterns. Default is 1.
    crop_left, crop_right, crop_top, crop_bottom : int or None, optional
        Crop edges of patterns. Defaults are None.
    normalize : bool, optional
        Normalize pattern sums to 1. Default is False.
    cutoff_ratio : float or None, optional
        Mask out pixels farther than `cutoff_ratio × max_radius`. Default is None.
    pad_k : int, optional
        Padding to apply to diffraction patterns. Default is 0.
    data_dtype : dtype, optional
        Output data type. Default is np.float32.
    rescale : float, optional
        Scale factor for intensity. Default is 1.
    exist_ok : bool, optional
        If True, skip writing if file exists. Default is True.
    
    Returns
    -------
    None
    
    Notes
    -----
    Saves a `.h5` file containing processed 4D-STEM data with standardized formatting for PyPty.

`create_sequence_from_points(pypty_params, yf, xf, width_roi=20)`
:   Create scan subsequence around specified feature points.
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    yf : list of int
        Y-coordinates of feature points (in scan points).
    xf : list of int
        X-coordinates of feature points.
    width_roi : int, optional
        Width of the reconstruction window around each point.
    
    Returns
    -------
    list
        List of scan indices to reconstruct.

`create_sub_sequence(pypty_params, left, top, width, height, sub)`
:   Define a measurement subsequence for local reconstructions.
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    left : int
        Leftmost scan coordinate.
    top : int
        Top scan coordinate.
    width : int
        Width of subregion (in scan points).
    height : int
        Height of subregion (in scan points).
    sub : int
        Sampling factor (take every Nth point).
    
    Returns
    -------
    dict
        Updated parameter dictionary with `sequence` key.

`get_approx_beam_tilt(pypty_params, power=3, make_binary=False, percentile_filter_value=None, percentile_filter_size=10)`
:   Estimate scan-position-dependent beam tilt from PACBED. 
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    power : int or str
        Degree of polynomial fitting.
    make_binary : bool or float
        If True or float > 0, binarize patterns.
    percentile_filter_value : float or None
        Value for optional percentile filtering.
    percentile_filter_size : int
        Filter size if filtering is enabled.
    
    Returns
    -------
    dict
        Updated dictionary with estimated tilts.

`get_focussed_probe_from_vacscan(pypty_params, mean_pattern)`
:   Reconstruct a focused probe from a vacuum PACBED pattern.
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    mean_pattern : ndarray
        Measured PACBED from vacuum.
    
    Returns
    -------
    dict
        Updated dictionary with a probe estimate.

`get_grid_for_upsampled_image(pypty_params, image, image_pixel_size, left_zero_of_scan_grid=0, top_zero_of_scan_grid=0)`
:   Map coordinates of an upsampled image onto the reconstruction grid.
    
    This function calculates where pixel of an arbitary image (e.g. upsampled tcBF image) will land on a grid corresponding to a ptychographic reconstruction.
    
    
    Parameters
    ----------
    pypty_params : dict
        Dictionary of PyPty reconstruction parameters.
    image : ndarray
        2D image (e.g., upsampled tcBF) to map.
    image_pixel_size : float
        Pixel size of the image in Å.
    left_zero_of_scan_grid : int, optional
        Pixel offset on left side of image relative to scan grid. Default is 0.
    top_zero_of_scan_grid : int, optional
        Pixel offset on top side of image relative to scan grid. Default is 0.
    
    Returns
    -------
    sc : ndarray
        Array of pixel coordinates [[y, x], ...] in reconstruction grid units.

`get_offset(x_range, y_range, scan_step_A, detector_pixel_size_rezA, patternshape, rot_angle_deg=0)`
:   Compute pixel offsets between scan grid and reconstruction grid. In PyPty framework, scan grid is usually rotated to compensate the misalignment between scan- and detector- axes. Also, a reconstruction grid is larger than the scanned FOV, this is done to accomodate the extent of the probe. 
    
    Parameters
    ----------
    x_range, y_range : int
        Scan dimensions.
    scan_step_A : float
        STEM scan step size in Å.
    detector_pixel_size_rezA : float
        Reciprocal space pixel size in Å⁻¹.
    patternshape : tuple
        Shape of diffraction patterns.
    rot_angle_deg : float, optional
        Rotation between scan and detector axes (degrees).
    
    Returns
    -------
    offy, offx : float
        Offset values (in reconstruction pixels).

`get_positions_pixel_size(x_range, y_range, scan_step_A, detector_pixel_size_rezA, patternshape, rot_angle_deg=0, flip_x=False, flip_y=False, print_flag=False, transform_axis_matrix=array([[1., 0.],
       [0., 1.]]))`
:   Generate scan positions in reconstruction pixel units.
    
    Parameters
    ----------
    x_range, y_range : int
        Scan grid size.
    scan_step_A : float
        STEM scan step size (Å).
    detector_pixel_size_rezA : float
        Pixel size in reciprocal space (Å⁻¹).
    patternshape : tuple
        Shape of the diffraction pattern.
    rot_angle_deg : float, optional
        Scan-detector rotation angle in degrees. Default is 0.
    flip_x, flip_y : bool, optional
        Flip scan axes. Default is False.
    print_flag : bool, optional
        Print pixel size. Default is False.
    transform_axis_matrix : array_like
        Optional 2x2 matrix to apply to positions.
    
    Returns
    -------
    positions : ndarray
        Scan positions in reconstruction pixels.
    pixel_size : float
        Size of one reconstruction pixel in Å.

`get_ptycho_obj_from_scan(params, num_slices=None, array_phase=None, array_abs=None, scale_phase=1, scale_abs=1, scan_array_A=None, fill_value_type=None)`
:   Construct an initial object guess using interpolated phase and amplitude maps. You can use output of dpc, wdd of tcBF reconstructions to generate it.
    
    
    Parameters
    ----------
    params : dict
        PyPty parameter dictionary.
    num_slices : int or str, optional
        Number of slices or "auto" to estimate from max phase shift.
    array_phase : ndarray, optional
        2D phase map to interpolate.
    array_abs : ndarray, optional
        2D amplitude map to interpolate.
    scale_phase : float, optional
        Scale factor for phase.
    scale_abs : float, optional
        Scale factor for amplitude.
    scan_array_A : ndarray or None, optional
        Spatial reference grid for the input maps (in Å).
    fill_value_type : str or None, optional
        Padding strategy outside scanned region: None, "edge", or "median".
    
    Returns
    -------
    dict
        Updated PyPty parameter dictionary with object guess.

`rotate_scan_grid(pypty_params, angle_deg)`
:   Apply a rigid rotation to the scan grid.
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    angle_deg : float
        Rotation angle in degrees.
    
    Returns
    -------
    dict
        Updated dictionary with rotated positions and angle.

`tiltbeamtodata(pypty_params, align_type='com')`
:   Align the probe momentum to the center of the measured PACBED pattern.
    
    Parameters
    ----------
    pypty_params : dict
        PyPty parameter dictionary.
    align_type : str, optional
        Type of alignment ("com" or "cross_corr").
    
    Returns
    -------
    dict
        Updated dictionary with shifted probe.