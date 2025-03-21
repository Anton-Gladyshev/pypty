Module pypty.signal_extraction
==============================

Functions
---------

`compensate_pattern_drift(aperture, patterns)`
:   Compensate for drift in diffraction patterns via phase correlation.
    
    Parameters
    ----------
    aperture : ndarray
        The binary aperture mask used for phase correlation.
    patterns : ndarray
        The diffraction patterns to be compensated for drift.
    
    Returns
    -------
    patterns : ndarray
        The compensated diffraction patterns.

`create_binned_dataset(path_orig, path_new, bin)`
:   Downsample a dataset by spatial binning and save it to a new file.
    
    Parameters
    ----------
    path_orig : str
        The file path of the original dataset.
    path_new : str
        The file path to save the binned dataset.
    bin : int
        The binning factor to downsample the dataset.

`get_aperture(params)`
:   Generate a binary aperture mask based on the mean diffraction pattern.
    
    Parameters
    ----------
    params : dict
        Dictionary containing parameters including data path, data padding, plotting option, and bright threshold.
    
    Returns
    -------
    params : dict
        Updated parameters dictionary containing the generated aperture mask.

`get_virtual_annular_detector(pypty_params, inner_rad=0, outer_rad=1, save=False, offset_x=0, offset_y=0)`
:   Create virtual detector signals from annular masks.
    
    Parameters
    ----------
    pypty_params : dict
        Dictionary containing parameters including data path, scan size, plotting option, and output folder.
    inner_rad : float, optional
        Inner radius of the annular mask. Default is 0.
    outer_rad : float, optional
        Outer radius of the annular mask. Default is 1.
    save : bool, optional
        Whether to save the resulting virtual detector signal as a .npy file. Default is False.
    offset_x : float, optional
        X-offset for the annular mask. Default is 0.
    offset_y : float, optional
        Y-offset for the annular mask. Default is 0.
    
    Returns
    -------
    signal : ndarray
        The computed virtual detector signal.

`getvirtualhaadf(pypty_params, save=True)`
:   Compute a virtual HAADF image from a 4D-STEM dataset.
    
    Parameters
    ----------
    pypty_params : dict
        Dictionary containing parameters including data path, scan size, plotting option, and output folder.
    save : bool, optional
        Whether to save the resulting HAADF image as a .npy file. Default is True.
    
    Returns
    -------
    haadf : ndarray
        The computed virtual HAADF image.