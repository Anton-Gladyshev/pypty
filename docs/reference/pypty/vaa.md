Module pypty.vaa
================

Functions
---------

`add_scalebar_ax(ax, x, y, width, height, x_t, y_t, px_size, unit)`
:   Add a scale bar to a given axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to which the scale bar will be added.
    x : float
        The x-coordinate of the bottom left corner of the scale bar.
    y : float
        The y-coordinate of the bottom left corner of the scale bar.
    width : float
        The width of the scale bar in pixels.
    height : float
        The height of the scale bar in pixels.
    x_t : float
        The x-coordinate for the text label.
    y_t : float
        The y-coordinate for the text label.
    px_size : float
        The pixel size in the same units as the width and height.
    unit : str
        The unit of measurement for the scale bar.

`complex_array_to_rgb(X, theme='dark', rmax=None)`
:   Convert a complex array to RGB format.
    
    Parameters
    ----------
    X : numpy.ndarray
        The input array of complex numbers.
    theme : str, optional
        The color theme, either 'dark' or 'light' (default is 'dark').
    rmax : float, optional
        Maximum absolute value for normalization (default is None).
    
    Returns
    -------
    numpy.ndarray
        The RGB representation of the input array.

`complex_pca(data, n_components)`
:   Perform PCA on complex data.
    
    Parameters
    ----------
    data : numpy.ndarray
        The input data array of shape (N_y, N_x, N_obs).
    n_components : int
        The number of principal components to retain.
    
    Returns
    -------
    numpy.ndarray
        The reduced data array of shape (N_y, N_x, n_components).

`fit_aberrations_to_wave(wave, px_size_A, acc_voltage, thresh=0, aberrations_guess=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], plot=True, ftol=1e-20, xtol=1e-20, loss='linear', max_mrad=inf)`
:   Calculate model positions based on step size and angle.
    
    Parameters
    ----------
    step_size : float
        The step size for the model.
    angle_rad : float
        The angle in radians.
    x : numpy.ndarray
        The x coordinates.
    y : numpy.ndarray
        The y coordinates.
    
    Returns
    -------
    tuple
        The model x and y coordinates.

`get_affine_tranform(positions, scan_size, px_size_A)`
:   Calculate the affine transformation matrix from positions.
    
    Parameters
    ----------
    positions : numpy.ndarray
        The measured positions.
    scan_size : tuple
        The size of the scan grid.
    px_size_A : float
        Pixel size in angstroms.
    
    Returns
    -------
    numpy.ndarray
        The deformation matrix.

`get_step_angle_scan_grid(positions, scan_size)`
:   Determine the step size and angle for a scan grid.
    
    Parameters
    ----------
    positions : numpy.ndarray
        The measured positions.
    scan_size : tuple
        The size of the scan grid.
    
    Returns
    -------
    tuple
        The step size and angle in degrees.

`mesh_model_positions(step_size, angle_rad, x, y)`
:   Calculate model positions based on step size and angle.
    
    Parameters
    ----------
    step_size : float
        The step size for the model.
    angle_rad : float
        The angle in radians.
    x : numpy.ndarray
        The x coordinates.
    y : numpy.ndarray
        The y coordinates.
    
    Returns
    -------
    tuple
        The model x and y coordinates.

`mesh_objective_positions(ini_guess, x, y, mesh_x, mesh_y)`
:   Objective function for mesh optimization.
    
    Parameters
    ----------
    ini_guess : list
        Initial guess for the optimization.
    x : numpy.ndarray
        The x coordinates.
    y : numpy.ndarray
        The y coordinates.
    mesh_x : numpy.ndarray
        The mesh x coordinates.
    mesh_y : numpy.ndarray
        The mesh y coordinates.
    
    Returns
    -------
    float
        The sum of squared differences.

`outputlog_plots(loss_path, skip_first=0, plot_time=True)`
:   Plot log file data from PyPty.
    
    Parameters
    ----------
    loss_path : str
        Path to the PyPty CSV file.
    skip_first : int, optional
        Number of initial iterations to skip (default is 0).
    plot_time : bool, optional
        If True, a second x-axis showing time in seconds will be added on top of the plot.
    
    Returns
    -------
    list
        List of plotted figures.

`plot_complex_modes(p, nm, sub)`
:   Plot complex modes in RGB format.
    
    Parameters
    ----------
    p : numpy.ndarray
        The input array of complex modes.
    nm : int
        The number of modes to plot.
    sub : int
        The number of rows for the subplot layout.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plotted complex modes.

`plot_modes(ttt)`
:   Plot the modes of a tensor.
    
    Parameters
    ----------
    ttt : numpy.ndarray
        A 3D or 4D array containing the modes to be plotted.

`radial_average(ff, r_bins, r_max, r_min, px_size_A, plot=True)`
:   Calculate the radial average of a 2D array.
    
    Parameters
    ----------
    ff : numpy.ndarray
        The input 2D array.
    r_bins : float
        The bin size for the radial average.
    r_max : float
        The maximum radius for averaging.
    r_min : float
        The minimum radius for averaging.
    px_size_A : float
        Pixel size in angstroms.
    plot : bool, optional
        If True, the radial average will be plotted (default is True).
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot if `plot` is True.