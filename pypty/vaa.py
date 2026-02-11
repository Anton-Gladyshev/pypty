import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects
from scipy import ndimage

try:
    from skimage.restoration import unwrap_phase
except:
    pass
from matplotlib import patches

from pypty import utils as pyptyutils
from tqdm import tqdm
import matplotlib
import csv

from pypty import fft as pyptyfft
from pypty import direct as pyptydirect




def plot_modes(ttt):
    """Plot probe/object modes in real and reciprocal space.

    Shows four panels per mode: real-space magnitude/phase and Fourier-space
    magnitude/phase.

    Parameters
    ----------
    ttt : numpy.ndarray
        Complex mode stack. Accepted shapes are ``(Ny, Nx, nmodes)`` or
        ``(Ny, Nx, nstates, nmodes)``.

    Returns
    -------
    None
        The function displays figures using Matplotlib.
    """
    if len(ttt.shape)==4:
        for i in range(ttt.shape[-1]):
            for j in range(ttt.shape[-2]):
                fig, ax=plt.subplots(1,4, figsize=(10,40))
                im0=ax[0].imshow(np.abs(ttt[:,:,j,i]), cmap="gray", vmax=np.max(np.abs(ttt)))
                im1=ax[1].imshow(np.angle(ttt[:,:,j,i]), cmap="gray")
                im2=ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ttt[:,:,j,i])))), cmap="gray")
                im3=ax[3].imshow(np.angle(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ttt[:,:,j,i])))), cmap="gray")
                fig.colorbar(im0, ax=ax[0], pad=0, fraction=0.047, location='bottom')
                fig.colorbar(im1, ax=ax[1], pad=0, fraction=0.047, location='bottom')
                fig.colorbar(im2, ax=ax[2], pad=0, fraction=0.047, location='bottom')
                fig.colorbar(im3, ax=ax[3], pad=0, fraction=0.047, location='bottom')
                ax[0].axis("off")
                ax[0].set_title("R-Space mag")
                ax[1].axis("off")
                ax[1].set_title("R-Space phase")
                ax[2].axis("off")
                ax[2].set_title("Q-Space mag")
                ax[3].axis("off")
                ax[3].set_title("Q-Space phase")
                plt.tight_layout()
                plt.show()
    else:
        for i in range(ttt.shape[-1]):
            fig, ax=plt.subplots(1,4, figsize=(10,40))
            im0=ax[0].imshow(np.abs(ttt[:,:,i]), cmap="gray", vmax=np.max(np.abs(ttt)))
            im1=ax[1].imshow(np.angle(ttt[:,:,i]), cmap="gray")
            im2=ax[2].imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ttt[:,:,i])))), cmap="gray")
            im3=ax[3].imshow(np.angle(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ttt[:,:,i])))), cmap="gray")
            fig.colorbar(im0, ax=ax[0], pad=0, fraction=0.047, location='bottom')
            fig.colorbar(im1, ax=ax[1], pad=0, fraction=0.047, location='bottom')
            fig.colorbar(im2, ax=ax[2], pad=0, fraction=0.047, location='bottom')
            fig.colorbar(im3, ax=ax[3], pad=0, fraction=0.047, location='bottom')
            ax[0].axis("off")
            ax[0].set_title("R-Space mag")
            ax[1].axis("off")
            ax[1].set_title("R-Space phase")
            ax[2].axis("off")
            ax[2].set_title("Q-Space mag")
            ax[3].axis("off")
            ax[3].set_title("Q-Space phase")
            plt.tight_layout()
            plt.show()



def fit_aberrations_to_wave(wave, px_size_A, acc_voltage, thresh=0,
                            aberrations_guess=[0,0,0,0,0,0,0,0,0,0,0,0],
                            plot=True, ftol=1e-20, xtol=1e-20, loss="linear", max_mrad=np.inf):
    """Fit aberration coefficients to a complex wave by matching Fourier phase.

    The function computes the Fourier transform of ``wave``, unwraps its phase
    (optionally masked by magnitude threshold and angular cutoff), and fits a
    Krivanek-style aberration expansion using nonlinear least squares.

    Parameters
    ----------
    wave : numpy.ndarray
        Complex real-space wave, shape ``(Ny, Nx)``.
    px_size_A : float
        Real-space pixel size in Å.
    acc_voltage : float
        Acceleration voltage in kV.
    thresh : float, optional
        Relative magnitude threshold (fraction of max) used to mask Fourier pixels
        before fitting.
    aberrations_guess : array_like, optional
        Initial guess for aberration coefficients in Å. Length sets the number of
        fitted aberration terms.
    plot : bool, optional
        If True, show diagnostic plots (fitted phase, target phase, difference).
    ftol : float, optional
        Relative tolerance for termination by the change of the cost function.
    xtol : float, optional
        Relative tolerance for termination by the change of the parameters.
    loss : str, optional
        Loss function for `scipy.optimize.least_squares`.
    max_mrad : float, optional
        Maximum scattering angle included in the fit, in mrad.

    Returns
    -------
    aberrations : numpy.ndarray
        Fitted aberration coefficients in Å.

    Notes
    -----
    This routine assumes the Fourier phase of the wave is dominated by probe
    aberrations after removing the center-of-mass (COM) tilt term.
    """
    wave=wave.copy()
    x=np.arange(wave.shape[0])-wave.shape[0]//2
    x,y=np.meshgrid(x,x)
    kx=np.fft.fftshift(np.fft.fftfreq(wave.shape[0]))
    kx,ky=np.meshgrid(kx,kx, indexing="xy")
    comx, comy=np.average(x, weights=np.abs(wave)**2), np.average(y, weights=np.abs(wave)**2)
    fourier_wave=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave)))
    mag=np.abs(fourier_wave)
    mag=mag>=thresh*np.max(mag)
    phase=-1*np.angle(fourier_wave)-2*np.pi*(kx*comx+ky*comy)
    phase*=mag
    phase= unwrap_phase(phase, wrap_around=(False, False))
    wavelength=12.4/np.sqrt(acc_voltage*(acc_voltage+2*511))
    kx,ky=kx*wavelength/px_size_A, ky*wavelength/px_size_A
    kr=(kx**2+ky**2)**0.5
    mag*=kr<=(max_mrad*1e-3)
    phase-=phase[kr==0]
    phase*=mag
    
    ctf_matrix=pyptyutils.get_ctf_matrix(kx, ky, len(aberrations_guess), wavelength, np)[:, mag]
    ctf_matrix=np.swapaxes(ctf_matrix, 0,1)
    def jac_ctf_fit(x):
        nonlocal ctf_matrix
        return ctf_matrix
    phase_crop=phase[mag]
    def objective(aberrations):
        nonlocal phase_crop, ctf_matrix
        ctf=np.sum(aberrations[None,:]*ctf_matrix, axis=1)
        return ctf-phase_crop
    result=least_squares(objective,aberrations_guess, jac=jac_ctf_fit, ftol=ftol, loss=loss, xtol=xtol)
    aberrations=result["x"]
    num_abs=len(aberrations)
    possible_n, possible_m, possible_ab=pyptyutils.convert_num_to_nmab(num_abs)
    aber_print, s=pyptyutils.nmab_to_strings(possible_n, possible_m, possible_ab), ""
    for i in range(len(aberrations)): s+=aber_print[i]+": %.2e Å; "%aberrations[i];
    sys.stdout.write("\nFitted aberrations: %s"%s[:-1])
    fitted_ctf=pyptyutils.get_ctf(aberrations, kx, ky, wavelength, angle_offset=0)*mag
    if plot:
        fig, ax=plt.subplots(1,3, figsize=(9,3))
        im0=ax[0].imshow(fitted_ctf)
        fig.colorbar(im0, ax=ax[0], orientation="horizontal", fraction=0.0475, pad=0)
        ax[0].set_title("Fitted phase")
        ax[0].axis("off")
        im1=ax[1].imshow(phase)
        fig.colorbar(im1, ax=ax[1], orientation="horizontal", fraction=0.0475, pad=0)
        ax[1].set_title("Target phase")
        ax[1].axis("off")

        im2=ax[2].imshow(phase-fitted_ctf)
        fig.colorbar(im2, ax=ax[2], orientation="horizontal", fraction=0.0475, pad=0)
        ax[2].set_title("Difference")
        ax[2].axis("off")
        plt.show()
    return aberrations
    
    
    
def mesh_model_positions(step_size, angle_rad, x, y):
    """Compute an ideal rotated scan grid from step size and angle.

    Parameters
    ----------
    step_size : float
        Step size in the same units as ``x``/``y``.
    angle_rad : float
        Rotation angle in radians.
    x, y : numpy.ndarray
        Flattened coordinate arrays describing the nominal scan grid.

    Returns
    -------
    x_model, y_model : numpy.ndarray
        Modeled coordinates after applying the rotation and step size.
    """
    x_model = step_size * np.cos(angle_rad) * x - step_size * np.sin(angle_rad) * y
    y_model = step_size * np.sin(angle_rad) * x + step_size * np.cos(angle_rad) * y
    return x_model, y_model
    
def mesh_objective_positions(ini_guess, x, y, mesh_x, mesh_y):
    """Objective for fitting a rotated/stepped grid to measured positions.

    Parameters
    ----------
    ini_guess : array_like
        Two parameters ``[step, angle]`` where angle is in radians.
    x, y : numpy.ndarray
        Flattened nominal grid coordinates.
    mesh_x, mesh_y : numpy.ndarray
        Flattened measured coordinates to be matched.

    Returns
    -------
    float
        Sum of squared residuals between modeled and measured coordinates.
    """
    step, angle=ini_guess
    x_model, y_model = mesh_model_positions(step, angle, x, y)
    return np.sum((x_model - mesh_x)**2 + (y_model - mesh_y)**2)

def get_step_angle_scan_grid(positions, scan_size, print_flag=1):
    """Estimate mean scan step and rotation angle from measured positions.

    A simple global fit is performed assuming a rectangular scan grid that is
    uniformly rotated and scaled.

    Parameters
    ----------
    positions : numpy.ndarray
        Measured positions with shape ``(N, 2)`` ordered as ``(y, x)``.
    scan_size : tuple of int
        Scan grid size ``(Ny, Nx)``.
    print_flag : bool, optional
        If True, prints the standard deviation of the residuals in pixels.

    Returns
    -------
    step : float
        Estimated step size (same units as `positions`).
    angle_deg : float
        Estimated rotation angle in degrees.
    """
    pos=positions.copy()
    posy, posx=pos[:,0], pos[:,1]
    x, y=np.meshgrid(np.arange(scan_size[1]),np.arange(scan_size[0]))
    x= x.flatten()-np.mean(x)
    y= y.flatten()-np.mean(y)
    posy-=np.mean(posy)
    posx-=np.mean(posx)
    result = minimize(mesh_objective_positions,  x0=[10,0],
                      args=(x, y, posx, posy), method="Powell",
                      bounds=[(0.001,10000),(-np.pi, np.pi)],
                      tol=1e-10, options={"maxiter":1000})
    angle = (result["x"][1])*180/np.pi
    step= (result["x"][0])
    final_mesh_x, final_mesh_y=mesh_model_positions(step, angle*np.pi/180, x, y)
    difference=np.stack((posy-final_mesh_y,posx-final_mesh_x))
    if print_flag:
        print("std: ", np.std(difference), " px.")
    return step, angle
    
    
def get_affine_tranform(positions,  scan_size, px_size_A, print_flag=1):
    """Estimate an affine deformation matrix from measured scan positions.

    The mapping is fit from ideal integer grid coordinates to measured positions
    using linear least squares. The returned matrix is scaled by ``px_size_A``.

    Parameters
    ----------
    positions : numpy.ndarray
        Measured positions with shape ``(N, 2)`` ordered as ``(y, x)``.
    scan_size : tuple of int
        Scan grid size ``(Ny, Nx)``.
    px_size_A : float
        Pixel size used to scale the affine transform, in Å.
    print_flag : bool, optional
        If True, prints deformation and shift parameters.

    Returns
    -------
    deformation : numpy.ndarray
        2×2 deformation matrix mapping ideal grid vectors to measured coordinates,
        scaled by ``px_size_A``.
    """
    x_perfect, y_perfect=np.meshgrid(np.arange(scan_size[1]), np.arange(scan_size[0]))
    x_perfect, y_perfect, off_perfect= x_perfect.flatten(), y_perfect.flatten(), np.ones(scan_size[1]*scan_size[0])
    yxo_perf=np.swapaxes(np.array([y_perfect, x_perfect, off_perfect]), 0,1)
    matrix=positions.T @ yxo_perf @ np.linalg.inv(yxo_perf.T @ yxo_perf)
    matrix=np.array(matrix)
    matrix*=px_size_A
    deformation=matrix[:, :2]
    if print_flag:
        print("Deformation matrix yy: %.2f, yx: %.2f , xy: %.2f, xx: %.2f"%(matrix[0,0], matrix[0,1],matrix[1,0], matrix[1,1]))
        print("Shift y: %.2f A, x: %.2f A"%(matrix[0,2], matrix[1, 2]))
    return deformation

        
def add_scalebar_ax(ax, x, y, width, height, x_t, y_t, px_size, unit, fontsize=20):
    """Add a scale bar annotation to an image axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    x, y : float
        Bottom-left corner of the bar in data coordinates.
    width : float
        Scale bar length in ``unit``.
    height : float
        Bar thickness in data pixels.
    x_t, y_t : float
        Text anchor position in data coordinates.
    px_size : float
        Pixel size in the same physical units as ``width``.
    unit : str
        Unit label (e.g. ``'Å'``, ``'nm'``).
    fontsize : int, optional
        Font size for the label.

    Returns
    -------
    None
        The function modifies ``ax`` in place.
    """
    rect=Rectangle([x,y], width/px_size, height, color="white", alpha=0.9)
    text2=ax.text(x_t, y_t, str(width)+" "+unit, c="w", fontsize=fontsize)
    text2.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    ax.add_patch(rect)
    rect.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])



def outputlog_plots(loss_path, skip_first=0, plot_time=True):
    """Plot PyPty optimization log columns from a CSV file.

    Parameters
    ----------
    loss_path : str
        Path to the CSV log file.
    skip_first : int, optional
        If nonzero, ignore iterations with epoch < ``skip_first``.
    plot_time : bool, optional
        If True, adds a secondary x-axis showing elapsed time (hours).

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        One figure per plotted column.
    """
    dat=[]
    with open(loss_path, 'r') as file:
        data = csv.reader(file, delimiter = ',')
        for d in data:
            dat.append(d)
    dat=(np.array(dat)[1:, :-1]).astype(float)
    epoch=dat[:,0]
    
    
    if skip_first!=0:
        where=epoch>=skip_first
        dat=dat[where]
    
    epoch=dat[:,0]
    time=dat [:,1] / 3600
    if dat.shape[1]==12:
        fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations",
                "dir. derivative", "new dir. derivative", "Constraints contribution", "Free GiB", "Total GiB", "Warnings"]
    else:
        fieldnames=["epoch", "time / s", "loss", "sse", "initial step", "matching step", "N linesearch iterations", "dir. derivative", "new dir. derivative", "F-axis postions reg.", "Deformation positons reg.", "Deformation tilts reg.", "F-axis tilts reg.", "l1 object reg. (phase)","l1 object reg. (abs)", "Q-space probe reg.", "R-space probe reg.", "TV object reg.", "V-object reg.", "MW-object reg.",  "S-axis postions reg", "S-axis tilts reg", "Probe-Modulation", "Free GiB", "Total GiB", "Warnings"]

    def forward(x):
        return np.interp(x,epoch,time)
    def inverse(x):
        return np.interp(x,time,epoch)
    figs=[]
    for datai in range(2, dat.shape[1], 1):
        fig,ax = plt.subplots(figsize=(10,4), dpi=300)
        ax.plot(epoch, dat[:, datai], ".-",linewidth=2, alpha=0.7)#, label=tit[iii])
        ax.set_xlabel("Iteration", fontsize = 14)
        ax.set_ylabel(fieldnames[datai], fontsize = 14)
        if plot_time:
            ax2 = ax.secondary_xaxis("top", functions=(forward,inverse)) # Create a dummy plot
            ax2.set_xlabel("time / h", fontsize = 14)
            plt.setp(ax2.get_xticklabels()[0], visible=False)
        figs.append(fig)
        plt.show()
    return figs


def radial_average(ff, r_bins, r_max, r_min, px_size_A, plot=True):
    """Compute radial average of a 2D Fourier-domain quantity.

    Parameters
    ----------
    ff : numpy.ndarray
        2D array to be radially averaged.
    r_bins : float
        Bin width in pixels (in Fourier grid radius units).
    r_max, r_min : float
        Maximum and minimum normalized radii (fractions of the max radius).
    px_size_A : float
        Real-space pixel size in Å (used to label spatial frequency axis).
    plot : bool, optional
        If True, plots the radial average.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle if ``plot=True``.
    """
    x_grid,y_grid=np.fft.fftshift(np.fft.fftfreq(ff.shape[1])), np.fft.fftshift(np.fft.fftfreq(ff.shape[0]))
    x_grid,y_grid=np.meshgrid(x_grid, y_grid, indexing="xy")
    mult=np.min([ff.shape[0], ff.shape[1]])
    r_max*=mult
    r_min*=mult
    r=mult*(x_grid**2+y_grid**2)**0.5
    unique_distances=np.arange(0, np.max(r), r_bins)
    unique_distances=unique_distances[((unique_distances<=r_max)*(unique_distances>=r_min)).astype(bool)]
    radial_avg=np.zeros_like(unique_distances)
    if plot:
        for iii in tqdm(range(len(unique_distances))):
            distance=unique_distances[iii]
            radial_mask = (r<=distance+r_bins)*(r>distance)
            radial_avg[iii] = np.mean(ff[radial_mask])
        fig=plt.figure(figsize=(10,3))
        plt.plot(unique_distances/(mult*px_size_A), radial_avg, "-")
        plt.xlabel("spatial freqency [A$^{-1}$]")
        plt.yscale("log")
        plt.show()
    return fig


def complex_pca(data, n_components):
    """Perform PCA on complex observations.

    The last axis is treated as the observation axis. The covariance is computed
    on mean-centered complex data and eigen-decomposed.

    Parameters
    ----------
    data : numpy.ndarray
        Complex array with shape ``(Ny, Nx, N_obs)``.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    data_reduced : numpy.ndarray
        Complex array with shape ``(Ny, Nx, n_components)``.
    """
    N_y, N_x, N_obs = data.shape
    reshaped_data = data.reshape(-1, N_obs)
    mean = np.mean(reshaped_data, axis=0)
    centered_data = reshaped_data - mean
    covariance_matrix = np.cov(centered_data, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    principal_components = eigenvectors[:, :n_components]
    reduced_data = centered_data @ principal_components
    data_reduced = reduced_data.reshape(N_y, N_x, n_components)
    return data_reduced


def complex_array_to_rgb(X, theme='dark', rmax=None):
    """Map a complex array to an RGB image using phase/magnitude encoding.

    Hue encodes the complex phase. Magnitude is mapped either to value (dark theme)
    or saturation (light theme).

    Parameters
    ----------
    X : numpy.ndarray
        Complex array.
    theme : {'dark', 'light'}, optional
        Rendering theme.
    rmax : float, optional
        Magnitude used for normalization. If None, uses ``abs(X).max()``.

    Returns
    -------
    rgb : numpy.ndarray
        RGB image array with shape ``X.shape + (3,)`` and dtype float.
    """
    absmax = rmax or np.abs(X).max()
    Y = np.zeros(X.shape + (3,), dtype='float')
    Y[..., 0] = np.angle(X) / (2 * np.pi) % 1
    if theme == 'light':
        Y[..., 1] = np.clip(np.abs(X) / absmax, 0, 1)
    elif theme == 'dark':
        Y[..., 1] = 1
        Y[..., 2] = np.clip(np.abs(X) / absmax, 0, 1)
    Y = matplotlib.colors.hsv_to_rgb(Y)
    return Y


def plot_complex_modes(p, nm, sub):
    """Plot a subset of complex modes in RGB with intensity percentages.

    Parameters
    ----------
    p : numpy.ndarray
        Complex mode stack with shape ``(Ny, Nx, nmodes)``.
    nm : int
        Number of modes to plot.
    sub : int
        Number of subplot rows.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    """
    p2=np.abs(p)**2
    pint=np.sum(np.abs(p)**2, (0,1))
    pint=100*pint/np.sum(pint)
    sort=np.argsort(pint)[::-1]
    p=p[:,:,sort]
    pint=pint[sort]
    fig, axes=plt.subplots(sub,nm//sub, figsize=(4*nm//sub,4*sub))
    try:
        axes=axes.flatten()
    except:
        axes=[axes]
    for i, ax in enumerate(axes):
        ax.imshow(complex_array_to_rgb(p[::-1,::-1,i], theme='dark', rmax=np.max(np.abs(p))))
        ax.axis("off")
        ax.text(15,0.9*p.shape[0], "%.1f %%"%(pint[i]), fontsize=15)
    plt.show()
    return fig




def plot_complex_colorwheel(ax, theme='dark', N=512, rmax=1.0, fontsize=15, background=0):
    """
    Complex-plane color wheel legend on an axis.
    This helper renders the HSV mapping used by :func:`complex_array_to_rgb` by drawing
    a filled disk of radius ``rmax`` in the complex plane. Hue encodes the complex phase
    (argument), while value/saturation encodes magnitude depending on ``theme``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into.
    theme : {'dark', 'light'}, optional
        Color theme that matches :func:`complex_array_to_rgb`. For ``'dark'``, magnitude
        is mapped to value (brightness) and saturation is fixed. For ``'light'``,
        magnitude is mapped to saturation.
    N : int, optional
        Grid size used to sample the complex plane (image will be ``N x N``).
    rmax : float, optional
        Maximum radius shown in the colorwheel. Values outside the disk are set to a
        uniform background.
    fontsize : int, optional
        Font size for the annotations ("0", ``rmax``, and ``2pi``).
    background : bool, optional
        If truthy, sets the outside-of-disk background to white; otherwise uses a
        theme-appropriate background (black for ``'dark'``, white for ``'light'``).

    Returns
    -------
    None
        The function draws directly onto ``ax``.
    """

    # Grid in the complex plane
    x = np.linspace(-rmax, rmax, N)
    y = np.linspace(-rmax, rmax, N)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    R = np.abs(Z)
    mask = R <= rmax
    Z_masked = Z.copy()
    Z_masked[~mask] = 0
    img = complex_array_to_rgb(Z_masked, theme=theme, rmax=rmax)
    print(img.shape)
    if background:
        img[R >=rmax,:]=1
    if theme == 'dark':
        bg = np.array([0, 0, 0])
        arrow_color = 'w'
    else:
        bg = np.array([1, 1, 1])
        arrow_color = 'k'
    img[~mask] = bg

    ax.imshow(img, extent=(-rmax, rmax, -rmax, rmax), origin='lower')
    ax.set_aspect('equal')
    ax.annotate(
        "",
        xy=(rmax * 0.9, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="<->", color=arrow_color, lw=1.5),
        zorder=10)
    # Label at origin
    text=ax.text(
        0, 0, "0",
        color=arrow_color,
        ha='right', va='top',
        fontsize=fontsize,
        zorder=11)
    text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    # Label at rmax
    text=ax.text(
        rmax * 0.85, 0,
        r"$%.2f$"%rmax,
        color=arrow_color,
        ha='center', va='bottom',
        fontsize=fontsize,
        zorder=11)
    text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    # -------------------------------
    # Add azimuthal arrow (2π)
    # -------------------------------
    r_arc = 0.8 * rmax

    # Arc (almost full circle)
    arc = patches.Arc(
        (0, 0),
        2 * r_arc, 2 * r_arc,
        angle=0,
        theta1=0,
        theta2=330,  # leave a small gap
        color=arrow_color,
        lw=1.5,
        zorder=10,
    )
    ax.add_patch(arc)

    # Little arrow segment at the end of the arc
    theta_start = np.deg2rad(320)
    theta_end = np.deg2rad(330)
    x_start, y_start = r_arc * np.cos(theta_start), r_arc * np.sin(theta_start)
    x_end, y_end = r_arc * np.cos(theta_end), r_arc * np.sin(theta_end)

    ax.annotate(
        "",
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
        zorder=11,
    )

    # Label "2π" near the arc
    theta_label = np.deg2rad(150)
    x_label = 1.1 * r_arc * np.cos(theta_label)
    y_label = 1.1 * r_arc * np.sin(theta_label)

    text= ax.text(
        x_label, y_label,
        r"$2\pi$",
        color="white",
        ha='center', va='center',
        fontsize=fontsize,
        zorder=11,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])

    ax.axis("off")




def crop_ptycho_object_to_stem_roi(o,params):
    """
    Crop a reconstructed ptychography object to the STEM scan region of interest.

    The function rotates the object by `params["PLRotation_deg"]` (in degrees) without
    reshaping and then crops the rotated object to the field of view implied by the
    scan parameters.

    Parameters
    ----------
    o : numpy.ndarray
        2D object array with axes (y, x). Can be real or complex.
    params : dict
        Parameter dictionary. Must contain:

        - ``PLRotation_deg`` : float
        - ``scan_size`` : tuple of int
        - ``scan_step_A`` : float
        - ``pixel_size_x_A`` : float
        - ``pixel_size_y_A`` : float

    Returns
    -------
    numpy.ndarray
        Cropped object array with axes (y, x).

    Notes
    -----
    The crop size is computed in physical units (Å) from `scan_size*scan_step_A` and
    then converted to pixels using the object pixel sizes.
    """
    pl_rot=params["PLRotation_deg"]
    scan_size=params["scan_size"]
    scan_step_A=params["scan_step_A"]
    pixel_size_x_A=params["pixel_size_x_A"]
    o=ndimage.rotate(o, 1*pl_rot, reshape=False, axes=(1,0), cval=1.0)
    fov_y, fov_x=scan_size[0]*scan_step_A, scan_size[1]*scan_step_A
    
    crop_y=(o.shape[0]*pixel_size_x_A-fov_y)/(2*pixel_size_x_A)
    crop_x=(o.shape[1]*pixel_size_x_A-fov_x)/(2*pixel_size_x_A)
    o=o[int(np.round(crop_y)): int(np.round(-crop_y)), int(np.round(crop_x)): int(np.round(-crop_x))]
    return o
