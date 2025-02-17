import os.path
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates, rotate

###################### Begin of the defintions  ######################
def fit_vector_field(x, y, u, v, Xi,Yi,method='nearest'):
    x,y,u,v=x.flatten(),y.flatten(),u.flatten(),v.flatten()
    Ui = -griddata((x, y), u, (Xi, Yi), method=method, fill_value=0)
    Vi = -griddata((x, y), v, (Xi, Yi), method=method, fill_value=0)
    return Vi, Ui
def warp_image(image, vector_field, order=1, extra_angle_rotation_compensation=0):
    """
    Warp an image based on a given vector field.
    Parameters:
    - image: 2D or 3D NumPy array representing the input image.
    - vector_field: Tuple of 2D NumPy arrays (U, V) representing the vector field.

    Returns:
    - Warped image: NumPy array containing the warped image.
    """
    U, V = vector_field
    rows, cols = np.indices(image.shape[:2])
    rows_flatten, cols_flatten = rows.flatten(), cols.flatten()
    warped_rows = rows_flatten + U.flatten()
    warped_cols = cols_flatten + V.flatten()
    angle_rad=extra_angle_rotation_compensation*np.pi/180
    warped_image = map_coordinates(image, (warped_rows, warped_cols), order=order, mode='constant')
    warped_image = warped_image.reshape(image.shape)
    if extra_angle_rotation_compensation!=0:
        warped_image=rotate(warped_image, extra_angle_rotation_compensation, reshape=False,  order=order)

    return warped_image
def get_tilt_coordinates(tilt_grid_rotation,flip_y, tilt_series,tilt_FOV, detector_px_size, thresh):
    if flip_y:
        tilt_series=tilt_series[:,:,::-1,:]
    rotation=tilt_grid_rotation*np.pi/180
    tilt_step=tilt_FOV/(tilt_series.shape[1]-1)
    x_range=tilt_series.shape[3]
    y_range=tilt_series.shape[2]
    x_range_tilt=tilt_series.shape[1]
    y_range_tilt=tilt_series.shape[0]
    x_detector, y_detector=np.meshgrid(np.arange(-x_range//2,x_range-x_range//2,1), np.arange(-y_range//2,y_range-y_range//2,1), indexing="xy")
    x_detector_full=np.copy(x_detector)
    y_detector_full=np.copy(y_detector)
    x_tilt, y_tilt=np.meshgrid(np.arange(-x_range_tilt//2,x_range_tilt-x_range_tilt//2,1)*tilt_FOV/(detector_px_size*x_range_tilt), np.arange(-y_range_tilt//2,y_range_tilt-y_range_tilt//2,1)*tilt_FOV/(detector_px_size*y_range_tilt), indexing="xy")
    x_tilt, y_tilt=x_tilt*np.cos(rotation)+np.sin(rotation)*y_tilt, -x_tilt*np.sin(rotation)+np.cos(rotation)*y_tilt
    tilt_series=(tilt_series>=thresh).astype(np.float32)#[tilt_series<thresh]=0
    ssum=np.sum(tilt_series, axis=(2,3))
    x=np.sum(tilt_series*x_detector[None, None,:,:], axis=(2,3))
    y=np.sum(tilt_series*y_detector[None, None,:,:], axis=(2,3))
    select=ssum!=0
    x_tilt=x_tilt[select]
    y_tilt=y_tilt[select]
    x=x[select]
    y=y[select]
    ssum=ssum[select]
    x/=ssum
    y/=ssum
    tilt_series=tilt_series[select,:,:]
    x_tilt-=np.mean(x_tilt-x)
    y_tilt-=np.mean(y_tilt-y)
    return x_tilt, y_tilt, x,y, x_detector_full, y_detector_full, tilt_series

def unwarp_4dstem(path_numpy_old, path_numpy_new="", field=None, order=5, save=True, extra_angle_rotation_compensation=0, threshold=0.2):
    print("unwarping the dataset")
    data_old=np.load(path_numpy_old)
    data_new=np.empty_like(data_old)
    for i in tqdm(range(data_old.shape[0])):
        for j in range(data_old.shape[1]):
            data_new[i,j]=warp_image(data_old[i,j], field, order=order, extra_angle_rotation_compensation=extra_angle_rotation_compensation)
    if not(threshold is None):
        data_new[data_new<threshold]=0
    if save:
        np.save(path_numpy_new, data_new)
    else:
        return data_new

###################### End of the defintions, now comes the fun part  ######################




tilt_FOV=16## mrad, this is the max range of the tilts in our series
detector_px_size=5e-2 ##mrad ## explains itself
thresh_tilt=1000  # this is a threshold for the tilts, converts the tilt-series into binary
tilt_grid_rotation=92 ## deg  ### rotation of a tilt grid, i estimated it by eye. One can potentially do it in a more clever way
flip_y=False ## indicate whether the tilt series is flipped (in our case it's not)
thresh_patterns=0.2 ### threshold for the interpolated patterns. After unwarping a bunch of negative values appear, so everything smaller than 0.2 counts is set to zero
extra_angle_rotation_compensation=-98.7+180 ##deg, PL rotation. Since we anyway interpolate the data, we can also compensate the PL rotation so that the object size can be smaller. This value was initially determined on the microscope with a HAADF & ELA images, this was -97.7 deg. I unwarped the ELA image and determined it a bit more precisely, -98.7 deg. Further there is an additional 180 deg rotation (the ptycho phase was comming out inverted without extra 180 deg rotation).

import argparse
p = argparse.ArgumentParser(description="Unwarp 4D datasets")
p.add_argument('--folder', help='Directory where the 4D datasets are stored')
p.add_argument('names', nargs='*',
               help='Names of the 4D datasets to unwarp, without the .npy extension')
args=p.parse_args()
path=args.folder or "/Users/anton/Desktop/ptychography_datasets/01082024/"
tilt_series_path=os.path.join(
    path,
    "20240801_vacuum_scan_pl_distortion_tilt_series_fov_16_mrad_ELA.npy")
## this is a path to ELA tilt series.


## here comes the fitting of a deformation vector field.
tilt_series_full=np.load(tilt_series_path)
x_tilt, y_tilt, x,y, x_detector_full, y_detector_full, tilt_series=get_tilt_coordinates(tilt_grid_rotation,flip_y, tilt_series_full,tilt_FOV, detector_px_size, thresh_tilt)
field=fit_vector_field(x, y, x_tilt-x, y_tilt-y, x_detector_full,y_detector_full,method='cubic')


#### now here comes the part where one iterated over all 4dstem data and corrects it!

name_collection = (args.names or
                   ["20240801_cryo_Apo_pos10_1060nm_200_dose_20eA2_4DSTEM"])

for name_4dstem_dataset in name_collection:
    print(f"Unwarping {name_4dstem_dataset}...")
    unwarp_4dstem(os.path.join(path, name_4dstem_dataset+".npy"),
                  os.path.join(path, name_4dstem_dataset+"_unwarped.npy"),
                  field, order=5, save=True, extra_angle_rotation_compensation=extra_angle_rotation_compensation, threshold=thresh_patterns)
    print("Done!")

## note, at the end the pattern still looks elliptic, althugh the tilt series grid looks straight =>  something was not aligned properly, but we should trust the unwarped elliptical patterns.
