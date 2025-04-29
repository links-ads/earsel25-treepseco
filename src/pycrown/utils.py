import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import filters

def _smooth_raster(raster:np.ndarray, ws, circular=False):
    """ Smooth a raster with a median filter

    Parameters
    ----------
    raster :      ndarray
                    raster to be smoothed
    ws :          int
                    window size of smoothing filter
    resolution :  int
                    resolution of raster in m
    circular :    bool, optional
                    set to True for disc-shaped filter kernel, block otherwise

    Returns
    -------
    ndarray
        smoothed raster
    """
    return filters.median_filter(
        raster, footprint=_get_kernel(ws, circular=circular))

def filter_chm(chm: np.ndarray, ws, circular=False):
    ''' Pre-process the canopy height model (smoothing and outlier removal).
    The original CHM (self.chm0) is not overwritten, but a new one is
    stored (self.chm).

    Parameters
    ----------
    ws :            int
                    window size of smoothing filter in metre (set in_pixel=True, otherwise)
    ws_in_pixels :  bool, optional
                    sets ws in pixel
    circular :      bool, optional
                    set to True for disc-shaped filter kernel, block otherwise
    '''
    chm0 = chm.copy()
    chm = _smooth_raster(chm0, ws, circular=circular)
    chm0[np.isnan(chm0)] = 0.
    zmask = (chm < 0.5) | np.isnan(chm) | (chm > 60.)
    chm[zmask] = 0
    return chm

def _get_kernel(radius=5, circular=False):
    """ returns a block or disc-shaped filter kernel with given radius

    Parameters
    ----------
    radius :    int, optional
                radius of the filter kernel
    circular :  bool, optional
                set to True for disc-shaped filter kernel, block otherwise

    Returns
    -------
    ndarray
        filter kernel
    """
    if circular:
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        return x**2 + y**2 <= radius**2
    else:
        return np.ones((int(radius), int(radius)))

def tree_detection(raster, ws=5, hmin=20):
    ''' Detect individual trees from CHM raster based on a maximum filter.
    Identified trees are either stores as list in the tree dataframe or
    returned as ndarray.
    
    Parameters
    ----------
    raster :        ndarray
                    raster of height values (e.g., CHM)
    resolution :    int, optional
                    resolution of raster in m
    ws :            float
                    moving window size (in metre) to detect the local maxima
    hmin :          float
                    Minimum height of a tree. Threshold below which a pixel
                    or a point cannot be a local maxima
    return_trees :  bool
                    set to True if detected trees shopuld be returned as
                    ndarray instead of being stored in tree dataframe
    ws_in_pixels :  bool
                    sets ws in pixel
    
    Returns
    -------
    ndarray (optional)
        nx2 array of tree top pixel coordinates
    '''
    
    # Maximum filter to find local peaks
    raster_maximum = filters.maximum_filter(raster, footprint=_get_kernel(ws, circular=True))
    tree_maxima = raster == raster_maximum
    
    # remove tree tops lower than minimum height
    tree_maxima[raster <= hmin] = 0
    
    # label each tree
    tree_markers, num_objects = ndimage.label(tree_maxima)
    
    if num_objects == 0:
        return np.empty((0, 2))
    
    # if canopy height is the same for multiple pixels,
    # place the tree top in the center of mass of the pixel bounds
    yx = np.array(
            ndimage.center_of_mass(
                raster, tree_markers, range(1, num_objects+1)
            ), dtype=np.float32
        ) + 0.5
    xy = np.array((yx[:, 1], yx[:, 0])).T
    
    return xy

def heatmap2points_pycrown(heatmap: np.ndarray, ws_smooth: int, ws_detection: int, hmin: float)->np.ndarray:
    if ws_smooth != 0:
        smoothed_heatmap = filter_chm(heatmap, ws=ws_smooth, circular=False)
    else:
        smoothed_heatmap = heatmap
    peaks_np = tree_detection(smoothed_heatmap, ws=ws_detection, hmin=hmin)
    return peaks_np