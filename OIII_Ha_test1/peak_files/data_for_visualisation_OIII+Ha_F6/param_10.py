import numpy as np
from astropy.io import fits


Cube_sel    = False
HST_sel     = False 
Velo_sel    = False
Sigma_sel   = False
S2hasig_sel = False

# Paths
path_data      = 'peak_files/data_for_visualisation_OIII+Ha_F6/'
path_amp_nonan = 'peak_files/data_for_visualisation_OIII+Ha_F6/M33_F6_HaOIII_amp_nonan.fits' 
path_cube      = ''
path_sky       = ''
path_hst_1     = ''
path_hst_2     = ''
path_velo      = ''
path_sigma     = ''
path_s2hasig   = ''
path_coord     = '../M33-Maps/M33-F6/M33-F6_SN3.LineMaps.map.6563.1x1.amplitude.fits'
    
# Central position (in pixel) of SITELLE FoV
x_cengal = 1025
y_cengal = 1025

ximin, ximax, yimin, yimax = 50, 2000, 50, 2000

# Local emission size
lista_les  = np.array([6, 10, 14, 20, 30])

# Laplacian range
laplacian_range = np.array([1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0])

# SNR map
imagen_SNR = 'data_for_visualisation/SNR3.fits'

size = 10

nbring_ = '3'

