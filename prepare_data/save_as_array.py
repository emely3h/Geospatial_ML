import numpy as np
from PIL import Image
import os


"""
for every entry in rgb
- load image 
- convert image to numpy array
- load associated wq_unflagged image
- concatenate arrays
- append saved array to X
- laod associated mask array
- append array to Y
at the end of folder save X and Y
"""


def save_data_x_y(root_directory, tile_size, step_size):
    x = []
    y = []
    rgb_folder = os.path.join(root_directory, 'tiles_rgb')
    for tile_name in os.listdir(rgb_folder):
        rgb_tile_path = os.path.join(rgb_folder, tile_name)
        tile_array_rgb = np.asarray(Image.open(rgb_tile_path))
        tile_name_split = tile_name.split('_')
        row_col = tile_name_split[1] + '_' + tile_name_split[2].split('.')[0]
        wq_tile_path = f'{root_directory}/tiles_wq_unflagged/wq_unflagged_{row_col}.tif'
        mask_tile_path = f'{root_directory}/tiles_wq_unflagged/wq_unflagged_{row_col}.tif'
        tile_array_wq = np.asarray(Image.open(wq_tile_path))
        width, height = tile_array_wq.shape
        x_array = np.zeros((width, height, 5))
        # Copy the first 4 channels of the 4-channel image into the new 5-channel image
        x_array[..., :4] = tile_array_rgb

        # Copy the 1-channel image into the new 5-channel image as the fifth channel
        x_array[..., 4] = tile_array_wq

        x.append(x_array)

        y.append(np.load(f'{root_directory}/mask_wq/wq_flags_applied_{row_col}.npy'))
    date = root_directory.split('/')[-1]
    x_numpy = np.array(x)
    np.save(os.path.join(root_directory, f'x_input_{tile_size}_{step_size}_{date}.npy'), x_numpy)

    y_numpy = np.array(y)
    np.save(os.path.join(root_directory, f'y_mask_{tile_size}_{step_size}_{date}.npy'), y_numpy)

    print(f'X_input array shape: {x_numpy.shape}')
    print(f'Y_input array shape: {y_numpy.shape}')