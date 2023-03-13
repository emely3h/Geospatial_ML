from prepare_folder import rename_folders, separate_unflagged_rgb, rename_files, unite_unflagged_flagged
from split_to_tiles import create_tiles
from create_mask import prepare
import os


def reformat_folders():
    rename_folders('../data/unflagged')
    rename_folders('../data/flags_applied')
    separate_unflagged_rgb('../data/unflagged/', '../data/unflagged_rgb/')
    separate_unflagged_rgb('../data/flags_applied/', '../data/flags_applied_rgb/')
    rename_files('../data/unflagged_rgb/')
    rename_files('../data/flags_applied/')
    rename_files('../data/flags_applied_rgb/')
    rename_files('../data/unflagged/')
    unite_unflagged_flagged('../data/flags_applied_rgb/', '../data/unflagged_rgb/', '../data/data_rgb')
    unite_unflagged_flagged('../data/flags_applied/', '../data/unflagged', '../data/data_without_rgb')


def count_files_in_directory(directory):
    file_count = 0
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_count += 1
    return file_count


def get_rgb_set(rgb_folder):
    rgb_set = set(os.listdir(rgb_folder))
    for file in rgb_set.copy():

        file_split = file.split('_')
        new_name = file_split[1] + '_' + file_split[2].split('.')[0]
        rgb_set.remove(file)
        rgb_set.add(new_name)
    return rgb_set


def get_wq_unflagged(wq_folder):
    wq = set(os.listdir(wq_folder))
    for file in wq.copy():
        file_split = file.split('_')
        new_name = file_split[2] + '_' + file_split[3].split('.')[0]
        wq.remove(file)
        wq.add(new_name)
    return wq


def get_wq_flagged(wq_folder):
    wq = set(os.listdir(wq_folder))
    for file in wq.copy():
        file_split = file.split('_')
        new_name = file_split[3] + '_' + file_split[4].split('.')[0]
        wq.remove(file)
        wq.add(new_name)
    return wq


def test_splitting(to_delete_set=None):
    print(count_files_in_directory(f'{root_directory}/tiles_rgb'))
    print(count_files_in_directory(f'{root_directory}/tiles_wq_flags_applied'))
    print(count_files_in_directory(f'{root_directory}/tiles_wq_unflagged'))

    rgb_set = get_rgb_set(f'{root_directory}/tiles_rgb')
    wq_flags = get_wq_flagged(f'{root_directory}/tiles_wq_flags_applied')
    wq_unflagged = get_wq_unflagged(f'{root_directory}/tiles_wq_unflagged')

    print(len(rgb_set))
    print(len(wq_flags))
    print(len(wq_unflagged))
    if to_delete_set:
        print(len(to_delete_set))

    print(rgb_set == wq_flags)
    print(rgb_set == wq_unflagged)
    print(wq_flags == wq_unflagged)

    if to_delete_set:
        print(to_delete_set.issubset(rgb_set))
        print(to_delete_set.issubset(wq_unflagged))
        print(to_delete_set.issubset(wq_flags))


root_directory = "../data/data_rgb/2022_06_20"
# root_directory = "../data/test"
# if the patch size is 250 and we want an overlap of 50, the step size would be 200
tile_size = 250
step_size = 240
max_image_pixels = 933120000

#reformat_folders()

#create_tiles(root_directory, tile_size, step_size, max_image_pixels)

#test_splitting()

#to_delete_set = prepare(root_directory)

import numpy as np
from PIL import Image

def load_data_x_y(root_directory):
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
    x_numpy = np.array(x)
    np.save(os.path.join(root_directory, 'x_input.npy'), x_numpy)

    y_numpy = np.array(y)
    np.save(os.path.join(root_directory, 'y_input.npy'), y_numpy)

    print(f'X_input array shape: {x_numpy.shape}')
    print(f'Y_input array shape: {y_numpy.shape}')

load_data_x_y(root_directory)


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


