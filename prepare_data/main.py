from prepare_folder import rename_folders, separate_unflagged_rgb, rename_files
from split_to_tiles import create_tiles

def reformat_folders():
    rename_folders('../data/unflagged')
    rename_folders('../data/flags_applied')
    separate_unflagged_rgb('../data/unflagged/', '../data/unflagged_rgb/')
    rename_files('../data/unflagged_rgb/')
    rename_files('../data/flags_applied/')
    rename_files('../data/unflagged/')


root_directory = "../data/unflagged_rgb/2022_06_20"
#root_directory = "../data/test"
# if the patch size is 250 and we want an overlap of 50, the step size would be 200
tile_size = 250
step_size = 240
max_image_pixels = 933120000

create_tiles(root_directory, tile_size, step_size, max_image_pixels)