from prepare_folder import rename_folders, separate_unflagged_rgb, rename_files, unite_unflagged_flagged
from split_to_tiles import create_tiles


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


# reformat_folders()

root_directory = "../data/data_rgb/2022_06_20"
# root_directory = "../data/test"
# if the patch size is 250 and we want an overlap of 50, the step size would be 200
tile_size = 250
step_size = 240
max_image_pixels = 933120000

# create_tiles(root_directory, tile_size, step_size, max_image_pixels)
