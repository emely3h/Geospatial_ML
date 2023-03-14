from prepare_folder import rename_folders, separate_unflagged_rgb, rename_files, unite_unflagged_flagged, delete_duplicate_data
from split_to_tiles import create_tiles
from filter_tiles import filter_useful_tiles
import os
from save_as_array import save_data_x_y
from dotenv import load_dotenv
load_dotenv()

data_path = os.environ.get('DATA_PATH')


def reformat_folders():
    rename_folders(f'{data_path}/unflagged')
    rename_folders(f'{data_path}/flags_applied')
    separate_unflagged_rgb(f'{data_path}/unflagged/', f'{data_path}/unflagged_rgb/')
    separate_unflagged_rgb(f'{data_path}/flags_applied/', f'{data_path}/flags_applied_rgb/')
    rename_files(f'{data_path}/unflagged_rgb/')
    rename_files(f'{data_path}/flags_applied/')
    rename_files(f'{data_path}/flags_applied_rgb/')
    rename_files(f'{data_path}/unflagged/')
    unite_unflagged_flagged(f'{data_path}/flags_applied_rgb/', f'{data_path}/unflagged_rgb/', f'{data_path}/data_rgb')
    unite_unflagged_flagged(f'{data_path}/flags_applied/', f'{data_path}/unflagged', f'{data_path}/data_without_rgb')
    delete_duplicate_data(data_path)


root_directory = f'{data_path}/data_rgb/2022_06_20'
tile_size = 256
step_size = 200

# create folderstructure from emely_marcel_flagging.zip dataset
#reformat_folders()

# split images into tiles, input one date folder splits rgb image and both wq images
# create_tiles(root_directory, tile_size, step_size, 933120000)

# tests if the splitting step worked
#test_splitting()

# deletes tiles that contain only useless pixels
#filter_useful_tiles(root_directory)

# saves final input data (X = input images and y = mask images) as 2 separate numpy arrays
#save_data_x_y(root_directory, tile_size, step_size)





