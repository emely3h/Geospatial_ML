import os

from dotenv import load_dotenv

from filter_tiles import filter_useful_tiles
from prepare_folder import rename_folders, separate_unflagged_rgb, rename_files, unite_unflagged_flagged, \
    delete_duplicate_data, extract_model_arrays
from save_as_array import save_data_x_y
from split_to_tiles import create_tiles

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


def prepare_all_data(data_path):
    for folder in os.listdir(data_path):
        print(f'================== START PREPARING FOLDER {folder} =================================')
        folder_path = os.path.join(data_path, folder)
        create_tiles(folder_path, tile_size, step_size, 933120000)
        filter_useful_tiles(folder_path)
        save_data_x_y(folder_path, tile_size, step_size)
        print(f'================== FINISHED PREPARING FOLDER {folder} =================================')
    extract_model_arrays(data_path)


date_folder = f'{data_path}/data_rgb/2022_07_05'
tile_size = 256
step_size = 200

# create folderstructure from emely_marcel_flagging.zip dataset
reformat_folders()

# split images into tiles, input one date folder splits rgb image and both wq images
create_tiles(date_folder, tile_size, step_size, 933120000)

# deletes tiles that contain only useless pixels
filter_useful_tiles(date_folder)

# saves final input data (X = input images and y = mask images) as 2 separate numpy arrays
save_data_x_y(date_folder, tile_size, step_size)

# prepare all folders at once
prepare_all_data(f'{data_path}/data_rgb')

# extract numpy arrays (x_train, y_mask) and copy them to a separate folder to prepare for google drive upload
extract_model_arrays(f'{data_path}/data_rgb')

# save all compressed numpy arrays in one file instead of one file per image
# only works if extract_mode_arrays() has been executed before
# when running local only enough RAM for 2 images => execute in colab
# combine_npz_arrays(f'{data_path}/data_rgb')
