from prepare_folder import rename_folders, separate_unflagged_rgb, rename_files, unite_unflagged_flagged
from split_to_tiles import create_tiles
from filter_tiles import filter_useful_tiles
import os
from save_as_array import save_data_x_y
from dotenv import load_dotenv
load_dotenv()

data_path = os.environ.get('DATA_PATH')


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
tile_size = 256
step_size = 200
max_image_pixels = 933120000

# create folderstructure from emely_marcel_flagging.zip dataset
#reformat_folders()

# split images into tiles, input one date folder splits rgb image and both wq images
#create_tiles(root_directory, tile_size, step_size, max_image_pixels)

# tests if the splitting step worked
#test_splitting()

# deletes tiles that contain only useless pixels
#to_delete_set = filter_useful_tiles(root_directory)

# saves final input data (X = input images and y = mask images) as 2 separate numpy arrays
save_data_x_y(root_directory)





