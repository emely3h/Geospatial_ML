import os
from dotenv import load_dotenv
load_dotenv()


data_path = os.environ.get('DATA_PATH')
root_directory = f'{data_path}/data_rgb/2022_06_20'


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