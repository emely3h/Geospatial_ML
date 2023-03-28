import os
from datetime import datetime
import shutil
import numpy as np


def rename_folders(dir_path):
    folders = os.listdir(dir_path)
    for folder in folders:
        old_path = os.path.join(dir_path, folder)

        folder_name = os.path.basename(folder)
        if folder_name != ".DS_Store":
            date_str = folder_name.split("_")[3]
            date = f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:]}"
            print(date)

            new_folder_name = dir_path + "/" + date
            new_folder_path = os.path.join(
                os.path.dirname(folder), new_folder_name)

            os.rename(old_path, new_folder_path)

            print(f"Folder {folder} renamed to {new_folder_path}")


def separate_unflagged_rgb(src_dir, dest_dir):
    os.makedirs(dest_dir)
    folders = os.listdir(src_dir)
    split_date = datetime.strptime("2022_06_15", "%Y_%m_%d").date()
    for folder in folders:
        folder_name = os.path.basename(folder)
        if folder_name != ".DS_Store":
            date = datetime.strptime(folder_name, "%Y_%m_%d").date()
            if date > split_date:
                new_folder_name = dest_dir + folder_name
                old_folder_path = src_dir + folder_name

                os.rename(old_folder_path, new_folder_name)

    print("Copied all unflagged folder with rgb image into a separate folder.")


def rename_file(old_filename, new_filename, root):
    file_path = os.path.join(root, old_filename)
    new_file_path = os.path.join(root, new_filename)
    os.rename(file_path, new_file_path)
    print(f"Renamed {file_path} to {new_file_path}")


def rename_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.startswith("RGB"):
                rename_file(filename, "rgb.tif", root)
            if filename.startswith("TUR"):
                if 'unflagged' in folder_path:
                    rename_file(filename, "wq_unflagged.tif", root)
                else:
                    rename_file(filename, "wq_flags_applied.tif", root)


def unite_unflagged_flagged(path1, path2, dest_path):
    subfolders1 = [os.path.join(path1, f) for f in os.listdir(path1) if os.path.isdir(os.path.join(path1, f))]
    subfolders2 = [os.path.join(path2, f) for f in os.listdir(path2) if os.path.isdir(os.path.join(path2, f))]

    common_folders = set([os.path.basename(f1) for f1 in subfolders1]).intersection(
        set([os.path.basename(f2) for f2 in subfolders2]))

    if common_folders:
        merged_folder = os.path.join(os.getcwd(), dest_path)
        if not os.path.exists(merged_folder):
            os.makedirs(merged_folder)

        for folder in common_folders:
            folder1 = os.path.join(path1, folder)
            folder2 = os.path.join(path2, folder)
            merged_subfolder = os.path.join(merged_folder, folder)
            if not os.path.exists(merged_subfolder):
                os.makedirs(merged_subfolder)
            for f in os.listdir(folder1):
                shutil.copy(os.path.join(folder1, f), merged_subfolder)
            for f in os.listdir(folder2):
                shutil.copy(os.path.join(folder2, f), merged_subfolder)
    else:
        print('No common folders found.')


def delete_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"{path} has been deleted.")


def delete_duplicate_data(data_root):
    flags_applied_rgb_folder = os.path.join(data_root, 'flags_applied_rgb')
    flags_applied_folder = os.path.join(data_root, 'flags_applied')
    unflagged_folder = os.path.join(data_root, 'unflagged')
    unflagged_rgb_folder = os.path.join(data_root, 'unflagged_rgb')

    delete_folder(flags_applied_folder)
    delete_folder(flags_applied_rgb_folder)
    delete_folder(unflagged_folder)
    delete_folder(unflagged_rgb_folder)


def extract_model_arrays(data_path):
    dest_path = f'{data_path}/google_drive'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for folder in os.listdir(data_path):
        folder_path = f'{data_path}/{folder}'

        print(f'Checking folder {folder_path}')

        arrays_dict = {}
        for file in os.listdir(folder_path):
            if file.startswith('x_input_'):
                array = np.load(f'{folder_path}/{file}')
                arrays_dict['x_input'] = array
            if file.startswith('y_mask'):
                array = np.load(f'{folder_path}/{file}')
                arrays_dict['y_mask'] = array
        np.savez_compressed(f'{dest_path}/{folder_path.split("/")[-1]}', **arrays_dict)
    print('Data ready for uploading to google drive.')


def combine_npz_arrays(data_path):
    drive_path = f'{data_path}/google_drive'
    arrays_dict = {}
    for file in os.listdir(drive_path):
        if file != '2022_08_09.npz':
            print(f'Adding image {file}')
            array = np.load(f'{drive_path}/{file}')
            x_input = array['x_input']
            y_mask = array['y_mask']
            if len(arrays_dict) < 1:
                arrays_dict['x_input'] = x_input
                arrays_dict['y_mask'] = y_mask
            else:
                arrays_dict['x_input'] = np.concatenate((arrays_dict['x_input'], x_input), axis=0)
                arrays_dict['y_mask'] = np.concatenate((arrays_dict['y_mask'], y_mask), axis=0)
    np.savez_compressed(f'{data_path}/{drive_path}/all_images', **arrays_dict)
    print('Combined all compressed numpy images into one single file.')

