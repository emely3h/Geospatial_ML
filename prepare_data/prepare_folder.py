import os
from datetime import datetime


def rename_folders(dir_path):
    folders = os.listdir(dir_path)
    for folder in folders:
        old_path = os.path.join(dir_path, folder)

        folder_name = os.path.basename(folder)
        date_str = folder_name.split('_')[3]
        date = f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:]}"

        new_folder_name = dir_path + '/' + date
        new_folder_path = os.path.join(os.path.dirname(folder), new_folder_name)

        os.rename(old_path, new_folder_path)

        print(f"Folder {folder} renamed to {new_folder_path}")


def separate_unflagged_rgb(src_dir, dest_dir):
    os.makedirs(dest_dir)
    folders = os.listdir(src_dir)
    split_date = datetime.strptime('2022_06_15', '%Y_%m_%d').date()
    for folder in folders:
        folder_name = os.path.basename(folder)
        date = datetime.strptime(folder_name, '%Y_%m_%d').date()
        if date > split_date:
            new_folder_name = dest_dir + folder_name
            old_folder_path = src_dir + folder_name

            os.rename(old_folder_path, new_folder_name)

    print("Copied all unflagged folder with rgb image into a separate folder.")
