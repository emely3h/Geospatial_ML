import os
from datetime import datetime


def rename_folders(dir_path):
    folders = os.listdir(dir_path)
    for folder in folders:
        old_path = os.path.join(dir_path, folder)

        folder_name = os.path.basename(folder)
        if (folder_name != '.DS_Store'):
            date_str = folder_name.split('_')[3]
            date = f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:]}"
            print(date)

            new_folder_name = dir_path + '/' + date
            new_folder_path = os.path.join(
                os.path.dirname(folder), new_folder_name)

            os.rename(old_path, new_folder_path)

            print(f"Folder {folder} renamed to {new_folder_path}")


def separate_unflagged_rgb(src_dir, dest_dir):
    os.makedirs(dest_dir)
    folders = os.listdir(src_dir)
    split_date = datetime.strptime('2022_06_15', '%Y_%m_%d').date()
    for folder in folders:
        folder_name = os.path.basename(folder)
        if (folder_name != '.DS_Store'):
            date = datetime.strptime(folder_name, '%Y_%m_%d').date()
            if date > split_date:
                new_folder_name = dest_dir + folder_name
                old_folder_path = src_dir + folder_name

                os.rename(old_folder_path, new_folder_name)

    print("Copied all unflagged folder with rgb image into a separate folder.")


def rename_file(old_filename, new_filename, root):
    file_path = os.path.join(root, old_filename)
    new_file_path = os.path.join(root, new_filename)
    os.rename(file_path, new_file_path)
    print(f'Renamed {file_path} to {new_file_path}')


def rename_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.startswith('RGB'):
                rename_file(filename, 'rgb.tif', root)
            if filename.startswith('TUR'):
                rename_file(filename, 'wq.tif', root)
