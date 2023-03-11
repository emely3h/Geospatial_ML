import os
from datetime import datetime
import shutil


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