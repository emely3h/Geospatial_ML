import os
from datetime import datetime
import numpy as np
from PIL import Image
from patchify import patchify


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

# if the patch size is 256 and we want an overlap of 56, the step size would be 200
def save_patches(root_directory, dest_folder_name="patches", patch_size=256, step=200, max_image_pixels = 933120000):
    Image.MAX_IMAGE_PIXELS = max_image_pixels
    for path, subdirs, files in os.walk(root_directory):
        path_list = path.split('/')
        path_list.insert(2, dest_folder_name)
        dest_dir = '/'.join(path_list)
        os.makedirs(dest_dir, exist_ok=True)
        for name in files:
            if name.endswith('tif'):
                file_path = os.path.join(path, name)
                img = np.asarray(Image.open(file_path))
                
                if len(img.shape)==3:
                    third_dim = img.shape[2]
                    patches_img  = patchify(img, (patch_size, patch_size, third_dim), step=step)
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            for k in range(third_dim):
                                patch = patches_img[i,j,k,:,:]
                                patch_name = f'{name}_{i}_{j}_{k}.tif'
                                print(patch_name)
                                patch_path = os.path.join(dest_dir, patch_name)
                                Image.fromarray(patch).save(patch_path)
                                print(f'Saved {patch_path}')
                else:
                    patches_img  = patchify(img, (patch_size, patch_size), step=step)
                    for i in range(patches_img.shape[0]):
                        for j in range(patches_img.shape[1]):
                            patch = patches_img[i,j,:,:]
                            patch_name = f'{name}_{i}_{j}.tif'
                            print(patch_name)
                            patch_path = os.path.join(dest_dir, patch_name)
                            Image.fromarray(patch).save(patch_path)
                            print(f'Saved {patch_path}')