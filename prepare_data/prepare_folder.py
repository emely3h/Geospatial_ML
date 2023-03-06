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



def generate_patch(img, patch_size, step):
    is3d = len(img.shape) == 3
    images = []
    patch_shape = (patch_size, patch_size, 3) if is3d else (patch_size, patch_size)
    patches_img = patchify(img, patch_shape, step=step)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            patch = patches_img[i, j, 0]
            patch = Image.fromarray(patch)
            images.append({
                'patch': patch,
                'i': i,
                'j': j,
            })
    return images

def create_nested_dir(path, new_dir, where=2):  
    path_list = path.split('/')
    path_list.insert(where, new_dir)
    dest_dir = '/'.join(path_list)
    os.makedirs(dest_dir, exist_ok=True)
    return dest_dir

# if the patch size is 256 and we want an overlap of 56, the step size would be 200
def save_patches(root_directory, new_dir="patches", patch_size=256, step=200, max_image_pixels=933120000):
    Image.MAX_IMAGE_PIXELS = max_image_pixels
    for path, _, files in os.walk(root_directory):
        if new_dir in path:
            return

        dest_dir = create_nested_dir(path, new_dir, 2)

        for name in files:
            if name.endswith('tif'):
                file_path = os.path.join(path, name)
                img = np.asarray(Image.open(file_path))
                images = generate_patch(img, patch_size, step)
                for image in images:
                    patch = image['patch']
                
                    i = image['i']
                    j = image['j']
                    patch_name = f'{name}_{i}_{j}.tif'
                    patch_path = os.path.join(dest_dir, patch_name)
                    patch.save(patch_path)
                    print(f'Saved {patch_path}')