from PIL import Image
import numpy as np
import os
import fnmatch


def label_pixels(img):
    mask1 = (img == 0)
    mask2 = (img == 255) | (img == 253)
    img[mask1] = 2
    img[mask2] = 0
    img[~(mask1 | mask2)] = 1
    return img


def check_save_tile(img):
    return np.any(img != 2)


def delete_tile(tile_name, folder_name):
    found_file = False
    for filename in os.listdir(folder_name):
        if fnmatch.fnmatch(filename, f'*_{tile_name}.tif'): #Todo fix error
            file_path = os.path.join(folder_name, filename)
            os.remove(file_path)
            found_file = True
            print(f"Found and deleted file '{file_path}'")
    if not found_file:
        raise Exception(f'Tried to delete tile {tile_name} in folder {folder_name}')


def delete_tiles(tiles_to_delete, folder):
    wq_folder = os.path.join(f'{folder}/tiles_wq_flags_applied')
    rgb_folder = os.path.join(f'{folder}/tiles_rgb')
    for tile in tiles_to_delete:
        delete_tile(tile, wq_folder)
        delete_tile(tile, rgb_folder)


# folder_path always date_folder like data/data_rgb/2022_06_20 (without / in the end)
def create_mask(root_folder, max_image_pixels):
    Image.MAX_IMAGE_PIXELS = max_image_pixels
    wq_unflagged_folder = f'{root_folder}/tiles_wq_unflagged'
    mask_folder = f'{root_folder}/mask_wq'
    tiles_to_delete = []

    if not os.path.exists(mask_folder):
        os.mkdir(mask_folder)

    counter = 0

    for filename in os.listdir(wq_unflagged_folder):
        print(f'Counter: {counter}')
        counter += 1
        file_path = os.path.join(wq_unflagged_folder, filename)
        img = np.asarray(Image.open(file_path)).copy()
        img = label_pixels(img)
        if check_save_tile(img):
            mask_array_path = os.path.join(mask_folder, f'{filename}')
            np.save(f'{mask_array_path}.npy', img)
            print(f"Saved {mask_array_path}")
        else:
            parts = filename.rsplit('_', 2)
            row_col_names = parts[1] + '_' + parts[2].split('.')[0]
            tiles_to_delete.append(row_col_names)
            print(f"Tile to delete {row_col_names}")
    return tiles_to_delete


def prepare(folder_path, max_image_pixels=933120000):
    tiles_to_delete = create_mask(folder_path, max_image_pixels)
    np.save(f'{folder_path}/tiles_to_delete.npy', tiles_to_delete)
    tiles_to_delete = np.load(f'{folder_path}/tiles_to_delete.npy')
    print(len(tiles_to_delete))
    print(tiles_to_delete)
    delete_tiles(tiles_to_delete, folder_path)
