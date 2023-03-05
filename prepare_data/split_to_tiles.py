import numpy as np
from PIL import Image
import PIL
from dotenv import load_dotenv
import os
load_dotenv()
data_path = os.environ.get('DATA_PATH')


def fil_up_tile(tile, tile_size, third_dimension):
    if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
        return tile
    new_tile = np.zeros((tile_size, tile_size, third_dimension))

    new_tile[:tile.shape[0], :tile.shape[1], :] = tile
    return new_tile


def split_image_into_tiles(array, tile_size, step_size, save_folder):
    result = []
    row_pixel, col_pixel, third_dim = array.shape
    row_index = 0
    print(f'Image shape: {array.shape}')
    counter = 1
    while row_index < row_pixel:
        print(f'Current row index: {row_index}')
        col_index = 0
        while col_index < col_pixel:
            tile = array[row_index:row_index + tile_size, col_index:col_index + tile_size, :]
            tile = fil_up_tile(tile, tile_size, third_dim)
            result.append(tile.tolist())

            img = Image.fromarray(np.uint8(tile.tolist()))
            img.save(f'{data_path}{save_folder}splitted_{counter}_{row_index}_{col_index}.png')
            counter += 1

            col_index += step_size
        row_index += step_size

    return result


def image_tiles_array(img_path, tile_size, step_size):
    PIL.Image.MAX_IMAGE_PIXELS = 191357533
    img = np.array(Image.open(img_path))
    print(img.shape)
    tiles = split_image_into_tiles(img, tile_size, step_size)
    print(f'finished splitting up img {img_path}')
    return tiles


test_x = image_tiles_array(f'{data_path}unflagged_rgb/2022_07_10/rgb.tif', 500, 490, 'unflagged_rgb/2022_07_10/')
#test = image_tiles_array('../data/test_image.jpeg', 100, 90)

def create_mask_array(img_path):
    PIL.Image.MAX_IMAGE_PIXELS = 191357533
    img_array = np.array(Image.open(img_path))
    print(img_array.shape)
    mask_array = np.zeros(img_array)
    for row, idx in enumerate(img_array):
        for col, idx in enumerate():
            pass
            # if not 253, 255 or 0 pixel is valid otherwise invalid => what does 254 mean?
    return mask_array

#test_y = create_mask_array('../data/flags_applied/2022_07_10/wq.tif')

# https://github.com/Devyanshu/image-split-with-overlap
# test with saving in between
# test with numpy strides
# use other libraries => cv2, patchify, dask