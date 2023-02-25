import numpy as np
from PIL import Image
from time import time
from sys import argv

# The file can be executed with the following command to create a 128x128 tile:
# python split_to_tiles_2.py 128 128

Image.MAX_IMAGE_PIXELS = 933120000


def _time(f):
    def wrapper(*args):
        start = time()
        r = f(*args)
        end = time()
        print("%s timed %f" % (f.__name__, end-start))
        return r
    return wrapper


@_time
def reshape_split(image: np.ndarray, kernel_size: tuple):

    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size

    # pad image to be evenly divisible by tile size
    h_pad = tile_height - img_height % tile_height
    w_pad = tile_width - img_width % tile_width
    pad_width = ((0, h_pad), (0, w_pad), (0, 0))
    image = np.pad(image, pad_width, mode='edge')

    tiled_array = image.reshape(image.shape[0] // tile_height,
                                tile_height,
                                image.shape[1] // tile_width,
                                tile_width,
                                channels)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array


img = np.asarray(Image.open('../data/unflagged_rgb/2022_07_10/rgb.tif'))

t1, t2 = (argv[1], argv[2])
tilesize = (int(t1), int(t2))

tiles = reshape_split(img, tilesize)

n = tiles.shape[0] * tiles.shape[1]
print("This array was split into %d tiles" % (n))
