import numpy as np
from PIL import Image
import os
from patchify import patchify


def generate_tiles(img, tile_size, step):
    tile_dicts = []
    tile_shape = (tile_size, tile_size)
    is3D = len(img.shape) == 3
    if is3D:
        tile_shape = (tile_size, tile_size, img.shape[2])
    tiles = patchify(img, tile_shape, step=step)
    for row in range(tiles.shape[0]):
        for column in range(tiles.shape[1]):
            tile = tiles[row, column]
            if is3D:
                tile = tiles[row, column, 0]
            tile = Image.fromarray(tile)
            tile_dicts.append(
                {
                    "tile": tile,
                    "row": row * step,
                    "column": column * step,
                }
            )
    return tile_dicts


# if the tile size is 256 and we want an overlap of 56, the step size would be 200
# Todo add padding
def create_tiles(
    folder_path,
    tile_size=256,
    step=200,
    max_image_pixels=933120000,
):
    Image.MAX_IMAGE_PIXELS = max_image_pixels

    for file in os.listdir(folder_path):
        if file == "wq.tif" or file == "rgb.tif":
            dest_dir = f'{folder_path}/tiles_{file.split(".")[0]}'
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            img = np.asarray(Image.open(f'{folder_path}/{file}'))
            # img = img.reshape((img.shape[0], -1))
            tile_dicts = generate_tiles(img, tile_size, step)
            for tile_dict in tile_dicts:
                tile = tile_dict["tile"]
                row = tile_dict["row"]
                column = tile_dict["column"]

                tile_name = f'{file.split(".")[0]}_{row}_{column}.tif'
                tile_path = os.path.join(dest_dir, tile_name)
                tile.save(tile_path)
                print(f"Saved {tile_path}")


def delete_all_tiles():
    pass
