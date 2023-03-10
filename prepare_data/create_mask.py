from PIL import Image
import numpy as np
import os


# convert all wq.tif tiles in flags_applied into mask => convert pixel value to label
def create_mask(folder_path, max_image_pixels=933120000,):
    Image.MAX_IMAGE_PIXELS = max_image_pixels

    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            if dir_name.startswith('tiles_wq'):
                folder = os.path.join(root, dir_name)
                for filename in os.listdir(folder):
                    file_pathname = os.path.join(folder, filename)
                    img = np.asarray(Image.open(file_pathname))

                    for row in range(img.shape[0]):
                        for column in range(img.shape[1]):
                            pixel = img[row, column]
                            if pixel == 0:
                                img[row, column] = 2
                            if pixel == 255 or pixel == 253:
                                img[row, column] = 0
                            else:
                                img[row, column] = 1
    # save tiles in same tiles_mask with same filename
    # if all pixels in tile are category 2 do not save tile but do delete same tile unflagged rgb image and unflagged wq image
