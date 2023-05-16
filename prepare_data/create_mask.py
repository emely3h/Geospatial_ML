import os

import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical


def label_pixels(img):
    mask1 = img == 0
    mask2 = (img == 255) | (img == 253)
    img[mask1] = 2
    img[mask2] = 0
    img[~(mask1 | mask2)] = 1
    return img


def create_mask(root_folder, max_image_pixels):
    Image.MAX_IMAGE_PIXELS = max_image_pixels
    wq_flagged_folder = f"{root_folder}/tiles_wq_flags_applied"
    mask_folder = f"{root_folder}/mask_wq"

    if not os.path.exists(mask_folder):
        os.mkdir(mask_folder)

    for filename in os.listdir(wq_flagged_folder):
        file_path = os.path.join(wq_flagged_folder, filename)
        img = np.asarray(Image.open(file_path)).copy()
        img = label_pixels(img)
        mask_array_path = os.path.join(mask_folder, f'{filename.split(".")[0]}')
        np.save(f"{mask_array_path}.npy", img)
        print(f"Saved {mask_array_path}")


def create_physical_mask(x_input: np.ndarray) -> np.ndarray:
    wq_channel = x_input[:, :, :, 4]
    labeled = label_pixels(wq_channel)
    print(f'shape after labeleling pixels: {labeled.shape} max: {np.max(labeled)} min: {np.min(labeled)}')
    return to_categorical(labeled, num_classes=3)
