import numpy as np


class Mask_Stats:

    """
    takes y_mask numpy array to instanciate object
    gives overview over how many pixels per label
    """

    def __init__(self, y_array):
        self.y_array = y_array

        self.pix_land = self.num_of_pixels_per_class(y_array, 2)
        self.pix_valid = self.num_of_pixels_per_class(y_array, 1)
        self.pix_invalid = self.num_of_pixels_per_class(y_array, 0)
        self.sum_pix = self.pix_land + self.pix_valid + self.pix_invalid

        self.pix_land_per = 100 / self.sum_pix * self.pix_land
        self.pix_valid_per = 100 / self.sum_pix * self.pix_valid
        self.pix_invalid_per = 100 / self.sum_pix * self.pix_invalid
        self.print_stats()

    def num_of_pixels_per_class(self, y_mask, label):
        flatten = np.reshape(y_mask, (-1,))
        pixel_match = (flatten == label)
        pix_per_class = np.count_nonzero(pixel_match)
        return pix_per_class

    def print_stats(self):
        print(f'Shape: {self.y_array.shape}')
        print(f'Land pixels: {self.pix_land}  {self.pix_land_per:.3f} %')
        print(f'Valid pixels: {self.pix_valid}  {self.pix_valid_per:.3f} %')
        print(f'Invalid pixels: {self.pix_invalid}  {self.pix_invalid_per:.3f} %')
        # print("Class distribution:", np.bincount(y_array))
        print(f'Sum: {(self.pix_land + self.pix_valid + self.pix_invalid) // 256 // 256}')