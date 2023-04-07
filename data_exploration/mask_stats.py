import numpy as np


class Mask_Stats:

    """
    A Mask_Stats object represents the percentage ratio of pixels in the different labels
    init function takes y_mask numpy array of shape (x, 256, 256)

    Parameters:
        pix_land: total number of land pixels
        pix_valid: total number of valid pixels
        pix_invalid: total number of invalid pixels
        pix_land_per: percentage of land pixels
        pix_valid_per: percentage of valid pixels
        pix_invalid_per: percentage of invalid pixels
    """

    __slots__ = [
        "y_array",
        "pix_land",
        "pix_valid",
        "pix_invalid",
        "sum_pix",
        "pix_land_per",
        "pix_valid_per",
        "pix_invalid_per",
    ]

    def __init__(self, y_array: np.ndarray):
        self.y_array = y_array
        self.pix_land: int = self.num_of_pixels_per_class(y_array, 2)
        self.pix_valid: int = self.num_of_pixels_per_class(y_array, 1)
        self.pix_invalid: int = self.num_of_pixels_per_class(y_array, 0)
        self.sum_pix: int = self.pix_land + self.pix_valid + self.pix_invalid

        self.pix_land_per: float = 100 / self.sum_pix * self.pix_land
        self.pix_valid_per: float = 100 / self.sum_pix * self.pix_valid
        self.pix_invalid_per: float = 100 / self.sum_pix * self.pix_invalid

    def num_of_pixels_per_class(self, y_mask: any, label: any) -> int:
        flatten = np.reshape(y_mask, (-1,))
        pixel_match = flatten == label
        pix_per_class = np.count_nonzero(pixel_match)
        return pix_per_class

    def print_stats(self) -> None:
        print(f"Shape: {self.y_array.shape}")
        print(f"Land pixels: {self.pix_land}  {self.pix_land_per:.3f} %")
        print(f"Valid pixels: {self.pix_valid}  {self.pix_valid_per:.3f} %")
        print(f"Invalid pixels: {self.pix_invalid}  {self.pix_invalid_per:.3f} %")
        # print("Class distribution:", np.bincount(y_array))
        print(
            f"Sum: {(self.pix_land + self.pix_valid + self.pix_invalid) // 256 // 256}"
        )


if __name__ == "__main__":
    # 12 sec -> 4sec
    y = np.zeros((6661, 256, 256), dtype=np.float32)
    mask_stats = Mask_Stats(y)
    mask_stats.print_stats()
