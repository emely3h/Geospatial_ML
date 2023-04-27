import numpy as np
import cv2
from typing import List


class FourierTransform:
    __slots__ = ["input_arr", "__rgb_arr"]

    def __init__(self, input_arr: List[np.ndarray]):
        self.input_arr = input_arr
        self.__rgb_arr = self.__extract_rgb(self.input_arr)

    def __extract_rgb(self, image_arr: List[np.ndarray]) -> List[np.ndarray]:
        is_four_dimentional = len(image_arr.shape) == 4

        if not is_four_dimentional:
            raise ValueError(
                "Input array must have [collection_num, row, column, channels]"
            )

        rgbs = [None] * len(image_arr)
        for i, img in enumerate(image_arr):
            rgb = img[:, :, :3]
            rgb = cv2.convertScaleAbs(rgb)
            rgbs[i] = rgb
        return rgbs

    def get_rgb_images(self):
        return self.__rgb_arr

    def generate_magnitude_spectrum(self) -> List[np.ndarray]:
        magnitude_spectrums = [None] * len(self.__rgb_arr)

        for i, img in enumerate(self.__rgb_arr):
            # Calculate the magnitude spectrum for each color channel
            f_r = np.fft.fft2(img[:, :, 0])
            f_g = np.fft.fft2(img[:, :, 1])
            f_b = np.fft.fft2(img[:, :, 2])
            mag_r = 20 * np.log(np.abs(np.fft.fftshift(f_r)))
            mag_g = 20 * np.log(np.abs(np.fft.fftshift(f_g)))
            mag_b = 20 * np.log(np.abs(np.fft.fftshift(f_b)))
            # Combine the magnitude spectra into a single grayscale image
            magnitude_spectrum = cv2.merge((mag_r, mag_g, mag_b))
            magnitude_spectrum = np.max(magnitude_spectrum, axis=2)
            magnitude_spectrums[i] = magnitude_spectrum
        return magnitude_spectrums

    def generate_hpf_images(self) -> List[np.ndarray]:
        hpf_images = [None] * len(self.__rgb_arr)

        for i, img in enumerate(self.__rgb_arr):
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Apply Fourier transform to image
            f = np.fft.fft2(gray_img)

            # # Shift zero frequency component to center
            fshift = np.fft.fftshift(f)

            # # Set low frequency coefficients to zero
            rows, cols = gray_img.shape
            crow, ccol = rows / 2, cols / 2
            crow = int(crow)
            ccol = int(ccol)
            fshift[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 0

            # Shift zero frequency component back to origin
            f_ishift = np.fft.ifftshift(fshift)
            # Invert Fourier transform to obtain filtered image
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = img_back.astype(np.uint8)
            hpf_images[i] = img_back
        return hpf_images
