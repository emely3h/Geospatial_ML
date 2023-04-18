import numpy as np
from typing import Tuple
from keras.utils import to_categorical
from tensorflow import keras
from prepare_data.create_mask import create_physical_mask


class JaccardIndexCalculator:

    __slots__ = [
        "split_x",
        "split_y",
        "tiles",
        "num_classes",
        "chunk_size",
        "num_chunks",
        "current_chunk_index",
        "all_intersections",
        "all_unions",
        "each_intersection",
        "each_union",
        "each_jaccard_index",
    ]

    def __init__(
        self,
        split_x: np.memmap,
        split_y: np.memmap,
        tiles: int,
        num_classes: int = 3,
        chunk_size: int = 1000,
    ):
        self.split_x = split_x
        self.split_y = split_y
        self.tiles = tiles
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.num_chunks = int(np.ceil(tiles / chunk_size))
        self.current_chunk_index = 0
        self.all_intersections = 0
        self.all_unions = 0
        self.each_intersection = np.zeros(num_classes + 1)
        self.each_union = np.zeros(num_classes + 1)
        self.each_jaccard_index = np.zeros(num_classes + 1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_chunk_index >= self.num_chunks:
            raise StopIteration
        print(f"Chunk No.{self.current_chunk_index}")
        print("Step1: Copying memmap to array")
        x_input_chunk, y_mask_chunk = self.copy_data_to_array()
        print("Step2: Creating Physical Mask")
        pred_physical = create_physical_mask(x_input_chunk)
        print("Step3: Hot Encoding")
        y_true = to_categorical(y_mask_chunk, num_classes=self.num_classes)
        print("Step4: Calculate Intersection and Union")
        self.intersection_union(y_true=y_true, y_pred=pred_physical)

        print("\n")
        self.current_chunk_index += 1

    def copy_data_to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        start_index = self.current_chunk_index * self.chunk_size
        end_index = start_index + self.chunk_size
        if end_index > self.tiles:
            end_index = self.tiles
        chunk_size = end_index - start_index
        print("chunk_size:", chunk_size)
        x_input = np.zeros((chunk_size, 256, 256, 5), dtype=np.float32)
        np.copyto(x_input, self.split_x[start_index:end_index])
        y_mask = np.zeros((chunk_size, 256, 256), dtype=np.float32)
        np.copyto(y_mask, self.split_y[start_index:end_index])
        return x_input, y_mask

    def intersection_union(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        for i in range(self.num_classes + 1):
            if not i == 3:
                y_true_f = keras.backend.flatten(y_true[..., i])
                y_pred_f = keras.backend.flatten(y_pred[..., i])
            else:
                y_true_f = keras.backend.flatten(y_true)
                y_pred_f = keras.backend.flatten(y_pred)
            intersection = keras.backend.sum(y_true_f * y_pred_f)
            union = (
                keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) - intersection
            )
            print("class: ", i)
            print("intersection: ", intersection)
            print("union: ", union)

            self.each_intersection[i] += intersection
            self.each_union[i] += union
            if not i == 3:
                self.all_intersections += intersection
                self.all_unions += union
        return

    def calculate_mean_jaccard_index(self) -> dict:
        for num_chunks in self:
            pass
        print("Summary \n")
        print("total tiles: ", self.tiles)
        print("chunk size: ", self.chunk_size)

        for i in range(self.num_classes + 1):
            class_name = ["invalid", "valid", "land", "all_original"]
            print("class name: ", class_name[i])
            print("each_intersection: ", self.each_intersection[i])
            print("each_union: ", self.each_union[i])
            self.each_jaccard_index[i] = (self.each_intersection[i] + 1.0) / (
                self.each_union[i] + 1.0
            )
            print("each_jaccard_index: ", self.each_jaccard_index[i])

        print("total of intersections: ", self.all_intersections)
        print("total of unions: ", self.all_unions)
        all_mean_jaccard = (self.all_intersections + 1.0) / (self.all_unions + 1.0)
        print("mean of Jaccard Index: ", all_mean_jaccard)

        return {
            "mean_jaccard_sum": all_mean_jaccard,
            class_name[0]: self.each_jaccard_index[0],
            class_name[1]: self.each_jaccard_index[1],
            class_name[2]: self.each_jaccard_index[2],
            class_name[3]: self.each_jaccard_index[3],
        }
