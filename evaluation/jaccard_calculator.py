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
        "intersections_sum",
        "unions_sum",
        "intersections",
        "unions",
        "jaccard_indexes",
        "mean_jaccard",
        "labels"
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
        self.intersections_sum = 0
        self.unions_sum = 0
        self.intersections = np.zeros(num_classes)
        self.unions = np.zeros(num_classes)
        self.jaccard_indexes = np.zeros(num_classes)
        self.mean_jaccard = 0
        self.labels = ["invalid", "valid", "land"]

        self.calculate_mean_jaccard_index()

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_chunk_index >= self.num_chunks:
            raise StopIteration
        print(f"Chunk No.{self.current_chunk_index}")
        x_input_chunk, y_mask_chunk = self.copy_data_to_array()
        print(f"shape x_input_chunk: {x_input_chunk.shape}, shape y_mask_chunk: {y_mask_chunk.shape}")
        pred_physical = create_physical_mask(x_input_chunk)
        y_true = to_categorical(y_mask_chunk, num_classes=self.num_classes)
        print(f"shape y_true: {y_true.shape}, shape pred_physical: {pred_physical.shape}")
        self.intersection_union(y_true=y_true, y_pred=pred_physical)

        print("\n")
        self.current_chunk_index += 1

    def copy_data_to_array(self) -> Tuple[np.ndarray, np.ndarray]:
        start_index = self.current_chunk_index * self.chunk_size
        end_index = start_index + self.chunk_size
        print(f'copying chunk from mmap [{start_index}:{end_index}]')
        if end_index > self.tiles:
            end_index = self.tiles
        chunk_size = end_index - start_index
        x_input = np.zeros((chunk_size, 256, 256, 5), dtype=np.float32)
        np.copyto(x_input, self.split_x[start_index:end_index])
        y_mask = np.zeros((chunk_size, 256, 256), dtype=np.float32)
        np.copyto(y_mask, self.split_y[start_index:end_index])
        return x_input, y_mask

    def intersection_union(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        for label in range(self.num_classes):
            y_true_f = keras.backend.flatten(y_true[..., label])
            y_pred_f = keras.backend.flatten(y_pred[..., label])
            intersection = keras.backend.sum(y_true_f * y_pred_f)
            union = (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) - intersection)

            print(f"chunk_jaccard_{self.labels[label]}: {(intersection + 1.0) / (union + 1.0)}, chunk_intersection: {intersection}, chunk_union: {union}")
            self.intersections[label] += intersection
            self.unions[label] += union
            self.intersections_sum += intersection
            self.unions_sum += union

    def calculate_mean_jaccard_index(self):
        for num_chunks in self:
            pass
        print("Summary \n")
        print("total tiles: ", self.tiles)
        print("chunk size: ", self.chunk_size)

        for label in range(self.num_classes):
            print("\nclass name: ", self.labels[label])
            print("intersection: ", self.intersections[label])
            print("union: ", self.unions[label])
            self.jaccard_indexes[label] = (self.intersections[label] + 1.0) / (
                    self.unions[label] + 1.0
            )
            print("jaccard_index: ", self.jaccard_indexes[label])

        print("\nsum of intersections: ", self.intersections_sum)
        print("sum of unions: ", self.unions_sum)
        self.mean_jaccard = (self.intersections_sum + 1.0) / (self.unions_sum + 1.0)
        print("mean_jaccard: ", self.mean_jaccard)

    def get_jaccards(self) -> dict:
        return {
            self.labels[0]: self.jaccard_indexes[0],
            self.labels[1]: self.jaccard_indexes[1],
            self.labels[2]: self.jaccard_indexes[2],
            "mean_jaccard": self.mean_jaccard
        }
