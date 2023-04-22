import numpy as np
from helpers import get_intersections_unions, get_confusion_matrix


class ChunkJaccardMatrix:
    """
    This class calculates and the confusion matrix and the jaccard index for each label in chunks.
    """

    __slots__ = [
        "conf_matrix_land",
        "conf_matrix_valid",
        "conf_matrix_invalid",
        "invalid_inter_union",
        "valid_inter_union",
        "land_inter_union",
        "current_chunk_index",
        "num_chunks",
        "num_tiles",
        "mean_jaccard",
        "jaccard_invalid",
        "jaccard_valid",
        "jaccard_land",

    ]

    def __init__(self, y_true: np.memmap, y_pred: np.memmap, num_tiles: int, chunk_size=1000):
        self.conf_matrix_land = self.confusion_matrix(y_true, y_pred, 2)
        self.conf_matrix_valid = self.confusion_matrix(y_true, y_pred, 1)
        self.conf_matrix_invalid = self.confusion_matrix(y_true, y_pred, 0)

        self.invalid_inter_union = (0, 0)
        self.valid_inter_union = (0, 0)
        self.land_inter_union = (0, 0)

        self.current_chunk_index = 0
        self.num_chunks = num_tiles // chunk_size
        self.num_tiles = num_tiles

        self.mean_jaccard = 0
        self.jaccard_invalid = 0
        self.jaccard_valid = 0
        self.jaccard_land = 0

        self.calculate_jaccard_matrix()

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_chunk_index >= self.num_chunks:
            raise StopIteration
        start = self.current_chunk_index * self.chunk_size
        end = start + self.chunk_size
        if self.current_chunk_index == self.num_chunks:
            end = self.tiles

        pred_chunk = np.argmax(self.y_true[start:end], axis=-1)

        y_true_chunk = np.copy(self.y_true[start:end])

        chunk_intersections, chunk_unions = get_intersections_unions(y_true_chunk, pred_chunk)

        for label in range(3):
            self.invalid_inter_union[label] += chunk_intersections[label]
            self.invalid_inter_union[label] += chunk_unions[label]

        chunk_confusion_matrix = get_confusion_matrix(y_true_chunk, pred_chunk)
        self.true_positives += chunk_confusion_matrix["true_positives"]
        self.false_positives += chunk_confusion_matrix["false_positives"]
        self.true_negatives += chunk_confusion_matrix["true_negatives"]
        self.false_negatives += chunk_confusion_matrix["false_negatives"]

        self.current_chunk_index += 1

    def calculate_jaccard_matrix(self):
        for num_chunks in self:
            pass
        self.calculate_jaccards()

    def calculate_jaccards(self):
        self.jaccard_invalid = (self.invalid_inter_union[0] + 1.0) / (self.invalid_inter_union[1] + 1.0)
        self.jaccard_valid = (self.valid_inter_union[0] + 1.0) / (self.valid_inter_union[1] + 1.0)
        self.jaccard_land = (self.land_inter_union[0] + 1.0) / (self.land_inter_union[1] + 1.0)
        total_intersection = self.invalid_inter_union[0] + self.valid_inter_union[0] + self.land_inter_union[0]
        total_union = self.invalid_inter_union[1] + self.valid_inter_union[1] + self.land_inter_union[1]
        self.mean_jaccard = (total_intersection + 1.0) / (total_union + 1.0)
