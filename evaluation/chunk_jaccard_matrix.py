import numpy as np
from evaluation.helpers import get_intersections_unions, get_confusion_matrix, ConfusionMatrix
from tensorflow.keras.utils import to_categorical


class ChunkJaccardMatrix:
    """
    This class calculates and the confusion matrix and the jaccard index for each label in chunks.
    """

    __slots__ = [
        "conf_matrix_land",
        "conf_matrix_valid",
        "conf_matrix_invalid",
        "intersections",
        "unions",
        "current_chunk_index",
        "num_chunks",
        "chunk_size",
        "y_true",
        "y_pred",
        "mean_jaccard",
        "jaccard_invalid",
        "jaccard_valid",
        "jaccard_land",

    ]

    def __init__(self, y_true: np.memmap, y_pred: np.memmap, chunk_size=1000):
        self.conf_matrix_land = ConfusionMatrix()
        self.conf_matrix_valid = ConfusionMatrix()
        self.conf_matrix_invalid = ConfusionMatrix()

        self.intersections = [0, 0, 0]
        self.unions = [0, 0, 0]

        self.current_chunk_index = 0
        self.num_chunks = y_true.shape[0] // chunk_size
        self.chunk_size = chunk_size
        self.y_true = y_true
        self.y_pred = y_pred

        self.mean_jaccard = 0
        self.jaccard_invalid = 0
        self.jaccard_valid = 0
        self.jaccard_land = 0

        self.calculate_jaccard_matrix()

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_chunk_index > self.num_chunks:
            raise StopIteration

        start = self.current_chunk_index * self.chunk_size
        end = start + self.chunk_size
        if self.current_chunk_index == self.num_chunks:
            end = self.y_true.shape[0]
        print(f'batch {self.current_chunk_index} {datetime.now()} start: {start} end: {end}')
        pred_chunk = np.argmax(self.y_pred[start:end], axis=-1)

        pred_chunk = to_categorical(pred_chunk, num_classes=3)

        y_true_chunk = np.copy(self.y_true[start:end])
        y_true_chunk = to_categorical(y_true_chunk, num_classes=3)

        chunk_intersections, chunk_unions = get_intersections_unions(y_true_chunk, pred_chunk)

        for label in range(3):
            self.intersections[label] += chunk_intersections[label]
            self.unions[label] += chunk_unions[label]

        self.conf_matrix_invalid.add_chunk(get_confusion_matrix(y_true_chunk, pred_chunk, 0))
        self.conf_matrix_valid.add_chunk(get_confusion_matrix(y_true_chunk, pred_chunk, 1))
        self.conf_matrix_land.add_chunk(get_confusion_matrix(y_true_chunk, pred_chunk, 2))

        self.current_chunk_index += 1

    def calculate_jaccard_matrix(self):
        for num_chunks in self:
            pass
        self.calculate_jaccards()

    def calculate_jaccards(self):
        jaccards = []
        total_intersection = 0
        total_union = 0
        for label in range(3):
            jaccards.append((self.intersections[label] + 1.0) / (self.unions[label] + 1.0))
            total_intersection += self.intersections[label]
            total_union += self.unions[label]

        self.jaccard_invalid = jaccards[0]
        self.jaccard_valid = jaccards[1]
        self.jaccard_land = jaccards[2]
        self.mean_jaccard = (total_intersection + 1.0) / (total_union + 1.0)