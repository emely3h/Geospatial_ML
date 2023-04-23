import numpy as np
from evaluation.chunk_jaccard_matrix import ChunkJaccardMatrix


class EvaluationMetricsTotal:
    """
    This class calculates and summarizes evaluation metrics based on the predicted and true labels.
    """

    __slots__ = [
        "chunk_jaccard_matrix",
        "mean_jaccard",
        "jaccard_invalid",
        "jaccard_valid",
        "jaccard_land",

        "conf_matrix_invalid",
        "conf_matrix_valid",
        "conf_matrix_land",

        "precision_invalid",
        "precision_valid",
        "precision_land",

        "sensitivity_recall_invalid",
        "sensitivity_recall_valid",
        "sensitivity_recall_land",

        "specificy_invalid",
        "specificy_valid",
        "specificy_land",

        "f1_invalid",
        "f1_valid",
        "f1_land",
    ]

    def __init__(self, y_true: np.memmap, y_pred: np.memmap):

        self.chunk_jaccard_matrix = ChunkJaccardMatrix(y_true, y_pred)

        self.mean_jaccard = self.chunk_jaccard_matrix.mean_jaccard
        self.jaccard_invalid = self.chunk_jaccard_matrix.jaccard_invalid
        self.jaccard_valid = self.chunk_jaccard_matrix.jaccard_valid
        self.jaccard_land = self.chunk_jaccard_matrix.jaccard_land

        self.conf_matrix_invalid = self.chunk_jaccard_matrix.conf_matrix_invalid
        self.conf_matrix_valid = self.chunk_jaccard_matrix.conf_matrix_valid
        self.conf_matrix_land = self.chunk_jaccard_matrix.conf_matrix_land

        self.precision_land = self.precision(self.conf_matrix_land)
        self.sensitivity_recall_land = self.sensitivity_recall(self.conf_matrix_land)
        self.specificy_land = self.specificy(self.conf_matrix_land)

        self.precision_valid = self.precision(self.conf_matrix_valid)
        self.sensitivity_recall_valid = self.sensitivity_recall(self.conf_matrix_valid)
        self.specificy_valid = self.specificy(self.conf_matrix_valid)

        self.precision_invalid = self.precision(self.conf_matrix_invalid)
        self.sensitivity_recall_invalid = self.sensitivity_recall(
            self.conf_matrix_invalid
        )
        self.specificy_invalid = self.specificy(self.conf_matrix_invalid)

        self.f1_land = self.f1_scores(self.conf_matrix_land)
        self.f1_invalid = self.f1_scores(self.conf_matrix_invalid)
        self.f1_valid = self.f1_scores(self.conf_matrix_valid)

    def precision(self, conf_matrix):
        if conf_matrix.true_positives == 0 or conf_matrix.false_positives == 0:
            print(
                f"Precision 0 values: {(conf_matrix.true_positives)} {conf_matrix.false_positives}"
            )
            return 0
        return conf_matrix.true_positives / (
                conf_matrix.true_positives + conf_matrix.false_positives
        )

    def sensitivity_recall(self, conf_matrix):
        if conf_matrix.true_positives == 0 or conf_matrix.false_negatives == 0:
            print(
                f"Sensitivity 0 values: {(conf_matrix.true_positives)} {conf_matrix.false_negatives}"
            )
            return 0
        return conf_matrix.true_positives / (
                conf_matrix.true_positives + conf_matrix.false_negatives
        )

    def negative_predictive(self, conf_matrix):
        if conf_matrix.true_negatives == 0 or conf_matrix.false_negatives == 0:
            print(
                f"negative_predictive Error 0 values: {conf_matrix.true_negatives} {conf_matrix.false_negatives}"
            )
            return 0
        return conf_matrix.true_negatives / (
                conf_matrix.true_negatives + conf_matrix.false_negatives
        )

    def specificy(self, conf_matrix):
        if conf_matrix.true_negatives == 0 or conf_matrix.false_positives == 0:
            print(
                f"specificy 0 values: {conf_matrix.true_negatives} {(conf_matrix.false_positives)}"
            )
            return 0
        return conf_matrix.true_negatives / (
                conf_matrix.true_negatives + conf_matrix.false_positives
        )

    def f1_scores(self, conf_matrix):
        prec = self.precision(conf_matrix)
        recall = self.sensitivity_recall(conf_matrix)
        if prec + recall == 0:
            print("f1 score 0")
            return 0
        return 2 * prec * recall / (prec + recall)

    def print_metrics(self):
        print(f"mean jaccard index: {self.mean_jaccard}")
        print(f"invalid jaccard index: {self.jaccard_invalid}")
        print(f"valid jaccard index: {self.jaccard_valid}")
        print(f"land jaccard index: {self.jaccard_land} \n")

        print(f"precision_invalid: {self.precision_invalid} \n")
        print(f"precision_valid: {self.precision_valid}")
        print(f"precision_land: {self.precision_land}")

        print(f"recall_invalid: {self.sensitivity_recall_invalid} \n")
        print(f"recall_valid: {self.sensitivity_recall_valid}")
        print(f"recall_land: {self.sensitivity_recall_land}")

        print(f"specificy_invalid: {self.specificy_invalid} \n")
        print(f"specificy_valid: {self.specificy_valid}")
        print(f"specificy_land: {self.specificy_land}")

        print(f"f1_valid: {self.f1_valid}")
        print(f"f1_invalid: {self.f1_invalid}")
        print(f"f1_land: {self.f1_land}")
