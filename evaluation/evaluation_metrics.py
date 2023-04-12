import numpy as np
from tensorflow import keras
import pickle


class EvaluationMetrics:
    """
    This class calculates and summarizes evaluation metrics based on the predicted and true labels.
    """

    __slots__ = [
        "jaccard",
        "jaccard_physical",
        "conf_matrix_land",
        "conf_matrix_valid",
        "conf_matrix_invalid",
        "precision_land",
        "sensitivity_recall_land",
        "specificy_land",
        "precision_valid",
        "sensitivity_recall_valid",
        "specificy_valid",
        "precision_invalid",
        "sensitivity_recall_invalid",
        "specificy_invalid",
        "f1_land",
        "f1_invalid",
        "f1_valid",
    ]

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_physical: np.ndarray):
        self.jaccard = self.jaccard_coef(y_true, y_pred)
        self.jaccard_physical = self.jaccard_coef(y_true, y_physical)

        self.conf_matrix_land = self.confusion_matrix(y_true, y_pred, 2)
        self.conf_matrix_valid = self.confusion_matrix(y_true, y_pred, 1)
        self.conf_matrix_invalid = self.confusion_matrix(y_true, y_pred, 0)

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

    def jaccard_coef(self, y_true, y_pred):
        y_true_f = keras.backend.flatten(y_true)
        y_pred_f = keras.backend.flatten(y_pred)

        intersection = keras.backend.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (
            keras.backend.sum(y_true_f)
            + keras.backend.sum(y_pred_f)
            - intersection
            + 1.0
        )  # todo reason for +1?

    def jaccard_rounding_issue(self, y_true, y_pred):
        # revert one hot encoding => binary tensor [0, 0, 1] back to label [2] (3D array to 2D array)
        label_map_true = np.argmax(y_true, axis=-1)
        label_map_pred = np.argmax(y_pred, axis=-1)
        # convert 2D array into 1D array
        flatten_true = np.reshape(label_map_true, (-1,))
        flatten_pred = np.reshape(label_map_pred, (-1,))
        # one hot encoding
        one_hot_true = np.eye(3)[flatten_true]
        one_hot_pred = np.eye(3)[flatten_pred]
        # calculate intersection (A geschnitten B)
        intersection = np.sum(one_hot_true * one_hot_pred)
        # calculate union (a u B, A vereint B)
        union = len(one_hot_true) + len(one_hot_pred) - intersection
        # return jaccard coefficient
        return (intersection + 1) / (union + 1)

    def confusion_matrix(self, y_true, y_pred, label):
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        # revert one hot encoding => binary tensor [0, 0, 1] back to label [2] (3D array to 2D array)
        label_map_true = np.argmax(y_true, axis=-1)
        label_map_pred = np.argmax(y_pred, axis=-1)
        # convert 2D array into 1D array
        flatten_true = np.reshape(label_map_true, (-1,))
        flatten_pred = np.reshape(label_map_pred, (-1,))

        tp_mask = (flatten_true == flatten_pred) & (flatten_true == label)
        true_positives = np.count_nonzero(tp_mask)

        fn_mask = (flatten_true == label) & (flatten_pred != label)
        false_negatives = np.count_nonzero(fn_mask)

        fp_mask = (flatten_true != label) & (flatten_pred == label)
        false_positives = np.count_nonzero(fp_mask)

        tn_mask = (flatten_true != label) & (flatten_pred != label)
        true_negatives = np.count_nonzero(tn_mask)
        print(
            f"print confusion matrix \n true_positives: {true_positives}, false_positives: {false_positives}, true_negatives: {true_negatives}, false_negatives: {false_negatives}"
        )
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
        }

    def precision(self, conf_matrix):
        if conf_matrix["true_positives"] == 0 or conf_matrix["false_positives"] == 0:
            print(
                f"Precision 0 values: {(conf_matrix['true_positives'])} {conf_matrix['false_positives']}"
            )
            return 0
        return conf_matrix["true_positives"] / (
            conf_matrix["true_positives"] + conf_matrix["false_positives"]
        )

    def sensitivity_recall(self, conf_matrix):
        if conf_matrix["true_positives"] == 0 or conf_matrix["false_negatives"] == 0:
            print(
                f"Sensitivity 0 values: {(conf_matrix['true_positives'])} {conf_matrix['false_negatives']}"
            )
            return 0
        return conf_matrix["true_positives"] / (
            conf_matrix["true_positives"] + conf_matrix["false_negatives"]
        )

    def negative_predictive(self, conf_matrix):
        if conf_matrix["true_negatives"] == 0 or conf_matrix["false_negatives"] == 0:
            print(
                f"negative_predictive Error 0 values: {conf_matrix['true_negatives']} {conf_matrix['false_negatives']}"
            )
            return 0
        return conf_matrix["true_negatives"] / (
            conf_matrix["true_negatives"] + conf_matrix["false_negatives"]
        )

    def specificy(self, conf_matrix):
        if conf_matrix["true_negatives"] == 0 or conf_matrix["false_positives"] == 0:
            print(
                f"specificy 0 values: {conf_matrix['true_negatives']} {(conf_matrix['false_positives'])}"
            )
            return 0
        return conf_matrix["true_negatives"] / (
            conf_matrix["true_negatives"] + conf_matrix["false_positives"]
        )

    def f1_scores(self, conf_matrix):
        prec = self.precision(conf_matrix)
        recall = self.sensitivity_recall(conf_matrix)
        if prec + recall == 0:
            print("f1 score 0")
            return 0
        return 2 * prec * recall / (prec + recall)

    def print_metrics(self):
        print(f"jaccard index: {self.jaccard} \n")
        print(f"physical jaccard: {self.jaccard_physical} \n")

        print(f"precision_land: {self.precision_land}")
        print(f"precision_valid: {self.precision_valid}")
        print(f"precision_invalid: {self.precision_invalid} \n")

        print(f"recall_invalid_land: {self.sensitivity_recall_land}")
        print(f"recall_invalid_land: {self.sensitivity_recall_valid}")
        print(f"recall_invalid_land: {self.sensitivity_recall_invalid} \n")

        print(f"specificy_invalid_land: {self.specificy_land}")
        print(f"specificy_invalid_valid: {self.specificy_valid}")
        print(f"specificy_invalid_invalid: {self.specificy_invalid} \n")

        print(f"f1_land: {self.f1_land}")
        print(f"f1_invalid: {self.f1_invalid}")
        print(f"f1_valid: {self.f1_valid}")

    # todo add pixel accuracy


def save_metrics(metrics_train, metrics_val, metrics_test, saving_path, count):
    with open(f"{saving_path}/metrics_test_{count}.pkl", "wb") as file:
        pickle.dump(metrics_train, file)
    with open(f"{saving_path}/metrics_val_{count}.pkl", "wb") as file:
        pickle.dump(metrics_val, file)
    with open(f"{saving_path}/metrics_train_{count}.pkl", "wb") as file:
        pickle.dump(metrics_test, file)


if __name__ == "__main__":
    import time

    start_time = time.time()
    # create sample arrays
    y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    y_pred = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    y_physical = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    # create instance of EvaluationMetrics
    metrics = EvaluationMetrics(y_true, y_pred, y_physical)

    # print results
    print(f"Jaccard Coefficient: {metrics.jaccard}")
    print(f"Jaccard Coefficient (physical): {metrics.jaccard_physical}")
    print(f"Confusion Matrix (land): {metrics.conf_matrix_land}")
    print(f"Confusion Matrix (valid): {metrics.conf_matrix_valid}")
    print(f"Confusion Matrix (invalid): {metrics.conf_matrix_invalid}")
    print(f"Precision (land): {metrics.precision_land}")
    print(f"Sensitivity and Recall (land): {metrics.sensitivity_recall_land}")
    print(f"Specificity (land): {metrics.specificy_land}")
    print(f"Precision (valid): {metrics.precision_valid}")
    print(f"Sensitivity and Recall (valid): {metrics.sensitivity_recall_valid}")
    print(f"Specificity (valid): {metrics.specificy_valid}")
    print(f"Precision (invalid): {metrics.precision_invalid}")
    print(f"Sensitivity and Recall (invalid): {metrics.sensitivity_recall_invalid}")
    print(f"Specificity (invalid): {metrics.specificy_invalid}")
    print(f"F1 Score (land): {metrics.f1_land}")
    print(f"F1 Score (valid): {metrics.f1_valid}")
    print(f"F1 Score (invalid): {metrics.f1_invalid}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time} seconds")
