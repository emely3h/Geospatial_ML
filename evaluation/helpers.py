import numpy as np
from tensorflow import keras
import pickle


def get_confusion_matrix(self, y_true, y_pred, label):
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


def get_intersections_unions(y_true: np.ndarray, y_pred: np.ndarray):
    intersections = []
    unions = []
    for label in range(3):
        y_true_f = keras.backend.flatten(y_true[..., label])
        y_pred_f = keras.backend.flatten(y_pred[..., label])
        intersection = keras.backend.sum(y_true_f * y_pred_f)
        union = (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) - intersection)

        print(f"chunk_jaccard_{label}: {(intersection + 1.0) / (union + 1.0)}, chunk_intersection: {intersection}, "
              f"chunk_union: {union}")
        intersections.append(intersection)
        unions.append(union)
    return intersections, unions


def save_metrics(metrics_train, metrics_val, metrics_test, saving_path, count):
    with open(f"{saving_path}/metrics_test_{count}.pkl", "wb") as file:
        pickle.dump(metrics_train, file)
    with open(f"{saving_path}/metrics_val_{count}.pkl", "wb") as file:
        pickle.dump(metrics_val, file)
    with open(f"{saving_path}/metrics_train_{count}.pkl", "wb") as file:
        pickle.dump(metrics_test, file)


