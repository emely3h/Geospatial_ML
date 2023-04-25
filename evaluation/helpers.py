import numpy as np
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import pandas as pd


class ConfusionMatrix:
    __slots__ = [
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives"
    ]

    def __init__(self, tp=0, fp=0, tn=0, fn=0):
        self.true_positives = tp
        self.false_positives = fp
        self.true_negatives = tn
        self.false_negatives = fn

    def add_chunk(self, conf_matrix):
        self.true_positives += conf_matrix.true_positives
        self.false_positives += conf_matrix.false_positives
        self.true_negatives += conf_matrix.true_negatives
        self.false_negatives += conf_matrix.false_negatives


def get_confusion_matrix(y_true, y_pred, label):
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

    return ConfusionMatrix(tp=true_positives, fp=false_positives, tn=true_negatives, fn=false_negatives)


def get_intersections_unions(y_true: np.ndarray, y_pred: np.ndarray):
    intersections = []
    unions = []
    for label in range(3):
        y_true_f = keras.backend.flatten(y_true[..., label])
        y_pred_f = keras.backend.flatten(y_pred[..., label])
        intersection = keras.backend.sum(y_true_f * y_pred_f)
        union = (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) - intersection)

        intersections.append(intersection)
        unions.append(union)

    return intersections, unions


def save_metrics(metrics_train, metrics_val, metrics_test, saving_path, count):
    with open(f"{saving_path}/metrics_test_{count}.pkl", "wb") as file:
        pickle.dump(metrics_test, file)
    with open(f"{saving_path}/metrics_val_{count}.pkl", "wb") as file:
        pickle.dump(metrics_val, file)
    with open(f"{saving_path}/metrics_train_{count}.pkl", "wb") as file:
        pickle.dump(metrics_train, file)


def plot_loss_acc(plots, y_scale, model_history, scale):
    loss = model_history['loss']
    x = [i for i in range(len(model_history['loss']))]
    val_loss = model_history['val_loss']
    acc = model_history['accuracy']
    val_acc = model_history['val_accuracy']

    plt.figure(figsize=(10, 6))
    if 'loss' in plots:
        print(f'Min training loss: {min(model_history["loss"])}')
        plt.scatter(x, loss, s=10, label='Training Loss')

    if 'accuracy' in plots:
        print(f'Max training accuracy: {max(model_history["accuracy"])}')
        plt.scatter(x, acc, s=10, label='Training Accuracy')

    if 'val_loss' in plots:
        print(f'Min validation loss: {min(model_history["val_loss"])}')
        plt.scatter(x, val_loss, s=10, label='Validation Loss')

    if 'val_accuracy' in plots:
        print(f'Max validation accuracy: {max(model_history["val_accuracy"])}')
        plt.scatter(x, val_acc, s=10, label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.ylim(scale)
    plt.legend()
    plt.show()


def display_image(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']
    colors = ['red', 'blue', 'yellow']
    cmap = ListedColormap(colors)
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        if i != 0:
            if i == 2:
                display_list[i] = np.argmax(display_list[i], axis=-1)

            plt.imshow(display_list[i], cmap=cmap)
        else:
            plt.imshow(keras.utils.array_to_img(display_list[i]))
    plt.show()


def display(list_train, list_mask, list_pred):
    for idx, img_train in enumerate(list_train):
        sample_image, sample_mask, sample_pred = list_train[idx], list_mask[idx], list_pred[idx]
        sample_image = sample_image[..., :4]
        display_image([sample_image, sample_mask, sample_pred])


def _load_metrics(path):
    metrics = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), "rb") as f:
            print(f'load {file}')
            metrics.append((pickle.load(f), file))

    return metrics


# classes using slots don't have a __dict__ method by default
def _metric_to_dict(metric):
    return {
        "mean_jaccard": float(metric.mean_jaccard),
        "jaccard_invalid": float(metric.jaccard_invalid),
        "jaccard_valid": float(metric.jaccard_valid),
        "jaccard_land": float(metric.jaccard_land),

        "f1_invalid": metric.f1_invalid,
        "f1_valid": metric.f1_valid,
        "f1_land": metric.f1_land,

        "precision_invalid": metric.precision_invalid,
        "precision_valid": metric.precision_valid,
        "precision_land": metric.precision_land,

        "sensitivity_invalid": metric.sensitivity_recall_invalid,
        "sensitivity_valid": metric.sensitivity_recall_valid,
        "sensitivity_land": metric.sensitivity_recall_land,

        "specificy_invalid": metric.specificy_invalid,
        "specificy_valid": metric.specificy_valid,
        "specificy_land": metric.specificy_land,
    }


def load_metrics_into_df(experiment):
    metrics = _load_metrics(f'../metrics/{experiment}')
    metrics_dicts = []
    metrics_titles = []

    for metric in metrics:
        metrics_dicts.append(_metric_to_dict(metric[0]))
        name_split = metric[1].split('_')
        metrics_titles.append(f'{name_split[1]}_{name_split[2][0]}')

    df = pd.DataFrame(metrics_dicts).transpose()
    df.columns = metrics_titles
    return df
