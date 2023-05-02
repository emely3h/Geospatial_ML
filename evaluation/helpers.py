import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from tensorflow.keras.utils import to_categorical

from evaluation.evaluation_metrics import EvaluationMetrics, ConfusionMatrix
from prepare_data.create_mask import create_physical_mask


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


def _display_image(display_list):
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
        _display_image([sample_image, sample_mask, sample_pred])


# classes using slots don't have a __dict__ method by default
def _calculate_save_metrics(y_true, y_pred, dataset, saving_path):
    cm_labels = []
    for label in range(0, 3):
        print(f'Calculate {dataset} metrics for label {label}...')
        m_fp = tf.keras.metrics.FalsePositives()
        m_fp.update_state(y_true[..., label].flatten(), y_pred[..., label].flatten())
        fp = m_fp.result().numpy()

        m_fn = tf.keras.metrics.FalseNegatives()
        m_fn.update_state(y_true[..., label].flatten(), y_pred[..., label].flatten())
        fn = m_fn.result().numpy()

        m_tp = tf.keras.metrics.TruePositives()
        m_tp.update_state(y_true[..., label].flatten(), y_pred[..., label].flatten())
        tp = m_tp.result().numpy()

        m_tn = tf.keras.metrics.TrueNegatives()
        m_tn.update_state(y_true[..., label].flatten(), y_pred[..., label].flatten())
        tn = m_tn.result().numpy()
        cm_labels.append(ConfusionMatrix(tn, fp, fn, tp))
    metrics = EvaluationMetrics(cm_labels[0], cm_labels[1], cm_labels[2])

    with open(saving_path, "wb") as file:
        print(f'saving metrics...\n')
        pickle.dump(metrics, file)
    return metrics


def calc_save_metrics_pred(dataset: str, num_tiles: int, experiment: str, y_true: np.memmap, model: int = None):
    y_pred = np.memmap(f"../models/{experiment}/predictions/pred_{dataset}_{model}.npy", mode="r",
                       shape=(num_tiles, 256, 256, 3), dtype=np.float32)
    y_pred = np.argmax(y_pred, axis=-1)

    y_pred = to_categorical(y_pred, num_classes=3, dtype="uint8")
    y_true = to_categorical(y_true, num_classes=3, dtype="uint8")
    saving_path = f"../metrics/{experiment}/metrics_{dataset}_{model}.pkl"
    return _calculate_save_metrics(y_true, y_pred, dataset, saving_path)


def calc_save_metrics_data(y_mask, x_input, dataset):
    y_true = to_categorical(y_mask, num_classes=3)
    x_input = np.copy(x_input)
    y_pred = create_physical_mask(x_input)
    saving_path = f"../metrics/data_exploration/metrics_{dataset}.pkl"
    return _calculate_save_metrics(y_true, y_pred, dataset, saving_path)


def _load_metrics(num_models, experiment):
    metrics = []
    path = f'../metrics/{experiment}'
    for model in range(0, num_models):
        for dataset in ['train', 'val', 'test']:
            file = f'metrics_{dataset}_{model}.pkl'
            with open(os.path.join(path, file), "rb") as f:
                m = pickle.load(f)
                m_dict = m.__dict__
                del m_dict['cm_invalid']
                del m_dict['cm_valid']
                del m_dict['cm_land']
                metrics.append((m_dict, file))
    return metrics


def load_metrics_into_df(num_models, experiment, title):
    metrics = _load_metrics(num_models, experiment)
    metrics_dicts = []
    metric_names = []

    for metric in metrics:
        metrics_dicts.append(metric[0])
        name_split = metric[1].split('_')
        metric_names.append(f'{name_split[1]}_{name_split[2][0]}')

    df = pd.DataFrame(metrics_dicts)
    df.index = metric_names
    df = df.transpose()

    df = df.style.set_table_attributes("style='display:inline'").set_caption(
        title).set_table_styles([{
        'selector': 'caption',
        'props': [
            ('color', 'white'),
            ('font-size', '20px')
        ]
    }])
    df.index.name = 'Evaluation Metrics'

    return df
