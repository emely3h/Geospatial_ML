import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from evaluation.evaluation_metrics_total import EvaluationMetricsTotal


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


def _load_metrics(path):
    metrics = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), "rb") as f:
            print(f'load {file}')
            metrics.append((pickle.load(f), file))

    return metrics


# classes using slots don't have a __dict__ method by default


def _save_metrics(metrics_train, metrics_val, metrics_test, saving_path, count):
    metrics_train = metrics_train.to_dict()
    metrics_val = metrics_val.to_dict()
    metrics_test = metrics_test.to_dict()

    with open(f"{saving_path}/metrics_test_{count}.pkl", "wb") as file:
        print(f'save test dict {count}...')
        pickle.dump(metrics_test, file)
    with open(f"{saving_path}/metrics_val_{count}.pkl", "wb") as file:
        print(f'save val dict {count}...')
        pickle.dump(metrics_val, file)
    with open(f"{saving_path}/metrics_train_{count}.pkl", "wb") as file:
        print(f'save train dict {count}...')
        pickle.dump(metrics_train, file)


def calculate_save_metrics(experiment, num_model, train_split_y, val_split_y, test_split_y, train_tiles, val_tiles,
                           test_tiles, ):
    pred_test_mmap = np.memmap(f'../models/experiment_3/predictions/pred_test_{num_model}.npy', mode="r",
                               shape=(test_tiles, 256, 256, 3), dtype=np.float32)
    pred_val_mmap = np.memmap(f'../models/experiment_3/predictions/pred_val_{num_model}.npy', mode="r",
                              shape=(val_tiles, 256, 256, 3), dtype=np.float32)
    pred_train_mmap = np.memmap(f'../models/experiment_3/predictions/pred_train_{num_model}.npy', mode="r",
                                shape=(train_tiles, 256, 256, 3), dtype=np.float32)

    print(f'start calculating test metrics: {datetime.now()}')
    metrics_test = EvaluationMetricsTotal(test_split_y, pred_test_mmap)
    print(f'start calculating validation metrics: {datetime.now()}')
    metrics_val = EvaluationMetricsTotal(val_split_y, pred_val_mmap)
    print(f'start calculating training metrics: {datetime.now()}')
    metrics_train = EvaluationMetricsTotal(train_split_y, pred_train_mmap)
    print(f'end: {datetime.now()}')

    print('Test metrics')
    metrics_test.print_metrics()
    print('\nValidation metrics')
    metrics_val.print_metrics()
    print('\nTraining metrics')
    metrics_train.print_metrics()

    _save_metrics(metrics_train, metrics_val, metrics_test, f'../metrics/{experiment}', num_model)


def load_metrics_into_df(experiment):
    metrics = _load_metrics(f'../metrics/{experiment}')
    metrics_dicts = []
    metrics_titles = []

    for metric in metrics:
        metrics_dicts.append(metric[0])
        name_split = metric[1].split('_')
        metrics_titles.append(f'{name_split[1]}_{name_split[2][0]}')

    df = pd.DataFrame(metrics_dicts).transpose()
    df.columns = metrics_titles
    return df
