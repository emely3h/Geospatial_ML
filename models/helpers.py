from models.unet_model import unet_2d
from tensorflow import keras
from keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import numpy as np
from typing import Tuple
import os
from tensorflow.keras.models import load_model


def normalizing_encoding(
    encoded_x: np.ndarray, unencoded_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    print("unencoded_y shape: ", unencoded_y.shape)
    print("\nEncoding data...")
    y_one_hot = to_categorical(unencoded_y, num_classes=3)
    print("\nencoded_y shape: ", y_one_hot.shape)
    print("\nencoded_x shape: ", encoded_x.shape)
    print("\nNormalizing data...")
    x_normal = encoded_x / 255
    print("\nNormalized x shape: ", x_normal.shape)
    return x_normal, y_one_hot


def define_model(
    input_shape=(256, 256, 5),
    num_classes=3,
    optimizer="adam",
    loss=categorical_crossentropy,
    metrics=["accuracy"],
):
    model = unet_2d(input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def train_model(model, x_train, y_train, x_val, y_val, epochs=100):
    early_stop = EarlyStopping(monitor="accuracy", patience=5)
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
    )
    return history


def save_model_history(history, model_name, saving_path="../models"):
    with open(f"{saving_path}/history_{model_name}.pkl", "wb") as file_pi:
        pickle.dump(history.history, file_pi)


def make_predictions(model, x_train, x_val, x_test):
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    return pred_train, pred_val, pred_test


def save_metrics(metrics_train, metrics_val, metrics_test, saving_path, count):
    with open(f"{saving_path}/metrics_test_{count}.pkl", "wb") as file:
        pickle.dump(metrics_train, file)
    with open(f"{saving_path}/metrics_val_{count}.pkl", "wb") as file:
        pickle.dump(metrics_val, file)
    with open(f"{saving_path}/metrics_train_{count}.pkl", "wb") as file:
        pickle.dump(metrics_test, file)


def get_mean_jaccard(all_metrics):
    jaccard_array = []
    jaccard_array_physical = []
    for idx, metric in enumerate(all_metrics):
        print(metric.jaccard)
        jaccard_array.append(metric.jaccard)
        jaccard_array_physical.append(metric.jaccard_physical)

    print()
    print(f"Mean jaccard index: {sum(jaccard_array) / 10}")
    print()
    print(f"Worst index: {min(jaccard_array)}")
    print(f"Best index: {max(jaccard_array)}")
    print(f"Variance: {max(jaccard_array) - min(jaccard_array)}")

    print()
    print(f"Mean physical jaccard index: {sum(jaccard_array_physical) / 10}")
    print()
    print(f"Worst physical index: {min(jaccard_array_physical)}")
    print(f"Best physical index: {max(jaccard_array_physical)}")
    print(f"Variance: {max(jaccard_array_physical) - min(jaccard_array_physical)}")


def initialize_saved_data(
    split_x: np.ndarray, split_y: np.ndarray, tiles: int
) -> Tuple[np.ndarray, np.ndarray]:
    print("Initializing saved data...")
    x_input = np.zeros((tiles, 256, 256, 5), dtype=np.float32)
    print("x_input shape:", x_input.shape)
    print("x_min:", np.min(x_input), "x_max:", np.max(x_input))

    print("\nCopying saved data to x_input...")
    np.copyto(x_input, split_x[0:tiles])
    print("x_input shape:", x_input.shape)
    print("x_min:", np.min(x_input), "x_max:", np.max(x_input))

    print("\nInitializing y_mask...")
    y_mask = np.zeros((tiles, 256, 256), dtype=np.float32)
    print("y_mask shape:", y_mask.shape)
    print("y_min:", np.min(y_mask), "y_max:", np.max(y_mask))

    print("\nCopying saved data to y_mask...")
    np.copyto(y_mask, split_y[0:tiles])
    print("y_mask shape:", y_mask.shape)
    print("y_min:", np.min(y_mask), "y_max:", np.max(y_mask))

    return x_input, y_mask


def jaccard_coef(y_true: np.ndarray, y_pred: np.ndarray) -> keras.backend.floatx():
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)

    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (
        keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) - intersection + 1.0
    )


def predictions_in_chunks(model, generator, num_run, dataset, num_tiles, batch_size, experiment):
    num_batches = generator.__len__()
    pred_mmap = np.memmap(f'../models/{experiment}/predictions/pred_{dataset}_{num_run}.npy', mode="w+",
                          shape=(num_tiles, 256, 256, 3), dtype=np.float32)

    for batch_idx in range(num_batches):
        batch_x, _ = generator.__getitem__(batch_idx)
        batch_preds = model.predict(batch_x)
        print(f'batch no: {batch_idx}, batch_x shape: {batch_x.shape}, batch_pred shape: {batch_preds.shape}')
        start = batch_idx * batch_size
        end = start + batch_size
        pred_mmap[start:end] = batch_preds


def get_filenames(experiment):
    files = os.listdir(f'../models/{experiment}/')
    return [f for f in files if f.startswith('model_')]


def predictions_for_models(train_generator, val_generator, test_generator, experiment, test_val_tiles, train_tiles, batch_size, model_range=None):
    saved_models = get_filenames(experiment)
    if model_range is None:
        model_range = (0, len(saved_models))
    print(f'All found models: {saved_models}')

    for idx in range(model_range[0], model_range[1]):
        print(f'Make predictions with model {saved_models[idx]}')
        model = load_model(f'../models/{experiment}/{saved_models[idx]}')
        num_run = saved_models[idx].split('_')[-1][0]
        print('Start predictions with test data...')
        predictions_in_chunks(model, test_generator, num_run, 'test', test_val_tiles, batch_size, experiment)
        print('Start predictions with validation data...')
        predictions_in_chunks(model, val_generator, num_run, 'val', test_val_tiles, batch_size, experiment)
        print('Start predictions with training data...\n')
        predictions_in_chunks(model, train_generator, num_run, 'train', train_tiles, batch_size, experiment)
