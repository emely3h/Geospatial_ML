from models.unet_model import unet_2d
from tensorflow import keras
from keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import numpy as np
from typing import Tuple
import tensorflow as tf


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


def copy_data_to_arrays(
    split_x: np.ndarray, split_y: np.ndarray, tiles: int, chunk_size: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    print("Copying data to arrays...")
    x_input = np.zeros((tiles, 256, 256, 5), dtype=np.float32)
    print("x_input shape:", x_input.shape)
    print("x_input min value:", np.min(x_input), "x_input max value:", np.max(x_input))

    for i in range(0, tiles, chunk_size):
        start = i
        end = min(i + chunk_size, tiles)
        np.copyto(x_input[start:end], split_x[start:end])

    print("Data copied to x_input...")
    print("x_input shape:", x_input.shape)
    print("x_input min value:", np.min(x_input), "x_input max value:", np.max(x_input))

    y_mask = np.zeros((tiles, 256, 256), dtype=np.float32)
    print("\nInitializing y_mask...")
    print("y_mask shape:", y_mask.shape)
    print("y_mask min value:", np.min(y_mask), "y_mask max value:", np.max(y_mask))

    for i in range(0, tiles, chunk_size):
        start = i
        end = min(i + chunk_size, tiles)
        np.copyto(y_mask[start:end], split_y[start:end])

    print("Data copied to y_mask...")
    print("y_mask shape:", y_mask.shape)
    print("y_mask min value:", np.min(y_mask), "y_mask max value:", np.max(y_mask))

    return x_input, y_mask


def save_pickle(data: any, path: str, name: str) -> None:
    print(f"Saving {name}.pkl to {path}...")
    with open(f"{path}/{name}.pkl", "wb") as file:
        pickle.dump(data, file)
    print(f"Saved {name}.pkl to {path}.")
