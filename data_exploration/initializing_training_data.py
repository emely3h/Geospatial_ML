from typing import Tuple
import numpy as np


def initialize_training_data(
    train_split_x: np.ndarray, train_split_y: np.ndarray, train_tiles: int
) -> Tuple[np.ndarray, np.ndarray]:
    print("Initializing training data...")
    x_input = np.zeros((train_tiles, 256, 256, 5), dtype=np.float32)
    print("x_input shape:", x_input.shape)
    print("x_min:", np.min(x_input), "x_max:", np.max(x_input))

    print("\nCopying training data to x_input...")
    np.copyto(x_input, train_split_x[0:train_tiles])
    print("x_input shape:", x_input.shape)
    print("x_min:", np.min(x_input), "x_max:", np.max(x_input))

    print("\nInitializing y_mask...")
    y_mask = np.zeros((train_tiles, 256, 256), dtype=np.float32)
    print("y_mask shape:", y_mask.shape)
    print("y_min:", np.min(y_mask), "y_max:", np.max(y_mask))

    print("\nCopying training data to y_mask...")
    np.copyto(y_mask, train_split_y[0:train_tiles])
    print("y_mask shape:", y_mask.shape)
    print("y_min:", np.min(y_mask), "y_max:", np.max(y_mask))

    return x_input, y_mask


# if __name__ == "__main__":
#     train_split_x = np.random.rand(100, 256, 256, 5).astype(np.float32)
#     train_split_y = np.random.rand(100, 256, 256).astype(np.float32)
#     train_tiles = 50

#     x_input, y_mask = initialize_training_data(
#         train_split_x, train_split_y, train_tiles
#     )
