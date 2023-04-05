import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split
from data_exploration.mask_stats import Mask_Stats


def _get_valid_split(x_input, y_mask, threshold):
    valid = False
    counter = 0
    threshold = threshold
    rand = 1
    while not valid:

        X_train, X_test, y_train, y_test = train_test_split(x_input, y_mask, test_size=0.2, random_state=rand)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=rand)

        stats_train = Mask_Stats(y_train)
        stats_val = Mask_Stats(y_val)
        stats_test = Mask_Stats(y_test)

        invalid_percentages = [stats_train.pix_invalid_per, stats_val.pix_invalid_per, stats_test.pix_invalid_per]
        if (max(invalid_percentages) - min(invalid_percentages)) <= threshold:
            valid = True
        print(f'Split counter {counter}, diff: {max(invalid_percentages) - min(invalid_percentages)}')
        counter += 1
        rand += 1

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_dataset(chunk_size, total_tiles, data_path, threshold):
    """
    Splits a dataset consisting of non overlapping tiles into a training set (60%), validation set (20%) and testing set (20%)
    Requires a folder called combined in the data_path that holds a memory map both for all input and all mask tiles

    Args:
        chunk_size: Number of tiles that get loaded into RAM at once, splitting is done on individual chunks
        total_tiles: Total number of tiles in the entire dataset that is being split
        data_path: path where memory maps that hold the train, test and validation split are stored
        threshold: maximum percentage difference of invalid pixels in different sets
    """
    print(f'Started at: {datetime.datetime.now()}')

    train_tiles = total_tiles // 100 * 60 + 1
    test_val_tiles = total_tiles // 100 * 20 + 1

    train_split_x = np.memmap(os.path.join(data_path, "train_split_x.npy"), mode="w+", shape=(train_tiles, 256, 256, 5),
                              dtype=np.float32)
    train_split_y = np.memmap(os.path.join(data_path, "train_split_y.npy"), mode="w+", shape=(train_tiles, 256, 256),
                              dtype=np.float32)
    test_split_x = np.memmap(os.path.join(data_path, "test_split_x.npy"), mode="w+",
                             shape=(test_val_tiles, 256, 256, 5), dtype=np.float32)
    test_split_y = np.memmap(os.path.join(data_path, "test_split_y.npy"), mode="w+", shape=(test_val_tiles, 256, 256),
                             dtype=np.float32)
    val_split_x = np.memmap(os.path.join(data_path, "val_split_x.npy"), mode="w+", shape=(test_val_tiles, 256, 256, 5),
                            dtype=np.float32)
    val_split_y = np.memmap(os.path.join(data_path, "val_split_y.npy"), mode="w+", shape=(test_val_tiles, 256, 256),
                            dtype=np.float32)

    y_mask_mm = np.memmap(f'{data_path}/combined/y_mask.npy', shape=(total_tiles, 256, 256), dtype=np.uint8, mode='r')
    x_input_mm = np.memmap(f'{data_path}/combined/x_input.npy', shape=(total_tiles, 256, 256, 5), dtype=np.uint8,
                           mode='r')

    chunk_size = chunk_size
    num_chunks = total_tiles // chunk_size
    rest = total_tiles % chunk_size

    train_idx = 0
    val_idx = 0
    test_idx = 0

    for c in range(0, (num_chunks)):
        print(f'{c}')
        if c == (num_chunks - 1):
            print(f'enter rest {(c + 1) * chunk_size}:{(c + 1) * chunk_size + rest}')
            y_mask = y_mask_mm[(c + 1) * chunk_size:(c + 1) * chunk_size + rest, ...]
            x_input = x_input_mm[(c + 1) * chunk_size:(c + 1) * chunk_size + rest, ...]
        else:
            print(f'cut at: {c * chunk_size}:{(c + 1) * chunk_size}')
            y_mask = y_mask_mm[c * chunk_size:(c + 1) * chunk_size, ...]
            x_input = x_input_mm[c * chunk_size:(c + 1) * chunk_size, ...]

        print(
            f'Chunk: {c}, x_input: {x_input.shape}, y_mask: {y_mask.shape}, train_idx: {train_idx}, val_idx: {val_idx}, test_idx: {test_idx}')

        X_train, X_val, X_test, y_train, y_val, y_test = _get_valid_split(x_input, y_mask, threshold)

        print(
            f'X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}')
        print()
        train_split_x[train_idx:train_idx + X_train.shape[0]] = X_train
        train_split_y[train_idx:train_idx + y_train.shape[0]] = y_train
        test_split_x[val_idx:val_idx + X_val.shape[0]] = X_val
        test_split_y[val_idx:val_idx + y_val.shape[0]] = y_val
        val_split_x[test_idx:test_idx + X_test.shape[0]] = X_test
        val_split_y[test_idx:test_idx + y_test.shape[0]] = y_test

        train_idx += X_train.shape[0]
        val_idx += X_val.shape[0]
        test_idx += X_test.shape[0]
    print(f'Finished at: {datetime.datetime.now()}')


def _validate(df, total_tiles, total_invalids, threshold, target):
    df_num_tiles = df['num_tiles'].sum()
    df_num_invalides = df['num_invalid_pix'].sum()

    percentage_tiles = 100 / total_tiles * df_num_tiles
    percentage_invalides = 100 / total_invalids * df_num_invalides

    if (target + threshold) >= percentage_tiles <= (target - threshold):
        return False
    elif (target + threshold) >= percentage_invalides <= (target - threshold):
        return False
    else:
        print(f'Total_tiles: {total_tiles} total_invalids: {total_invalids}')
        print(f'percentage_tiles: {percentage_tiles}, percentage_invalides: {percentage_invalides} ')
        print()

        return True


def group_images(threshold, df):
    """
    Calculates possible splits for dataset with overlapping tiles. When applying
    a 60-20-20 split this means we will use 10 image for training,
    3 images for validation and 3 images for testing. See data_exploration notebook.

    Args:
        threshold: max percentage deviation from 60-20-20 split
        df: dataframe, containing amount of tiles and invalid pixels per image
    """

    total_tiles = df['num_tiles'].sum()
    total_invalids = df['num_invalid_pix'].sum()

    valid = False
    count = 0

    while not valid:
        print(f'Count: {count}')
        df_copy = df.copy()

        training_set = df_copy.sample(n=10)
        df_copy = df_copy.drop(training_set.index)

        validation_set = df_copy.sample(n=3)
        df_copy = df_copy.drop(validation_set.index)

        test_set = df_copy.sample(n=3)
        df_copy = df_copy.drop(test_set.index)

        train_validate = _validate(training_set, total_tiles, total_invalids, threshold, 60)
        val_validate = _validate(validation_set, total_tiles, total_invalids, threshold, 20)
        test_validate = _validate(test_set, total_tiles, total_invalids, threshold, 20)

        if train_validate and val_validate and test_validate:
            valid = True
        else:
            training_set = None
            validation_set = None
            test_set = None
            count += 1

    return training_set, validation_set, test_set