import numpy as np
import os
import datetime


# Helper function: writes chunk to the output file using the memory-mapped array
def _write_chunk_to_mm(data, mmap_idx, output_file_x, output_file_y):

    x_input_chunk = data["x_input"]
    y_mask_chunk = data["y_mask"]
    num_tiles = x_input_chunk.shape[0]
    mmap_idx_end = mmap_idx+num_tiles

    print(f'output file indexes: {mmap_idx} : {mmap_idx_end}  x_input_shape {x_input_chunk.shape} y_mask_shape {y_mask_chunk.shape}')
    print(f'max_x_input: {np.max(x_input_chunk)} min_x_input: {np.min(x_input_chunk)} x_input_len_uniques: {len(np.unique(x_input_chunk))}')
    print(f'y_mask_min: {np.min(y_mask_chunk)} y_mask_max: {np.max(y_mask_chunk)} y_mask_uniques: {np.unique(y_mask_chunk)} type: {type(y_mask_chunk[0][0][0])} \n')

    output_file_x[mmap_idx:mmap_idx_end, ...] = x_input_chunk
    output_file_y[mmap_idx:mmap_idx_end, ...] = y_mask_chunk


# Load and concatenate tiles from original images into one memory map, each for x_input and y_mask tiles
def create_mmaps(total_tiles, data_path, mmap_path_x, mmap_path_y, compressed_path, img_names=[]):
    print(f'Started at: {datetime.datetime.now()}')

    x_output_shape = (total_tiles, 256, 256, 5)
    y_output_shape = (total_tiles, 256, 256)

    output_file_x = np.memmap(os.path.join(data_path, mmap_path_x), mode="w+", shape=x_output_shape, dtype=np.uint8)
    output_file_y = np.memmap(os.path.join(data_path, mmap_path_y), mode="w+", shape=y_output_shape, dtype=np.uint8)

    file_count = 0
    mmap_idx = 0

    if len(img_names) is 0:
        for file in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path, file)) and file.startswith('2022'):
                img_names.append(file)
    print(f'Images: {img_names}')
    for file in img_names:
        file_count += 1

        # Load the compressed numpy array in chunks using np.memmap
        with np.load(os.path.join(data_path, file), mmap_mode="r") as data:
            num_tiles = data['x_input'].shape[0]
            print(f'loading file {file_count}: {file} shape: {data["x_input"].shape} {data["y_mask"].shape}, num_tiles: {num_tiles}')
            _write_chunk_to_mm(data, mmap_idx, output_file_x, output_file_y)
            mmap_idx += num_tiles

    print('finished concatenating arrays')

    output_file_x.flush()
    output_file_y.flush()
    print('finished flushing')

    np.savez_compressed(os.path.join(data_path, compressed_path), x_input=output_file_x, y_mask=output_file_y)
    print('finish compressing')

    # Delete the memory-mapped array to free up resources
    del output_file_x
    del output_file_y

    print(f'Finished at: {datetime.datetime.now()}')
