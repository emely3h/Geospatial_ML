import numpy as np
import os
import datetime


# Helper function: writes chunk to the output file using the memory-mapped array
def _write_chunk_to_mm(data, chunk, chunk_size, start_idx, end_idx, npz_start_idx, npz_end_idx, output_file_x,
                       output_file_y, rest):
    if rest:
        y_mask_chunk = data["y_mask"][npz_start_idx:npz_end_idx, ...]
        x_input_chunk = data["x_input"][npz_start_idx:npz_end_idx, ...]
    else:
        x_input_chunk = data["x_input"][chunk * chunk_size:(chunk + 1) * chunk_size, ...]
        y_mask_chunk = data["y_mask"][chunk * chunk_size:(chunk + 1) * chunk_size, ...]

    print(f'Npz file indexes: {npz_start_idx} : {npz_end_idx}')
    print(
        f'output file indexes: {start_idx} : {end_idx}  Chunk shape {y_mask_chunk.shape} min: {np.min(y_mask_chunk)} max: {np.max(y_mask_chunk)} uniques: {np.unique(y_mask_chunk)} type: {type(y_mask_chunk[0][0][0])}')

    output_file_x[start_idx:end_idx, ...] = x_input_chunk
    output_file_y[start_idx:end_idx, ...] = y_mask_chunk


# Load and concatenate tiles from original images into one memory map, each for x_input and y_mask tiles
def create_mmaps(total_tiles, data_path, mmap_path_x, mmap_path_y, compressed_path, img_names=None):
    print(f'Started at: {datetime.datetime.now()}')

    x_output_shape = (total_tiles, 256, 256, 5)
    y_output_shape = (total_tiles, 256, 256)

    # memory-mapped array to hold the output data
    output_file_x = np.memmap(os.path.join(data_path, mmap_path_x), mode="w+", shape=x_output_shape, dtype=np.uint8)
    output_file_y = np.memmap(os.path.join(data_path, mmap_path_y), mode="w+", shape=y_output_shape, dtype=np.uint8)

    file_count = 0
    chunk_size = 500
    start_idx = 0
    end_idx = chunk_size

    if img_names is None:
        for file in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path, file)) and file.startswith('2022'):
                img_names.append(file)

    for file in img_names:
        file_count += 1

        # Load the compressed numpy array in chunks using np.memmap
        with np.load(os.path.join(data_path, file), mmap_mode="r") as data:
            npz_start_idx = 0
            npz_end_idx = chunk_size

            num_chunks = data["x_input"].shape[0] // chunk_size
            rest = data["x_input"].shape[0] % chunk_size
            print(f'loading file {file_count}: {file} shape: {data["x_input"].shape} {data["y_mask"].shape}')

            for chunk in range(num_chunks):
                print(f'Chunk {chunk}')
                _write_chunk_to_mm(data, chunk, chunk_size, start_idx, end_idx, npz_start_idx, npz_end_idx,
                                   output_file_x, output_file_y, False)

                if chunk != (num_chunks - 1):
                    start_idx = end_idx
                    end_idx += chunk_size
                    npz_start_idx = npz_end_idx
                    npz_end_idx += chunk_size

            # Calculate + append rest of .npz
            print(f'Rest: {rest}')
            start_idx = end_idx
            end_idx = end_idx + rest

            npz_start_idx = (chunk + 1) * chunk_size
            npz_end_idx = (chunk + 1) * chunk_size + rest
            _write_chunk_to_mm(data, chunk, chunk_size, start_idx, end_idx, npz_start_idx, npz_end_idx,
                               output_file_x, output_file_y, True)

            print()
            start_idx = end_idx
            end_idx += chunk_size

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
