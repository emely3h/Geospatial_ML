# Geospatial ML

## Project Intro

Satellite images can be used to calculate water quality and depth. Most steps in this process are fully automated, but
it is still essential that in the last step, a geographer reviews the calculated results and flags out certain parts of
the image, if necessary. Causes for this are cloud shadows, which make pixels darker than they actually are, sun glint,
and some others. Our project aims to automate the final reviewing process by training a convolutional neural network (
e.g. U-Net) to predict the correct label for each pixel (Semantic Image Segmentation).

todo: add graphic

## Installation

To get started with this project, follow the steps below:
We are using poetry as a dependency manager.

Requirements: Python 3.10 ≤

Install [poetry](https://python-poetry.org/)

1. Clone the repository using `git clone <repository_url>`.

2. Activate poetry virtual environment.

```
# Create a virtual environment
$ poetry shell
# Install all packages
$ poetry install
```

## Dataset

Our dataset contains 15 large satellite images taken from the reservoir Lake Cahora Bassa in Mozambique at different
times. The satellite images are RGB images with an additional alpha channel. In addition to the satellite images, the
dataset includes a water-quality image (wq image) with one channel and a mask image with one channel.

The water quality image and the mask image contain one channel valued between 0 and 255.

- All pixels with 253 and 255 are labeled as invalid.
- All pixels with value 0 are labeled as land.
- The rest of the pixels are labeled as valid.

Small excerpt from the dataset:

![example_images](docs/example_images.png)

As the example images show that the wq image is already partly flagged. The white parts represent the clouds while the
gray parts represent cloud shadows. However the flagging is not the same as in the mask image. A geologist manually
adapts the flagging with his expertise. The challenge of this project is to train a neural network so that the result is
closer to the mask image than the wq image.

To train our machine learning model we combined the rgb images and the water quality images into one input image
with 5 channels. As the size of each image is too large to use it as input for the neural network we split each image
into smaller tiles of size 256 x 256 once with an overlap of 56 pixels and once without overlapping.

The following diagram gives an overview of the data preparation steps:

![overview](docs/overview.png)

## Project Structure

### [prepare_data/](https://github.com/emely3h/Geospatial_ML/tree/main/prepare_data)

To split the images into tiles, follow these steps:

1. Create an empty `/data` folder in the project root directory.
2. Copy the two subfolders `unflagged` and `flags_applied` from the original data folder into the project `/data`
   folder.
3. Create a `.env` file in the root directory and define the `DATA_PATH` variable
4. Run `main.py` in the `/prepare_data` directory to prepare the data for analysis.

The result is a folder named /google_drive that contains one compressed file per date. Each file consists of the
true_mask array and the x_input array which are used to train the neural network.

5. Execute the `/full_dataset_splitting_mmaps.ipynb` notebook

### [models/](https://github.com/emely3h/Geospatial_ML/tree/main/models)

In this folder, all models used for training are saved and some additional helper function. To train our network used
the u-net architecture.

### [experiments/](https://github.com/emely3h/Geospatial_ML/tree/main/experiments)

All experiments that have been done are saved here.

### [data_exploration/](https://github.com/emely3h/Geospatial_ML/tree/main/experiments)

In this folder we explored our dataset to know about the class imbalance, number of tiles per image and the overlap
between the input image and the true
mask ([physics_jaccard](https://github.com/emely3h/Geospatial_ML/blob/main/data_exploration/physics_jaccard.ipynb)).

### [evaluation/](https://github.com/emely3h/Geospatial_ML/tree/main/evaluation)

This folder contains all helper methods and functions to calculate metrics for each trained folder which are saved on
google drive.

## Working with google colab

_Only edit and push jupyter notebooks on google colab, edit and push .py scripts always locally!_

As it is only possible to push changes on jupyter notebooks to github from google colab it is best to do all changes in
the scripts locally.
Run all the prepare data scripts locally and only store the final numpy array in google drive.

- The MachineLearning folder will appear in your google drive `shared with me` subfolder. To access it from within colab
  you need to create a shortcut. (left click on the folder, 'add shortcut')
