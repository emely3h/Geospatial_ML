# Geospatial ML

## Overview

The Geospatial ML project is a machine learning initiative developed by eomap, a company that studies and analyzes satellite images to make precise statements about various water quality parameters, water depth biodiversity, and many more. While most of the steps in this process are fully automated, a final review by a geographer is still required to ensure accuracy. This review involves flagging out certain parts of the image, such as cloud shadows, sun glint, and others that may affect the results.
The aim of this project is to train a neural network to identify pixels that provide a valid end result with high probability. To achieve this goal, eomap has provided us with a large dataset of already flagged and unflagged images.

## Getting Started

To get started with this project, follow the steps below:
We are using poetry as a dependency manager.

One time

Install Python 3.10≤

Install [poetry](https://python-poetry.org/)

1.  Clone the repository using `git clone <repository_url>`.

2. Activate poetry virtual environment.
```
# Create a virtual environment
$ poetry shell
# Install all packages
$ poetry install
```

## Requirements

To run this project, you will need the following:

- Python 3
- Jupyter Notebook


## Prepare Data

To set up the project, follow these steps:

1. Create an empty `/data` folder in the project root directory.
2. Copy the two subfolders `unflagged` and `flags_applied` from the original data folder into the project `/data` folder.
3. Create a `.env` file in the root directory and define the `DATA_PATH` variable
4. Run `main.py` in the `/prepare_data` directory to prepare the data for analysis.

## Working with google colab

_Only edit and push jupyter notebooks on google colab, edit and push .py scripts always locally!_

As it is only possible to push changes on jupyter notebooks to github from google colab it is best to do all changes in the scripts locally.
Run all the prepare data scripts locally and only store the final numpy array in google drive.

- The MachineLearning folder will appear in your google drive `shared with me` subfolder. To access it from within colab you need to create a shortcut. (left click on the folder, 'add shortcut')
