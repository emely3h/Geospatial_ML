# Geospatial ML

## Overview

The Geospatial ML project is a machine learning initiative developed by eomap, a company that studies and analyzes satellite images to make precise statements about various water quality parameters, water depth biodiversity, and many more. While most of the steps in this process are fully automated, a final review by a geographer is still required to ensure accuracy. This review involves flagging out certain parts of the image, such as cloud shadows, sun glint, and others that may affect the results.

The aim of this project is to train a neural network to identify pixels that provide a valid end result with high probability. To achieve this goal, eomap has provided us with a large dataset of already flagged and unflagged images.

## Getting Started

To get started with this project, follow the steps below:

1.  Clone the repository using `git clone <repository_url>`.
2.  Create and activate a virtual environment by running the following commands:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Install the required packages by running `pip install -r requirements.txt` in the terminal.
4.  Start Jupyter Notebook by running `jupyter lab` in the terminal.
5.  To deactivate the virtual environment, simply run `deactivate` in the terminal.

## Requirements

To run this project, you will need the following:

- Python 3
- Jupyter Notebook
- The packages listed in `requirements.txt`

## Usage

Create an empty /data folder in the project root directory. Copy the two subfolders 'unflagged' and 'flags_applied' of the data folder into the project data folder.
Execute main.py in /prepare_data

To use this project, simply open the Jupyter Notebook and run the code cells.
