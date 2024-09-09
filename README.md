# Planetary System Detection Using Synthetic Data and Machine Learning

This project generates synthetic data of planetary systems, processes the data to extract periodograms, and trains a machine learning model to detect planetary signals from frequency data. The project simulates planetary systems, labels the frequency data (periodograms) to indicate whether planets are present, and then uses this labeled data to train a machine learning model.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Generate Synthetic Planetary Systems](#step-1-generate-synthetic-planetary-systems)
  - [Step 2: Preprocess Data and Generate Periodograms](#step-2-preprocess-data-and-generate-periodograms)
  - [Step 3: Train the Model](#step-3-train-the-model)

## Overview

The primary goal of this project is to simulate star systems with planets, generate their periodograms, and train a machine learning model to detect planets based on these frequency signals. 

1. **Synthetic Data Generation**: We generate data for thousands of star systems.
2. **Data Preprocessing**: The synthetic systems are converted into periodograms using simulated observational data.
3. **Model Training**: The preprocessed data is used to train a model to detect planets based on frequency patterns.

## Project Structure

- `main_generate_systems.py`: Generates synthetic planetary systems.
- `main_preprocess.py`: Processes the generated data, simulates observations, and generates periodograms.
- `main_train.py`: Trains a machine learning model to detect planets based on the processed data.
- `code/models`: Contains the trained models.
- `code/scripts`: Contains helper scripts used for data processing and training.
- `env.yml`: Conda environment configuration file.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/anterezic1999/diplomski_public.git
    cd planetary-system-detection
    ```

2. Set up the Conda environment from the provided `env.yml` file:

    ```bash
    conda env create -f env.yml
    conda activate env
    ```

    This will install all the required dependencies.

## Usage

### Step 1: Generate Synthetic Planetary Systems

First, generate the synthetic planetary systems using `main_generate_systems.py`. By default, it will generate 10,000 systems.

```bash
python main_generate_systems.py
```
This will generate the star systems and save them in a suitable format for the next step.

Step 2: Preprocess Data and Generate Periodograms
Next, run main_preprocess.py to process the synthetic planetary systems, generate their periodograms, and store the processed data in an H5 file.

```bash
python main_preprocess.py <input_directory> <output_file> --time_span <days> --chunk_size <chunk_size> --entropy <entropy> --max_files <max_files>
```
Arguments:
input_directory: Path to the directory containing the generated planetary systems in JSON format.
output_file: Path where the processed periodograms will be saved as an H5 file.
- time_span: The time span for the simulation in days (optional, default is auto-generated based on real data).
- chunk_size: The number of systems to process in each chunk (default: 10,000).
- entropy: A measure between 0 and 1 of observation variability (default: 1).
- max_files: The maximum number of files to process (default: 20,000).
Example:

```bash
python main_preprocess.py ./data/systems ./data/periodograms.h5 --time_span 1000 --chunk_size 5000 --entropy 0.8
```
Step 3: Train the Model
Once the data is preprocessed, use main_train.py to load the data, prepare it for training, and train the machine learning model.

```bash
python main_train.py <path_to_h5_files>
```
Arguments:
path: Path to the directory containing the H5 files with processed periodogram data.

This will normalize the data, train a machine learning model, and save the trained model as planet_detection_model_full.pth.
