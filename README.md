# Trash Classifier Python Project

## Overview
This project leverages a Convolutional Neural Network (CNN) to classify images of trash into different categories: Cardboard, Food Organic, Glass, Metal, Misc, Paper, Plastic, and Vegetation. The classification system aims to improve the efficiency and sustainability of waste management and recycling processes.


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Files](#files)
- [References](#references)

## Installation
To run the proof of concept user interface locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/dj4free/trashClassifierPython.git
    cd trashClassifierPython
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the web application:
    ```bash
    python main.py
    ```

## Usage
1. Navigate to the web application (locally or [hosted version](https://dolphin-app-hel2u.ondigitalocean.app/)).
2. Upload an image of trash for classification or select a random image.
3. View the classification results and confidence scores.

## Model Training
The model is trained using the RealWaste dataset from the UCI Machine Learning Repository. The process includes data preprocessing, training with a CNN, and validation through K-fold cross-validation.

### Steps:
1. Load and preprocess data.
2. Define the CNN architecture.
3. Train the model with cross-validation.
4. Evaluate and fine-tune the model.

## Evaluation
The model's performance is evaluated using K-fold cross-validation, yielding metrics such as accuracy, precision, recall, and F1-score. Results show an issue with overfitting due to small dataset and unbalanced sample sizes. Augmentation was used only to balance the training set.

### Results:
- Average Training Accuracy: 96.04%
- Average Validation Accuracy: 74.61%
- Average Validation F1-Score: 0.1383
- Average Validation Recall: 0.1428

## Files
- `main.py`: The main script to run the web application.
- `TrashClassifier.ipynb`: Jupyter Notebook for model training and evaluation.
- `TrashClassifier_best_model.h5`: Jupyter Notebook trained model with fine-tuned hyperparameters.
- `requirements.txt`: List of dependencies.
- `Dockerfile` and `compose.yaml`: Docker configuration files for containerized deployment.
- `sampleImageRandom.py`: Script for random image selection.

## References
[DOI](https://doi.org/10.24432/C5SS4G) by Single, Sam, Iranmanesh, Saeid, and Raad, Raad. (2023). RealWaste. UCI Machine Learning Repository.

[Keras Conv2D layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)

[TensorFlow plot templates](https://www.tensorflow.org/guide/basics)
