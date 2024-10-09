# Hand Sign Language Recognition Project

## Overview

This project aims to develop a system that can recognize hand signs from live video feeds, which could potentially be used for sign language translation. The project includes creating a custom dataset of hand gestures, building and training a Convolutional Neural Network (CNN), and deploying the model to detect hand signs in real-time using a live camera feed.

## Project Structure

1. **CNN Creation**:
    - This notebook is dedicated to building a Convolutional Neural Network (CNN) for hand sign classification.
    - Data preprocessing includes rescaling, shear, zoom, and flip transformations using `ImageDataGenerator`.
    - The model is trained on a dataset of 23,900 images divided into 10 classes.
    - Libraries used: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, and Scikit-learn.

2. **Code for Detecting Hand Signs in Live Cam**:
    - This notebook implements the live detection of hand signs using a pre-trained CNN model.
    - It uses OpenCV to capture video from the camera and define a region of interest (ROI) where the hand gestures are detected.
    - The model processes the video frames in real-time and subtracts the background to focus on the hand signs.
    - Libraries used: TensorFlow, Keras, OpenCV, NumPy, and Matplotlib.

3. **Creating Sign Language Dataset**:
    - This notebook is used for creating the dataset by capturing hand gestures in real-time.
    - A background subtraction technique is applied to enhance the clarity of the hand signs.
    - The images captured from the region of interest (ROI) are used to create the dataset.
    - Libraries used: OpenCV, NumPy.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## How to Run

1. **Dataset Creation**:
    - Use the `Creating Sign Language Dataset.ipynb` to capture hand signs. Ensure you have a camera connected for real-time data collection.
    - Save the dataset in a structured folder format, ensuring each folder corresponds to a specific sign class.

2. **Training the Model**:
    - Run the `CNN Creation.ipynb` notebook to preprocess the dataset and train the CNN model.
    - After training, save the model for use in real-time detection.

3. **Real-time Hand Sign Detection**:
    - Open the `Code for detecting the hand signs in live cam.ipynb` to use the pre-trained model for real-time detection of hand gestures.
    - Ensure the camera is connected and properly configured for live video feed processing.

## Potential Use Cases

- Sign language translation for individuals with hearing impairments.
- Gesture-based control systems.
- Interactive applications in the field of human-computer interaction.

## Acknowledgements

- TensorFlow and Keras for model building.
- OpenCV for image and video processing.
- The creators of the datasets used in training.
