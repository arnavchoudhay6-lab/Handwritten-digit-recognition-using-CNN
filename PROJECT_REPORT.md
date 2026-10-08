# PROJECT REPORT

## Title
Handwritten Digit Recognition Using Convolutional Neural Network (CNN)

## 1. Introduction
Handwritten digit recognition is a common application of Artificial Intelligence and Computer Vision. The aim of this project is to recognize handwritten digits from 0 to 9 using a Convolutional Neural Network.

## 2. Problem Statement
A computer cannot directly understand a handwritten digit as a human does. This project develops a deep learning model that learns patterns from handwritten digit images and predicts the correct digit.

## 3. Objectives
- To understand the basics of Deep Learning.
- To use a CNN for image classification.
- To train a model using the MNIST dataset.
- To predict handwritten digits.
- To create a simple user interface using Streamlit.

## 4. Dataset
The project uses the MNIST dataset. It contains grayscale images of handwritten digits from 0 to 9. Each image is 28 x 28 pixels.

## 5. Methodology
1. Load the MNIST dataset.
2. Normalize image pixel values.
3. Reshape images for CNN input.
4. Build the CNN model.
5. Train the model.
6. Evaluate the model on test data.
7. Save the trained model.
8. Use Streamlit for prediction.

## 6. CNN Architecture
- Input: 28 x 28 x 1
- Convolution Layer: 32 filters
- Max Pooling
- Convolution Layer: 64 filters
- Max Pooling
- Flatten
- Dense Layer: 128 neurons
- Dropout
- Output Layer: 10 classes

## 7. Result
The CNN is expected to achieve approximately 98% test accuracy. The application predicts a digit from an uploaded handwritten image.

## 8. Advantages
- Fast prediction.
- Good accuracy.
- Simple user interface.
- Easy to extend.
- Demonstrates a real Deep Learning application.

## 9. Limitations
- Best suited to single-digit images.
- Image quality can affect prediction.
- The model is trained mainly on MNIST-style digits.

## 10. Future Scope
- Add a drawing canvas.
- Deploy the app online.
- Recognize multiple digits.
- Add more datasets.
- Build an Android version.

## 11. Conclusion
This project demonstrates how a Convolutional Neural Network can be used for handwritten digit classification. It provides practical experience with image preprocessing, model training, evaluation and deployment.
