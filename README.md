# Handwritten Digit Recognition using CNN

A beginner-friendly Deep Learning project that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Project Title
**Handwritten Digit Recognition Using Convolutional Neural Network (CNN)**

## Objective
To build a deep learning model that can identify handwritten digits from an image.

## Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Streamlit
- Pillow

## Project Structure
```text
handwritten_digit_cnn/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── digit_cnn.keras       # created after training
└── sample/
    └── README.txt
```

## How to Run

### 1. Install Python
Python 3.10 or newer is recommended.

### 2. Install libraries
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```

The MNIST dataset is downloaded automatically and the trained model is saved as:
```text
models/digit_cnn.keras
```

### 4. Start the web app
```bash
streamlit run app.py
```

Then open the local Streamlit address shown in the terminal.

## How It Works
1. MNIST handwritten digit images are loaded.
2. Images are normalized to values between 0 and 1.
3. A CNN learns important visual patterns.
4. The model predicts one of the ten classes: 0 to 9.
5. Streamlit provides a simple interface for prediction.

## Model
The CNN contains:
- Convolution layers
- Max Pooling layers
- Flatten layer
- Dense layer
- Softmax output layer

## Expected Result
The model should normally achieve around **98% test accuracy**, depending on training settings and environment.

## Future Scope
- Add a drawing canvas.
- Deploy the application online.
- Add prediction confidence.
- Train on more handwritten datasets.
- Convert the model for mobile use.

## Author
Add your name, college, course and section here before submitting.
