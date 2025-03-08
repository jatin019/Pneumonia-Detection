# Pneumonia Detection Web App

## Project Overview

This project is a **deep learning-based web application** for detecting Pneumonia from chest X-ray images. It uses **three different models**—**CNN, VGG16, and ResNet-50**—to classify images as either **Normal** or **Pneumonia**. After comparison, **ResNet-50** performed the best and was saved as `my_pneumonia_detection_model.h5` for further use in the application. The application is built using **TensorFlow** for image classification and **Streamlit** for the web interface.

## Features

- Upload a **chest X-ray image** (`.jpg`, `.jpeg`, `.png`).
- The app **preprocesses** the image and passes it through the trained ResNet-50 model.
- Displays the **prediction (Normal / Pneumonia) with confidence score**.
- **User-friendly web interface** using Streamlit.

## Installation & Setup

### Prerequisites

Make sure you have **Python 3.7+** installed on your system. You also need `pip` for package management.

### Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install manually:

```bash
pip install tensorflow numpy matplotlib streamlit
```

## How to Run the Web App

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command:
   ```bash
   streamlit run app.py
   ```
4. The web app will open in your **default web browser**.

## Model & Prediction

- The application uses a pre-trained **ResNet-50 model (********`my_pneumonia_detection_model.h5`********\*\*\*\*)**.
- Initially, three models (**CNN, VGG16, and ResNet-50**) were trained and compared.
- **ResNet-50** outperformed the other models and was selected for deployment.
- The uploaded image is resized to **224x224 pixels** before being passed to the model.
- The model predicts whether the chest X-ray indicates **Pneumonia** or is **Normal**.

## Model Comparison Results

| Model     | Accuracy          |
| --------- | ----------------- |
| CNN       | 75.64%            |
| VGG16     | 70.83%            |
| ResNet-50 | **89.58%** (Best) |

## File Structure
```
/your-project-folder
│── app.py                 # Streamlit web app
│── my_pneumonia_detection_model.h5 # Best performing deep learning model (ResNet-50)
│── requirements.txt        # List of dependencies
│── README.md               # Project documentation
│── code.py                 # Python file with model training and preprocessing
│── chest_xray/             # Dataset folder
│   ├── train/              # Training images
│   ├── test/               # Testing images
│   ├── val/                # Validation images
```


##  Dependencies

- **TensorFlow** – Deep learning framework
- **NumPy** – Numerical computations
- **Matplotlib** – Image processing
- **Streamlit** – Web interface



