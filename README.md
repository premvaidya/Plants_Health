# Plant Disease Recognition System 🌿🔬

This project is an end-to-end deep learning solution for automatic plant disease detection using leaf images. Leveraging a custom-trained Convolutional Neural Network (CNN), this system can classify leaves into 38 different categories, ranging from healthy specimens to various crop diseases, with a user-friendly web interface built using Streamlit.

## Features

- **Detect 38 Classes:** Including specific diseases for Apple, Tomato, Grape, Potato, and more, as well as their healthy states.
- **Streamlit Interface:** Easily upload leaf images and get instant predictions for plant health and disease.
- **Deep Learning Backend:** Trained on a dataset of 87,000+ images with a robust CNN architecture.
- **High Accuracy:** Achieves >96% validation accuracy after 10 epochs.
- **Model Export:** Pre-trained model and training history are included for quick deployment and reproducibility.

---

## Table of Contents

- [Demo](#demo)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Demo

- Go to the **Disease Recognition** page, upload a plant leaf image (preferably clear and centered).
- Click **Predict** to get a disease diagnosis and suggested action.

---

## Screenshots

### Home Page

> _Friendly UI with project summary and guidance._

### Disease Prediction

> _Upload an image and get instant results._

---

## Installation

### 1. Clone this repository

git clone <repository-url>
cd Plants_Health

text

### 2. Install Requirements

Python 3.8+ is recommended.

pip install -r requirements.txt

text

Typical requirements:

streamlit
tensorflow
numpy
pandas
matplotlib
opencv-python

text

### 3. Download Model

Ensure that `trained_model.keras` is present in your working directory.  
If not, you’ll need to train the model (`Train_plant_disease.ipynb`) or download it if provided.

---

## Usage

### Run the Streamlit App

streamlit run main.py

text

- Open the browser link Streamlit provides.
- Use the sidebar to navigate between **Home**, **About**, and **Disease Recognition**.

---

## Model Details

- **Input:** Color leaf images, resized to 128x128 pixels.
- **Architecture:** 10-layer CNN with increasing filter depth and dropout for regularization.  
  - Conv2D layers (32 → 64 → 128 → 256 → 512 filters)
  - Dense layer with 1500 neurons before classification
  - Dropout: 25% (after conv), 40% (after dense)
  - Output: 38 neurons (softmax) for multiclass classification
- **Optimizer:** Adam (`lr=0.0001`)
- **Loss:** Categorical Crossentropy
- **Performance:**  
  - Training accuracy: ~99%  
  - Validation accuracy: ~96%

---

## Dataset

- Based on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- ~87,000 images of healthy and diseased leaf samples across 38 categories.
- Split: 80% train, 20% validation, with separate test samples for final inference.

---

## Project Structure

Plants_Health/
│
├── main.py # Streamlit web app
├── Train_plant_disease.ipynb # Model training notebook (with model definition)
├── Test_Plant_Disease.ipynb # Model testing/validation notebook
├── trained_model.keras # Saved trained model
├── training_hist.json # Model training history
├── home_page.jpg # UI image for the homepage
├── requirements.txt # Python dependencies
└── [data/ ...] # (Optional) Folder for train/valid/test images

text

---

## Acknowledgements

- Inspired by open-source plant disease datasets and similar deep learning agricultural projects.
- Dataset by Vipooool ([Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)).
- Built using TensorFlow, Streamlit, and open Python libraries.

---

## License

*Specify your license here (e.g., MIT, Apache, GPL, etc.)*

---

## Contact

For issues or contributions, please submit a pull request or open an issue in the repository.

---

**Keep our crops healthy and our planet green!**
