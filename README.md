readme_content = """# Plant Disease Recognition System 🌿🔬

An end-to-end deep learning project for automatic plant disease detection using leaf images.  
Built with a custom Convolutional Neural Network (CNN) and a Streamlit web interface.

## Features

- Detects **38 classes** of healthy and diseased plant leaves (Apple, Tomato, Grape, Potato, etc.).
- **Streamlit Web App** for instant predictions from uploaded images.
- Trained on **87,000+ images** with >96% validation accuracy.
- **Pre-trained model included** for quick deployment.

---

## Installation

### 1. Clone the repository
git clone https://github.com/premvaidya/Plants_Health.git
cd Plants_Health

text

### 2. Install dependencies
pip install -r requirements.txt

text
Common requirements:
streamlit
tensorflow
numpy
pandas
matplotlib
opencv-python

text

### 3. Model file
Ensure `trained_model.keras` is in your directory, or train using `Train_plant_disease.ipynb`.

---

## Usage

streamlit run main.py

text
- Open the provided link in your browser.
- Use the sidebar to navigate pages.

---

## Model Details

- **Input:** 128×128 color leaf images.
- **Architecture:** 10-layer CNN with dropout for regularization.
- **Optimizer:** Adam (`lr=0.0001`)
- **Loss:** Categorical Crossentropy
- **Accuracy:** ~99% training, ~96% validation.

---

## Dataset

- [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (~87k images, 38 categories).
- Split: Train (80%), Validation (20%), separate test set.

---

## Project Structure

Plants_Health/
├── main.py <br>
├── Train_plant_disease.ipynb <br>
├── Test_Plant_Disease.ipynb <br>
├── trained_model.keras <br>
├── training_hist.json <br>
├── home_page.jpg <br>
├── requirements.txt <br>
└── [data/ ...] <br>

---

## Acknowledgements

- Dataset by Vipoooool (Kaggle).
- Built with TensorFlow, Streamlit, and OpenCV.

---


**Keep our crops healthy and our planet green!**
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("✅ README.md file created successfully!")
Save that script as make_readme.py and run:

bash
python make_readme.py
You’ll now have a ready-to-use README.md in your current folder.

