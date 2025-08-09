PLANT DISEASE RECOGNITION SYSTEM
A deep learning-powered application to identify and classify plant diseases from leaf images, built with TensorFlow and Streamlit.

🌱 Overview
This project enables accurate detection of plant diseases using images of leaves. Leveraging a Convolutional Neural Network trained on thousands of annotated leaf images from 38 distinct classes (crops and disease types), it helps farmers and gardeners identify problematic diseases early for improved crop management.

🚩 Features
38-class classification covering major crops and their common diseases

Highly accurate custom CNN, trained to >96% validation accuracy

Simple, user-friendly web interface built with Streamlit

Fast inference: Upload an image, get instant predictions and actionable disease names!

🌾 Supported Plant/Disease Classes
Example classes include:

Apple (Apple_scab, Black_rot, Cedar_apple_rust, healthy)

Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper_bell, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

Disease and healthy states for each

See the full class list in main.py

📊 Dataset
Based on Kaggle Plant Diseases Dataset:

87,000+ RGB images of plant leaves (healthy and diseased)

38 classes

Split: 80% training / 20% validation / separate test set

🛠️ Project Structure
Train_plant_disease.ipynb
Jupyter notebook for data loading, augmentation, CNN model building, training, evaluation, and saving the trained model.

Test_Plant_Disease.ipynb
Notebook for loading the trained model and running predictions on test images.

main.py
Streamlit app for interactive disease recognition and user dashboard.

trained_model.keras
Saved TensorFlow model (excluded from repo due to size).

training_hist.json
Training logs (accuracy/loss over epochs).

🚀 Getting Started
Prerequisites
Python 3.8+

TensorFlow 2.x

Streamlit

Numpy, Pandas, Matplotlib, Seaborn

Install dependencies:

bash
pip install tensorflow streamlit numpy pandas matplotlib seaborn
1. Model Training (optional)
If you wish to retrain:

text
python
# In Jupyter/Colab:
open Train_plant_disease.ipynb
# Point `train/` and `valid/` directories to your image dataset

# (Training takes several hours on CPU!)
A trained model trained_model.keras is required to run the inference app.

2. Run the Streamlit App
Place main.py, trained_model.keras, and home_page.jpeg in the same directory.

bash
streamlit run main.py
This will launch the interactive web dashboard in your default browser.

3. Using the App
Navigate to "Disease Recognition"

Upload a photo of a diseased or healthy leaf (JPEG/PNG)

Hit "Predict"

See instant prediction and disease name!

🏗️ Model Architecture
Multiple Conv2D and MaxPool2D blocks (32→512 filters)

Flatten + Dense layers (1500 units)

Output: 38-class softmax

Dropout layers for regularization

Adam optimizer, categorical_crossentropy loss

10 epochs, learning rate 1e-4

Validation accuracy achieved: >96%

📈 Results
Training accuracy: up to 98.9%

Validation accuracy: up to 96.4%

Rapid inference suitable for real-time usage

📃 Citation
Dataset:
Vipooool, "New Plant Diseases Dataset", Kaggle, https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

🤝 Acknowledgements
Kaggle PlantVillage Dataset

TensorFlow, Keras, Streamlit

🔒 License
MIT License (you may define your own)

💡 Contact & Contributing
Pull requests and suggestions welcome!
Created by [Saransh Vaidya]
For questions and contributions, open an Issue or Pull Request on the repository.

