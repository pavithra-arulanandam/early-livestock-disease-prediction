Early Livestock Animal Disease Prediction Using Machine Learning

This project aims to predict livestock diseases based on animal type, age, and symptoms using a Random Forest Classifier. 
It is designed to assist farmers and veterinary professionals by providing early disease detection to improve animal health and reduce treatment costs.

Features
- Predict disease based on animal type, age, temperature, and symptoms
- Trained using a labeled dataset of livestock symptoms and diseases
- Web interface built with Flask, HTML/CSS
- Supports input validation and user-friendly interaction
- Results shown with high accuracy (80%+)

Project Overview
- Frontend: HTML/CSS for UI
- Backend: Python Flask
- Model: Random Forest Classifier from sklearn
- Deployment: Local server

Project Structure
early-livestock-disease-prediction/
├── app/
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   └── styles.css
├── model/
│   └── rf_model.pkl
├── data/
│   └── (Place dataset here after downloading from Kaggle)
├── app.py
├── requirements.txt
├── README.md

Dataset
The dataset used for training was sourced from Kaggle.
License: The dataset’s license is marked as Unknown, so it is not included in this repository.

To use this project:
- Download the dataset from Kaggle manually.
- Place the file inside the /data folder.

How to Run the Project
1. Clone the Repository
   git clone https://github.com/yourusername/early-livestock-disease-prediction.git
   cd early-livestock-disease-prediction

2. Install Required Packages
   pip install -r requirements.txt

3. Run the Flask App
   python app.py

4. Open in Browser
   Go to http://127.0.0.1:5000 to use the app.

Future Enhancements
- Add more symptoms and disease classes
- Improve model performance with deep learning
- Deploy online using platforms like Heroku or Render
- Add multilingual support for rural accessibility

About Me
I’m Pavithra, a Computer Science student passionate about data, design, and development.
Let’s connect and learn together!

Email: pavithraarulanandam@gmail.com

License
This project is for educational purposes only. The original dataset belongs to its respective author(s) on Kaggle.
