import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (Change file path accordingly)
file_path = "c:\\Users\\WELCOME\\Documents\\miniproject\\animal_disease_dataset.csv"  # Update this with the correct CSV file path
df = pd.read_csv(file_path)  # Load CSV instead of Excel

# Preprocessing
df.fillna("Unknown", inplace=True)  # Fill missing values if any
label_encoders = {}  # Dictionary to store encoders

# Encode categorical variables
for col in ['Animal_Type', 'symptom1', 'symptom2', 'symptom3', 'Disease']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoder for later use

# Split data
X = df[['Animal_Type', 'Age', 'Temperature', 'symptom1', 'symptom2', 'symptom3']]
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and encoders
joblib.dump(model, "disease_prediction_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Function for prediction
def predict_disease(animal_type, age, temp, sym1=None, sym2=None, sym3=None):
    # Load saved model and encoders
    model = joblib.load("disease_prediction_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")

    # Encode inputs safely (avoid errors for unknown inputs)
    def encode_input(value, col):
        le = label_encoders[col]
        return le.transform([value])[0] if value in le.classes_ else 0

    encoded_input = [
        encode_input(animal_type, 'Animal_Type'),
        age,
        temp,
        encode_input(sym1, 'symptom1') if sym1 else 0,
        encode_input(sym2, 'symptom2') if sym2 else 0,
        encode_input(sym3, 'symptom3') if sym3 else 0,
    ]

    # Predict
    disease_code = model.predict([encoded_input])[0]
    predicted_disease = label_encoders['Disease'].inverse_transform([disease_code])[0]
    
    # If predicted disease is 'Unknown' or not found, return the default message
    if predicted_disease.lower() == "unknown":
        return "Disease not found, you may consult a doctor."
    return predicted_disease

# Example usage
animal_type = input("Enter Animal Type: ")
age = int(input("Enter Age: "))
temperature = float(input("Enter Temperature: "))

sym1 = input("Enter Symptom 1 (or press Enter to skip): ") or None
sym2 = input("Enter Symptom 2 (or press Enter to skip): ") or None
sym3 = input("Enter Symptom 3 (or press Enter to skip): ") or None

predicted_disease = predict_disease(animal_type, age, temperature, sym1, sym2, sym3)
print(f"Predicted Disease: {predicted_disease}")