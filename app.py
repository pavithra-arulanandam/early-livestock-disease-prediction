from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load("disease_prediction_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Form Page
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Process Form and Show Result
@app.route('/result', methods=['POST'])
def result():
    animal_type = request.form['animal_type']
    age = int(request.form['age'])
    temperature = float(request.form['temperature'])
    sym1 = request.form.get('sym1', None)
    sym2 = request.form.get('sym2', None)
    sym3 = request.form.get('sym3', None)

    def encode_input(value, col):
        le = label_encoders[col]
        return le.transform([value])[0] if value in le.classes_ else 0

    encoded_input = [
        encode_input(animal_type, 'Animal_Type'),
        age,
        temperature,
        encode_input(sym1, 'symptom1') if sym1 else 0,
        encode_input(sym2, 'symptom2') if sym2 else 0,
        encode_input(sym3, 'symptom3') if sym3 else 0,
    ]

    disease_code = model.predict([encoded_input])[0]
    predicted_disease = label_encoders['Disease'].inverse_transform([disease_code])[0]

    return render_template('result.html', disease=predicted_disease)

if __name__ == '__main__':

    app.run(debug=True)
