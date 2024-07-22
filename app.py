from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Define a mapping for smoking history categories
smoking_history_map = {
    'never': 0,
    'former': 1,
    'current': 2,
    'not current': 0,  # Map 'not current' to 'never'
    'No Info': 0,      # Map 'No Info' to 'never'
    'ever': 1          # Map 'ever' to 'former'
}

# Define a mapping for gender categories
gender_map = {
    'Male': 0,
    'Female': 1
}

# Define labels for prediction outcomes
prediction_labels = {
    0: "Low Diabetes Risk",
    1: "High Diabetes Risk"
}

# Define a route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = request.form['smoking_history']
        bmi = float(request.form['bmi'])
        HbA1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])
        
        # Map smoking history and gender to numerical categories
        smoking_history = smoking_history_map.get(smoking_history, 0)  # Default to 0 if not found
        gender = gender_map.get(gender, 0)  # Default to 0 if not found
        
        # Create DataFrame from input data
        data = {
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [HbA1c_level],
            'blood_glucose_level': [blood_glucose_level]
        }
        
        data_df = pd.DataFrame(data)
        
        # Make prediction
        prediction = model.predict(data_df)
        
        # Prepare response with prediction label
        prediction_label = prediction_labels.get(prediction[0], "Unknown")
        output = {
            'prediction': prediction_label
        }
        
        return jsonify(output)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Define a home route to render the HTML form for user input
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
