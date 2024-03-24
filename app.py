from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('Precipitation Prediction.pkl')

# Function to check if a string is numeric
def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# A route to render the form
@app.route('/')
def index():
    return render_template('index.html')

# a route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    date = request.form['date']
    temp_max = request.form['temp_max']
    temp_min = request.form['temp_min']
    wind = request.form['wind']
    
    # Check if input for numeric features are numeric
    if not all(map(is_numeric, [temp_max, temp_min, wind])):
        return render_template('index.html', error_message='Please enter only numeric values for temperature and wind.')
    
    # Preprocess the input data
    temp_max = float(temp_max)
    temp_min = float(temp_min)
    wind = float(wind)
    input_data = pd.DataFrame({'date': [date], 'temp_max': [temp_max], 'temp_min': [temp_min], 'wind': [wind]})
    
    # Make predictions
    prediction = model.predict(input_data)
    
    # Display the prediction result
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
