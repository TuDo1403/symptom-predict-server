import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        req_data = request.get_json()
        inputs = req_data['inputs']
    df_pivoted = pd.read_csv('data/data.csv')
    symptoms = df_pivoted.columns[1:].values

    with open('data/NB_model.sav', 'rb') as input_file:
        model = pickle.load(input_file)
    user_symptoms = list(inputs.split(','))
    test_input = [0]*397
    for symptom in user_symptoms:
        test_input[np.where(symptoms==symptom)[0][0]] = 1

    response = jsonify({'results': model.predict([test_input]).tolist()})
    return response

if __name__ == '__main__':
    app.run()