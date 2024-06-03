import os
import cv2
import requests
import joblib
import tempfile
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from dataclasses import dataclass
from typing import Any, Dict
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'png', 'jpeg'])
CORS(app)

class MaturityData:
    kelembaban: float
    suhu: float
    kematangan: str

def allowed_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def check_server_availability(destination_url, timeout=30):
    try:
        response = requests.get(destination_url, timeout=timeout)
        if response.status_code == 400:
            return True
        else:
            return False
    except requests.exceptions.Timeout:
        return False

def loadmodelCNN():
    model = joblib.load('asset/Model/apple_maturity_cnn.h5')
    return model

def processImage(images, target_size=(128, 128)):
    img = cv2.imread(images)
    img_resized = cv2.resize(img, target_size)
    img_flat = img_resized.flatten()
    img_2d = img_flat.reshape(1, -1)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    images.append(img_hsv)

def predict_class(images):
    model = loadmodelCNN()
    predictions = model.predict(images)

    return predictions

def loadModelDT():
    model = joblib.load('asset/model/decision_tree/decision_tree_model_v1.1.sav')
    column_transformer = joblib.load('asset/model/decision_tree/column_transformer_v1.1.sav')
    return model, column_transformer

def preprocess_input(input_data, column_transformer):
    # Convert input_data (dictionary) to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Transform the input data using the loaded ColumnTransformer
    input_encoded = column_transformer.transform(input_df)
    print("encoded data : ", input_encoded)
    return input_encoded

def make_predictions(input_encoded, model):
    print("encoded input: ", input_encoded)
    predictions = model.predict(input_encoded)
    return predictions.tolist()

class MaturityData:
    def __init__(self, kelembapan, suhu, kematangan):
        self.kelembapan = kelembapan
        self.suhu = suhu
        self.kematangan = kematangan
@app.before_request
def remove_trailing_slash():
    if request.path != '/' and request.path.endswith('/'):
        return redirect(request.path[:-1])

@app.route("/", methods=['GET'])
def homepage():
    try:
        # Membuka file HTML
        with open("static/index.html", "r") as file:
            return file.read()
    except IOError as e:
        print("Error:", e)
        return "Error: File not found", 500

@app.route("/api/recommendation", methods=['POST'])
def plant_recommendation():
    try:
        input_data = request.get_json()
        if not input_data:
            raise ValueError("No input data provided")

        maturity = MaturityData(**input_data)
        data = {
            "kelembaban": maturity.kelembapan,
            "suhu": maturity.suhu,
            "kematangan": maturity.kematangan
        }
        model, column_transformer = loadModelDT()
        encoded_data = preprocess_input(data, column_transformer)
        prediction = make_predictions(encoded_data, model)

        return jsonify({
            "data":{
                "plantRecommendation": prediction[0]
                },
            "status":{
                    "code":200,
                    "message":"successfully recommending plant"
                }}
            ), 200
    
    except Exception as err:
        app.logger.error(f"handler: bind input error: {err}")
        return jsonify({"error": f"cannot embed data: {err}"}), 400

@app.route("/api/predict", methods = ['POST'])
def soil_prediction():
        image = request.files["image"]
        if image:
            # Membuat file sementara untuk menyimpan file gambar
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, 'temp_image.jpg')
            image.save(temp_path)

            processed_image = processImage(temp_path)
            print(processed_image)
            predicted_class = predict_class(processed_image)
            print("predicted : ", predicted_class[0])

            os.remove(temp_path)
            os.rmdir(temp_dir)
            return jsonify({
                "data": {
                    "jenis_tanah": predicted_class[0]
                }, 
                "status": {
                    "code": 200,
                    "message": "successfully predicted soil type"
                },
            }), 200
        else:
            return jsonify({
                "error": "image file needed"
                }), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))