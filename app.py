import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import requests
import joblib
import tempfile
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from dataclasses import dataclass
from typing import Any, Dict
import tensorflow as tf
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'png', 'jpeg'])
CORS(app)

# bikin koneksi firebase disini

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
    model = keras.models.load_model('Model/apple_maturity_cnn_model2.h5')
    return model

def processImage(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    print("Shape after processing: ", img_hsv.shape)
    img_normalized = tf.cast(img_hsv, tf.float32) / 255.0
    img_batched = tf.expand_dims(img_normalized, axis=0)
    return img_batched

def predict_class(images):
    print("in predict")
    model = loadmodelCNN()
    print("model loaded")
    predictions = model.predict(images)
    predicted_class = tf.argmax(predictions, axis=1).numpy()
    print("success")

    return predicted_class

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

def plant_recommendation(predictiion):
    try:
        # maturity = MaturityData(**input_data)
        # ambil value input dari iot
        data = {
            "kelembaban": maturity.kelembapan,
            "suhu": maturity.suhu,
            "kematangan": prediction
        }
        model, column_transformer = loadModelDT()
        encoded_data = preprocess_input(data, column_transformer)
        prediction = make_predictions(encoded_data, model)

        return jsonify({
            "data":{
                "maturityEstimation": prediction[0]
                },
            "status":{
                    "code":200,
                    "message":"successfully estimationing maturity"
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
            print("processed image shape :", processed_image.shape)
            predicted_class = predict_class(processed_image)
            print("predicted : ", predicted_class[0])

            prediction = str(predicted_class[0])

            os.remove(temp_path)
            os.rmdir(temp_dir)
            return jsonify({
                "data": {
                    "kematangan": prediction
                }, 
                "status": {
                    "code": 200,
                    "message": "successfully Predict Maturity of Apple"
                },
            }), 200
        else:
            return jsonify({
                "error": "image file needed"
                }), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))