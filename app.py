import os
import tempfile
import cv2
import base64
import json
import firebase_admin.db
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify, session
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


# Inisialisasi Flask app
app = Flask(__name__)
CORS(app)


load_dotenv()

def firebase_config_converter():
    firebasejson = os.getenv('firebasejson')

    if not firebasejson:
        print("Environment variable 'firebasejson' is not set or is empty.")
        return None

    try:
        key = json.loads(firebasejson)
    except json.JSONDecodeError as err:
        print(f"Failed to decode JSON: {err}")
        return None

    return key
keyjson=firebase_config_converter()
cred = credentials.Certificate(keyjson)
firebase_admin.initialize_app(cred, {'databaseURL': os.environ.get('firebaseUrl')})
# Load model CNN
def loadmodelCNN():
    model = keras.models.load_model('Model/model_Ripeness(4) (1).h5')
    return model

# Proses gambar
def processImage(image, target_size=(128, 128)):
    # Konversi base64 ke gambar PIL
    image = Image.open(io.BytesIO(image))
    
    # Simpan gambar ke file sementara
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'temp_image.png')
    image.save(temp_path)
    
    # Resize dan konversi gambar ke HSV
    img = np.array(image)
    img_resized = cv2.resize(img, target_size)
    # img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    img_normalized = tf.cast(img_resized, tf.float32) / 255.0
    img_batched = tf.expand_dims(img_normalized, axis=0)
    
    return img_batched, temp_path, temp_dir

# Prediksi kelas
def predict_class(images):
    model = loadmodelCNN()
    predictions = model.predict(images)
    predicted_class = tf.argmax(predictions, axis=1).numpy()
    return predicted_class

# Load model Regressor
def loadModelDTRegressor():
    model = joblib.load('Model/decision_tree_Regressor_model.pkl')
    label_encoder = joblib.load('Model/label_encoder (1).pkl')
    return model, label_encoder

# Preprocessing input
def preprocess_input(input_data, label_encoder):
    input_df = pd.DataFrame([input_data])
    input_df['Ripeness'] = label_encoder.transform(input_df['Ripeness'])
    input_encoded = input_df[['Temperature', 'Humidity', 'Ripeness']]
    return input_encoded

# Prediksi menggunakan model Regressor
def make_predictions(input_encoded, model):
    predictions = model.predict(input_encoded)
    return predictions.tolist()

# Konversi prediksi menjadi label kematangan
def switch(prediction):
    if prediction == 0:
        return "Ripe"
    elif prediction == 1:
        return "Half Ripe"
    elif prediction == 2:
        return "Immature"

# Fungsi untuk mengambil data suhu dan kelembaban dari Firebase
def get_sensor_data():
    ref = db.reference('sensor')
    sensor_data = ref.get()
    suhu = sensor_data.get('temperature', None)
    kelembaban = sensor_data.get('kelembaban', None)
    return suhu, kelembaban

# Route untuk halaman utama
@app.route("/", methods=['GET'])
def homepage():
    try:
        with open("static/index.html", "r") as file:
            return file.read()
    except IOError as e:
        return "Error: File tidak ditemukan", 500

def getImage():
    ref = db.reference('image')
    image = ref.get()
    return image

def string_to_base64(base64_string):
    # Mengonversi string ke byte
    if ',' in base64_string:
        base64_string = base64_string.split(",")[1]
    
    # Decode base64 string
    image_data = base64.b64decode(base64_string)
    return image_data

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Knowledge base
knowledge_base = {
    "leaf spot": {
        "definition": "Leaf spot is a plant disease characterized by small brown or black spots on leaves, caused by fungal or bacterial infections. These spots can lead to premature leaf drop and reduced photosynthesis, weakening the plant.",
        "symptoms": "The appearance of small brown or black spots on leaves. Over time, these spots may enlarge, merge, and cause significant leaf damage.",
        "causes": "Fungal or bacterial infections. Common pathogens include fungi like Septoria and Cercospora, and bacteria like Xanthomonas.",
        "control_methods": "Use appropriate fungicides, maintain cleanliness around the tree, and prune infected parts. Ensure proper air circulation and avoid overhead watering to reduce moisture on leaves, which can promote infection."
    },
    "fruit rot": {
        "definition": "Fruit rot is a disease affecting apples, caused by fungal infections such as Botrytis or Monilinia, resulting in rotting and brown or black discoloration of the fruit. It can lead to significant crop losses both pre- and post-harvest.",
        "symptoms": "Apples become rotten and turn brown or black. Affected fruits may develop a fuzzy or moldy appearance due to fungal growth.",
        "causes": "Fungal infections like Botrytis or Monilinia. These fungi thrive in warm, humid conditions and can spread rapidly through contact or air.",
        "control_methods": "Remove infected fruits, use fungicides, and maintain garden hygiene. Store fruits in a cool, dry place and handle them carefully to prevent wounds that can serve as entry points for pathogens."
    },
    "stem cancer": {
        "definition": "Stem canker is a disease affecting tree trunks, caused by the fungal infection Nectria galligena, marked by lesions that exude sap and cause swelling and death of the trunk. It can significantly impair the structural integrity and vitality of the tree.",
        "symptoms": "Lesions on the tree trunk that exude sap, swelling, and eventual death of the trunk. Over time, these cankers may enlarge, girdle the trunk, and cut off nutrient flow.",
        "causes": "Fungal infection by Nectria galligena. This pathogen can enter through wounds or natural openings in the bark.",
        "control_methods": "Prune infected parts, use fungicides, and maintain garden hygiene. Ensure proper pruning techniques to avoid creating large wounds and promote quick healing."
    },
    "tip blight": {
        "definition": "Tip blight is a disease in apple trees causing the tips of branches to dry out and die, usually due to fungal or bacterial infections. It can lead to reduced growth and productivity of the tree.",
        "symptoms": "The tips of branches dry out and die. Affected areas may also exhibit discoloration, wilting, and dieback of the shoots.",
        "causes": "Fungal or bacterial infections. Common pathogens include fungi like Diplodia and bacteria like Pseudomonas.",
        "control_methods": "Prune infected parts, use fungicides or bactericides, and maintain garden hygiene. Avoid excessive nitrogen fertilization, which can make the tree more susceptible to infections."
    }
}

# Function to preprocess text
def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    cleaned_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalpha()]
        words = [word for word in words if word not in stop_words]
        words = [stemmer.stem(word) for word in words]
        cleaned_sentence = ' '.join(words)
        cleaned_sentences.append(cleaned_sentence)
    
    return cleaned_sentences

# Find relevant answer using TF-IDF and cosine similarity
# def find_answer_tfidf(sentences, query):
#     tfidf_vectorizer = TfidfVectorizer().fit_transform(sentences + [query])
#     cosine_similarities = cosine_similarity(tfidf_vectorizer[-1], tfidf_vectorizer[:-1]).flatten()
#     most_similar_sentence_index = np.argmax(cosine_similarities)
#     return sentences[most_similar_sentence_index]

# Find disease name based on symptoms
def find_disease_from_symptom(query):
    for disease, info in knowledge_base.items():
        if any(symptom in query.lower() for symptom in info['symptoms'].lower().split(", ")):
            return disease
    return None

# Find relevant answer from knowledge base
def find_answer_from_knowledge_base(query):
    if "disease" in query.lower():
        return ", ".join(knowledge_base.keys())
    query_words = set(query.lower().split())
    
    for key, value in knowledge_base.items():
        key_words = set(key.split())
        if query_words & key_words:
            if key in query.lower():
                if "symptoms" in query.lower():
                 return value["symptoms"]
            elif "causes" in query.lower():
                return value["causes"]
            elif "cause" in query.lower():
                return value["causes"]
            elif "control methods" in query.lower():
                return value["control_methods"]
            elif "control" in query.lower():
                return value["control_methods"]
            elif "treat" in query.lower():
                return value["control_methods"]
            elif "overcome" in query.lower():
                return value["control_methods"]
            elif "what is" in query.lower():
                return value["definition"]
            elif "definition" in query.lower():
                return value["definition"]
            elif "have" in query.lower():
                return value["definition"]
            else:
                return f"definition: {value['definition']}, symptoms: {value['symptoms']}, causes: {value['causes']}, control_methods: {value['control_methods']}"
            
# Partial keyword matching
    for key, value in knowledge_base.items():
        if any(word in query.lower() for word in key.split()):
            return f"definition: {value['definition']}, symptoms: {value['symptoms']}, causes: {value['causes']}, control_methods: {value['control_methods']}"
    return None

# Initialize Flask
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your_secret_key'
# app.config['SESSION_TYPE'] = 'filesystem'
# Session(app)

# Preprocess the knowledge base text
knowledge_text = "\n".join([f"{k}: {v}" for k, v in knowledge_base.items()])
sentences = preprocess_text(knowledge_text)

@app.route("/api/predict", methods=['POST'])
def ripness_prediction():
        imageString = getImage()
        image64 = string_to_base64(imageString)
        img_batched, temp_path, temp_dir = processImage(image64)
        predicted_class = predict_class(img_batched)
        prediction = int(predicted_class[0])
        kematangan = switch(prediction)
        

        # Ambil data suhu dan kelembaban dari Firebase
        suhu, kelembaban = get_sensor_data()
        if suhu is None or kelembaban is None:
            return jsonify({
                "error": "Data suhu atau kelembaban tidak ditemukan di Firebase"
            }), 400
        print(suhu, kelembaban)
        data = {
            "Humidity": float(kelembaban),
            "Temperature": float(suhu),
            "Ripeness": kematangan
        }

        #Load model Regressor dan encoder
        model, label_encoder = loadModelDTRegressor()
        encoded_data = preprocess_input(data, label_encoder)
        estimasi = make_predictions(encoded_data, model)

        #Hapus file sementara setelah diproses
        os.remove(temp_path)
        os.rmdir(temp_dir)

        return jsonify({
            "data": {
                "Ripeness": kematangan,
                "Estimation": estimasi[0]
            }, 
            "status": {
                "code": 200,
                "message": "Berhasil memprediksi kematangan apel"
            }
        }), 200

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        session.clear()
        return jsonify({'greeting': 'Hi, how can I help you about apple disease?'})
    
    elif request.method == 'POST':
        user_input = request.json.get('query')
        if not user_input:
            return jsonify({'error': 'No query provided'}), 400
        

        # Logic to detect greetings
        greetings = ["hi", "hello", "hey"]
        assistance_requests = [
            "Can you help me?",
            "Could you help me out?",
            "Can you assist me?",
            "Help me, please",
            "I need help",
            "Please, give me a hand",
            "Could you lend me a hand?",
            "Could you give an example?",
            "Please, I need your help",
            "I have a question",
            "I want to ask something",
            "I need to ask something",
            "Can you give me a hand?",
            "I have a quick question",
            "I need to ask",
            "So, here's the thing",
            "I would like to ask",
            "I want to ask",
            "I need to ask something",
            "I have a question",
            "I have something to ask"
        ]

        if any(greet in user_input.lower() for greet in greetings):
            return jsonify({'answer': 'Hi, how can I help you today?'})
        
        if any(phrase in user_input.lower() for phrase in assistance_requests):
            return jsonify({'answer': 'Sure, what can I help you with today about apple disease?'})
        
        # Check the knowledge base for a relevant answer
        answer_from_kb = find_answer_from_knowledge_base(user_input)
        if answer_from_kb:
            return jsonify({'answer': "This is the answer: " + answer_from_kb})
        
        # If not found in the knowledge base, try to find the disease name based on symptoms
        disease_from_symptom = find_disease_from_symptom(user_input)
        if disease_from_symptom:
            return jsonify({'answer': f'Based on your question, this is the probable answer: {disease_from_symptom}.'})
        
        # answer_from_tfidf = find_answer_tfidf(sentences, user_input)
        return jsonify({'answer': "That question is out of the chatbot's knowledge"})
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))
