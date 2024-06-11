import requests
from bs4 import BeautifulSoup
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify, session
from flask_session import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Knowledge base
knowledge_base = {
    "bercak daun": {
        "pengertian": "Bercak daun adalah penyakit pada daun tanaman yang disebabkan oleh infeksi jamur atau bakteri, ditandai dengan munculnya bercak-bercak kecil berwarna cokelat atau hitam.",
        "gejala": "Munculnya bercak-bercak kecil berwarna cokelat atau hitam pada daun.",
        "penyebab": "Infeksi jamur atau bakteri.",
        "cara mengatasi": "Menggunakan fungisida yang sesuai, menjaga kebersihan lingkungan sekitar pohon, dan memangkas bagian yang terinfeksi."
    },
    "busuk buah": {
        "pengertian": "Busuk buah adalah penyakit pada buah apel yang disebabkan oleh infeksi jamur seperti Botrytis atau Monilinia, menyebabkan buah menjadi membusuk dan berwarna cokelat atau hitam.",
        "gejala": "Buah apel membusuk dan berwarna cokelat atau hitam.",
        "penyebab": "Infeksi jamur seperti Botrytis atau Monilinia.",
        "cara mengatasi": "Membuang buah yang terinfeksi, menggunakan fungisida, dan menjaga kebersihan kebun."
    },
    "kanker batang": {
        "pengertian": "Kanker batang adalah penyakit pada batang pohon yang disebabkan oleh infeksi jamur Nectria galligena, ditandai dengan luka pada batang yang mengeluarkan getah dan menyebabkan batang membengkak dan mati.",
        "gejala": "Luka pada batang pohon yang mengeluarkan getah, batang membengkak dan mati.",
        "penyebab": "Infeksi jamur Nectria galligena.",
        "cara mengatasi": "Memangkas bagian yang terinfeksi, menggunakan fungisida, dan menjaga kebersihan kebun."
    },
    "mati pucuk": {
        "pengertian": "Mati pucuk adalah penyakit pada tanaman apel yang menyebabkan ujung-ujung cabang mengering dan mati, biasanya disebabkan oleh infeksi jamur atau bakteri.",
        "gejala": "Ujung-ujung cabang mengering dan mati.",
        "penyebab": "Infeksi jamur atau bakteri.",
        "cara mengatasi": "Memangkas bagian yang terinfeksi, menggunakan fungisida atau bakterisida, dan menjaga kebersihan kebun."
    }
}


# Fungsi untuk melakukan web scraping dan mendapatkan teks
def scrape_apple_diseases(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    paragraphs = soup.find_all('p')
    text = "\n".join([para.get_text() for para in paragraphs])
    return text

# Fungsi untuk mengekstrak teks dari beberapa PDF
def extract_text_from_pdfs(pdf_paths):
    all_text = ""
    for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                all_text += page.extract_text() + "\n"
    return all_text

# Fungsi untuk membersihkan dan memproses teks
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

# Temukan jawaban yang relevan menggunakan TF-IDF dan cosine similarity
def find_answer_tfidf(sentences, query):
    tfidf_vectorizer = TfidfVectorizer().fit_transform(sentences + [query])
    cosine_similarities = cosine_similarity(tfidf_vectorizer[-1], tfidf_vectorizer[:-1]).flatten()
    most_similar_sentence_index = np.argmax(cosine_similarities)
    return sentences[most_similar_sentence_index]

# Temukan nama penyakit berdasarkan gejala
def find_disease_from_symptom(query):
    for disease, info in knowledge_base.items():
        if any(symptom in query.lower() for symptom in info['gejala'].lower().split(", ")):
            return disease
    return None

# Temukan jawaban yang relevan dari knowledge base
def find_answer_from_knowledge_base(query):
    if "apa saja" in query.lower():
        return ", ".join(knowledge_base.keys())
    query_words = set(query.lower().split())
    
    for key, value in knowledge_base.items():
        key_words = set(key.split())
        if query_words & key_words:
            if key in query.lower():
                if "gejala" in query.lower():
                 return value["gejala"]
            elif "mengalami" in query.lower():
                return value["pengertian"]
            elif "penyebab" in query.lower():
                return value["penyebab"]
            elif "menyebabkan" in query.lower():
                return value["penyebab"]
            elif "cara mengatasi" in query.lower():
                return value["cara mengatasi"]
            elif "mengobati" in query.lower():
                return value["cara mengatasi"]
            elif "mengatasi" in query.lower():
                return value["cara mengatasi"]
            elif "apa itu" in query.lower():
                return value["pengertian"]
            elif "pengertian" in query.lower():
                return value["pengertian"]
            else:
                return f"Pengertian: {value['pengertian']}, Gejala: {value['gejala']}, Penyebab: {value['penyebab']}, Cara mengatasi: {value['cara mengatasi']}"
            
# Pencocokan kata kunci parsial
    for key, value in knowledge_base.items():
        if any(word in query.lower() for word in key.split()):
            return f"Pengertian: {value['pengertian']}, Gejala: {value['gejala']}, Penyebab: {value['penyebab']}, Cara mengatasi: {value['cara mengatasi']}"
    return None

# Inisialisasi Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# URL dari website yang memiliki informasi tentang penyakit pada daun apel
scrap_result = []
url = ["https://www.kompas.com/homey/read/2022/09/04/095956376/ketahui-ini-4-penyakit-yang-menyerang-tanaman-apel", "https://www.idntimes.com/science/discovery/timmy-si-penulis/penyakit-yang-menyerang-tanaman-apel-1"]
for content in url :
    scraped_text = scrape_apple_diseases(content)
    scrap_result.append(scraped_text)

# Paths dari PDF lokal
pdf_paths = ["D:/API/Knowledge/Penyakit_Apel_Data_ChatBot.pdf"]  # Ganti dengan jalur PDF yang benar
pdf_text = extract_text_from_pdfs(pdf_paths)

# Gabungkan teks dari knowledge base, website, dan PDF
knowledge_text = "\n".join([f"{k}: {v}" for k, v in knowledge_base.items()])
combined_text = knowledge_text + "\n" + scraped_text + "\n" + pdf_text
sentences = preprocess_text(combined_text)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        session.clear()
        return jsonify({'greeting': 'Halo, adakah yang bisa saya bantu seputar penyakit pada daun apel?'})
    
    elif request.method == 'POST':
        user_input = request.json.get('query')
        if not user_input:
            return jsonify({'error': 'No query provided'}), 400
        
        if 'history' not in session:
            session['history'] = []

        session['history'].append(user_input)
        context_query = " ".join(session['history'])

        # Logika untuk mendeteksi sapaan
        greetings = ["hi", "hello", "hai", "halo", "hey"]
        assistance_requests = [
            "bisa kah kamu membantu saya",
            "dapatkah kamu membantu saya",
            "bisa kamu membantu saya",
            "bantu saya",
            "butuh bantuan",
            "tolong bantu saya",
            "dapat kamu membantu saya",
            "coba berikan contoh",
            "tolong dong bantu saya",
            "saya mau nanya nih",
            "saya mau nanya",
            "mau nanya sesuatu",
            "bisa bantu saya",
            "mau nanya dong",
            "mau nanya",
            "jadi gini",
            "saya ingin bertanya",
            "ingin bertanya",
            "ingin bertanya sesuatu",
            "saya ingin bertanya",
            "pengen nanya"
        ]

        if any(greet in user_input.lower() for greet in greetings):
            return jsonify({'answer': 'Halo! Bagaimana saya bisa membantu Anda hari ini?'})
        
        if any(phrase in user_input.lower() for phrase in assistance_requests):
            return jsonify({'answer': 'Tentu, saya di sini untuk membantu Anda. Apa yang bisa saya bantu?'})
        
        # Cek knowledge base untuk jawaban yang relevan
        answer_from_kb = find_answer_from_knowledge_base(user_input)
        if answer_from_kb:
            return jsonify({'answer': "Ini jawabannya: " + answer_from_kb})
        
        # Jika tidak ditemukan di knowledge base, coba cari nama penyakit berdasarkan gejala
        disease_from_symptom = find_disease_from_symptom(user_input)
        if disease_from_symptom:
            return jsonify({'answer': f'Berdasarkan gejala yang Anda sebutkan, kemungkinan penyakitnya adalah {disease_from_symptom}.'})
        
        # Jika tidak ditemukan di knowledge base atau berdasarkan gejala, gunakan TF-IDF dan cosine similarity
        answer_from_tfidf = find_answer_tfidf(sentences, user_input)
        return jsonify({'answer': "Pertanyaan Tersebut Diluar Tema dari ChatBot"})

if __name__ == '__main__':
    app.run(debug=True, port=8081)





 




    
