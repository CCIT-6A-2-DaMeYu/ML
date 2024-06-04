import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK resources
nltk.download('punkt')

# Fungsi untuk melakukan web scraping dan mendapatkan teks
def scrape_apple_diseases(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Misalnya kita mencari semua paragraf di dalam artikel
    paragraphs = soup.find_all('p')
    text = "\n".join([para.get_text() for para in paragraphs])
    return text

# URL dari website yang memiliki informasi tentang penyakit pada daun apel
url = "https://www.kompas.com/homey/read/2022/09/04/095956376/ketahui-ini-4-penyakit-yang-menyerang-tanaman-apel"
scraped_text = scrape_apple_diseases(url)

# Fungsi untuk memproses teks menjadi daftar kalimat
def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

# Preprocessing teks yang di-scrape
sentences = preprocess_text(scraped_text)

# Fungsi untuk menemukan jawaban yang relevan menggunakan TF-IDF dan cosine similarity
def find_answer_tfidf(sentences, query):
    tfidf_vectorizer = TfidfVectorizer().fit_transform(sentences + [query])
    cosine_similarities = cosine_similarity(tfidf_vectorizer[-1], tfidf_vectorizer[:-1]).flatten()
    most_similar_sentence_index = np.argmax(cosine_similarities)
    return sentences[most_similar_sentence_index]

# Inisialisasi Flask
app = Flask(__name__)

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        # Memberikan sapaan awal
        return jsonify({'greeting': 'Halo, adakah yang bisa saya bantu seputar penyakit pada apel?'})

    elif request.method == 'POST':
        user_input = request.json.get('query')
        if not user_input:
            return jsonify({'error': 'No query provided'}), 400

        answer = find_answer_tfidf(sentences, user_input)
        return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, port=8081)
