## API Projek Kekhususan

### API Prediksi Kematangan Apel dan Estimasi Kematangan

End point dibawah ini dipakai untuk menjalankan model cnn dan juga decision tree regressor, yang fungsinya adalah untuk mengklasifikasi apel menjadi label (Matang, Setengah Matang, dan Belum Matang), setelah nya hasil dari model CNN akan dipakai oleh model Decision Tree Regressor untuk mengestimasi kematangan dengan memperhatikan faktor kematangan buah, suhu dan juga kelembaban
 * #### URL : 
    * http://34.128.114.101:8081/
  * #### Method :

| Method                                                           | Path                   |
| ---------------------------------------------------------------- | --------------------- |
| ![](https://storage.kodeteks.com/POST.png)                       | `/api/predict`       |

### Example Result
  ```json
    {
        "data": {
            "Estimasi": 60.0,
            "Kematangan": "Belum Matang"
    },
        "status": {
            "code": 200,
            "message": "Berhasil memprediksi kematangan apel"
        }
    }
 ```

### Input Data dari Firebase
  ```json
    {
        "image": "base64",
        "kelembaban": "sensor humadity",
        "temperature": "sensor temperature"
    }
 ```

## API Chatbot

 * #### URL : 
    * http://34.128.114.101:8081/
  * #### Method :

| Method                                                           | Path                   |   Description             |
| ---------------------------------------------------------------- | --------------------- |--------------------- |
| ![](https://storage.kodeteks.com/POST.png)                       | `/chatbot`       | Request Method Post berfungsi untuk mengirimkan pertanyaan ke chatbot  |
|![](https://pub-cc8247a7807d42d1bd2453b3dae2f678.r2.dev/GET.png)  | `/chatbot` |Request Method Get berfungsi untuk awalan dari chatbot, yaitu berupa sapaan  |

### Example Result GET
  ```json
     {
        "greeting": "Hi, how can I help you about apple disease?"
    }
 ```

### Example Query chatbot POST
  ```json
     {
        "query":"my apple have some spot on leaf, what kind of disease is that?"
      }
 ```

### Example Result chatbot POST
  ```json
     {
        "answer": "This is the answer: Leaf spot is a plant disease characterized by small brown or black spots on leaves, caused by fungal or bacterial infections. These spots can lead to premature leaf drop and reduced photosynthesis, weakening the plant."
    }
 ```

### Example Bad Query
  ```json
     {
        "query":"what is my name"
    }
 ```

### Example Output
  ```json
     {
        "answer": "That question is out of the chatbot's knowledge"
    }
 ```

