# 🖼️ Image Caption Generator (CNN + LSTM, Flask UI)

This project implements a **deep learning-based image captioning system** with a clean web interface built using **Flask + HTML/CSS**. The backend uses a pre-trained **CNN+LSTM model** and **DenseNet201** for feature extraction, and a custom-trained tokenizer.

## 🧠 Key Features

- Upload an image and generate a descriptive caption using AI
- Pre-trained CNN+LSTM model with DenseNet201 as feature extractor
- Web-based frontend (HTML + CSS) with image preview
- Flask-based Python backend with model inference

## 🗂️ Folder Structure

```
project/
│
├── app.py                 # Main Flask app
├── cnn_lstm_final.h5      # Trained CNN+LSTM captioning model
├── cnn_tokenizer.pkl      # Trained tokenizer used during model training
│
├── static/
│   └── uploads/           # Uploaded image files
│
├── templates/
│   └── index.html         # Frontend UI
│
├── requirements.txt       # Python dependencies
└── README.md              # This file

```

## 🚀 How to Run

### 1. Clone the repo and move into it:

```bash
git clone https://github.com/ankush-48/image-captioning-ui.git
cd image-captioning-ui 
````

### 2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the Flask server:

```bash
python app.py
```

### 5. Visit the web app:

Open your browser and go to:
📍 `http://127.0.0.1:5000/`

## 🧠 How It Works

1. Upload an image using the UI.
2. Flask saves it to `static/uploads/`.
3. The model extracts features using DenseNet201.
4. Caption is generated word-by-word using a trained LSTM model.
5. Final caption is displayed below the image.

## 📝 Model Details

* **Feature Extractor**: DenseNet201 (pre-trained on ImageNet)
* **Caption Generator**: LSTM-based decoder trained on image-text pairs
* **Tokenizer**: Custom-trained tokenizer (`cnn_tokenizer.pkl`)
* **Max Caption Length**: 34 tokens (change in `app.py` if needed)

## 🧪 Sample Output

> 🖼️ Uploaded Image → 📝 Generated Caption:
> “a group of people standing next to a building with lights”

## ⚠️ Notes

* Ensure `cnn_lstm_final.h5` and `cnn_tokenizer.pkl` are present in the root directory.
* This project does not train a model — only uses a trained one for inference.
* Heavy model files are not tracked in GitHub (use Google Drive or similar to share).

## ✅ To-Do (Optional Enhancements)

* Add attention visualization to improve interpretability
* Integrate HuggingFace transformers for advanced generation
* Deploy using Docker or on platforms like Vercel/Fly.io
