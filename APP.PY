from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Dummy model integration (Replace this with your actual model)
# def generate_caption(image):
#     return "This is a dummy caption. Replace this with your model output."

# def preprocess_image(file_path):
#     image = Image.open(file_path).convert("RGB")
#     return image

# Load trained model and components
print("Loading trained model...")

try:
    model = load_model('cnn_lstm_final.h5')
    
    # Load feature extractor (DenseNet201)
    densenet = DenseNet201()
    feature_extractor = Model(inputs=densenet.input, outputs=densenet.layers[-2].output)
    
    # Load tokenizer (you'll need to save this during training)
    try:
        with open('cnn_tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    
        # Get vocab size and max length from tokenizer
        vocab_size = len(tokenizer.word_index) + 1
        print(f"Vocabulary size: {vocab_size}")

    except FileNotFoundError:
        print("Tokenizer not found. Creating dummy tokenizer for demo.")
        tokenizer = None
    
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_extractor = None
    tokenizer = None
    vocab_size = None

def idx_to_word(integer, tokenizer):
    """Convert integer to word using tokenizer"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_features(image_path):
    """Extract features from image using DenseNet201"""
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        features = feature_extractor.predict(image, verbose=0)
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def generate_caption(image_path):
    """Generate caption using trained CNN+LSTM model"""
    if model is None or feature_extractor is None or tokenizer is None:
        return "Model not loaded. Please check if model files exist."
    
    try:
        # Extract image features
        features = extract_features(image_path)
        if features is None:
            return "Error processing image."
        
        if tokenizer is None:
            return "Caption: A beautiful scene captured in the image."
        
        # Generate caption word by word
        max_length = 34  # Adjust based on your training
        in_text = 'startseq'
        
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            
            pred = model.predict([features, sequence], verbose=0)
            pred_idx = np.argmax(pred)
            
            # word = None
            # for w, idx in tokenizer.word_index.items():
            #     if idx == pred_idx:
            #         word = w
            #         break

            word = idx_to_word(pred_idx, tokenizer)
            
            if word is None or word == 'endseq':
                break
                
            in_text += ' ' + word
        
        # Clean up caption
        caption = in_text.replace('startseq', '').strip()
        return caption if caption else "A scene captured in the image."
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Error generating caption. Please try again."

def preprocess_image(file_path):
    """Preprocess image for display"""
    try:
        image = Image.open(file_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         file = request.files['image']
#         if file and file.filename:
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
#             image = preprocess_image(filepath)
#             caption = generate_caption(image)
#             return render_template('index.html', filename=filename, caption=caption)
#     return render_template('index.html', filename=None, caption=None)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Generate caption using the trained model
            caption = generate_caption(filepath)
            
            return render_template('index.html', filename=filename, caption=caption)
    return render_template('index.html', filename=None, caption=None)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
