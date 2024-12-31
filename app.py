from flask import Flask, request, jsonify
import numpy as np
import re
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import os
app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))


model_path = os.path.join(script_dir, "artifacts", "multi_label_intent_model.h5")
tokenizer_path = os.path.join(script_dir, "artifacts", "tokenizer.pkl")
embeddings_path = os.path.join(script_dir, "artifacts", "embeddings.npy")


model = load_model(model_path)
tokenizer = joblib.load(tokenizer_path)
embedding_weights = np.load(embeddings_path)

stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"and"}
lemmatizer = WordNetLemmatizer()

def expand_contractions(text):
    contractions_dict = {
        "don't": "donot",
        "can't": "cannot",
        "won't": "willnot",
        "isn't": "isnot",
        "aren't": "arenot",
        "didn't": "didnot",
        "hasn't": "hasnot",
        "haven't": "havenot",
        "wasn't": "wasnot",
        "weren't": "werenot",
        "shouldn't": "shouldnot",
        "couldn't": "couldnot",
        "wouldn't": "wouldnot",
        "I've": "I have",
        "you've": "you have",
        "they've": "they have",
        "we've": "we have",
        "I'd": "I would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
        "that'll": "that will",
    }
    pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    return pattern.sub(lambda x: contractions_dict[x.group()], text)

def preprocess_query(query):
    query = query.lower()
    query = expand_contractions(query)
    tokens = word_tokenize(query)
    tokens = [word for word in tokens if word.isalnum()]  
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    cleaned_query = " ".join(lemmatized_tokens)
    return cleaned_query


def predict_intent(query):
    cleaned_query = preprocess_query(query)
    X = tokenizer.texts_to_sequences([cleaned_query])
    X = pad_sequences(X, maxlen=100)
    y_pred = model.predict(X)
    print(y_pred)
    y_pred_binary = (y_pred > 0.3).astype(int)
    labels = ["Product Inquiry", "Order Tracking", "Refund Request", "Store Policy"]
    predicted_intents = [labels[i] for i in range(len(labels)) if y_pred_binary[0][i] == 1]
    return predicted_intents

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    query = data["query"]
    predicted_intents = predict_intent(query)
    return jsonify({"predicted_intents": predicted_intents})

if __name__ == "__main__":
    app.run(debug=True)
