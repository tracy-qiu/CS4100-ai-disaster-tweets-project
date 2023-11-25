import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import *
import joblib

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

import numpy as np

import os

# Get directory and full paths
script_directory = os.path.dirname(os.path.abspath(__file__))

# Load models
naive_bayes_full_path = os.path.join(script_directory, "naive_bayes_model.pk1")
naive_bayes_model = joblib.load(naive_bayes_full_path)
logistic_regression_full_path = os.path.join(script_directory, "logistic_regression_model.pk1")
logistic_regression_model = joblib.load(logistic_regression_full_path)
feed_forward_full_path = os.path.join(script_directory, "feed_forward_model.pk1")
feed_forward_model = joblib.load(feed_forward_full_path)
lstm_full_path = os.path.join(script_directory, "lstm_model.pk1")
lstm_model = joblib.load(lstm_full_path)
bidirectional_lstm_full_path = os.path.join(script_directory, "bidirectional_lstm_model.pk1")
bidirectional_lstm_model = joblib.load(bidirectional_lstm_full_path)


# Load vectorizers - SAME VECTORIZER USED TO TRAIN MODELS MUST BE USED
vectorizer_full_path = os.path.join(script_directory, "vectorizer.pk1")
vectorizer = joblib.load(vectorizer_full_path)
word2vec_vectorizer_full_path = os.path.join(script_directory, "word2vec_vectorizer.pk1")
word2vec_vectorizer = joblib.load(word2vec_vectorizer_full_path)

# Create Tkinter Window
window = Tk()
window.title("Disaster Tweet Classification")
window.geometry('700x500')

label1 = Label(text='Enter Tweet to Classify', padx=5, pady=5, font=('Calibri', 20) )
label1.grid(row=1,column=0, columnspan=2, padx=5, pady=5)

entry = Text(height=10)
entry.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

result_label = Label(text='', padx=5, pady=5)
result_label.grid(row=3, column=0, columnspan=2)

probability_label = Label(text='', padx=5, pady=5)
probability_label.grid(row = 7, column = 0, columnspan=2)

#Naive Bayes and Logistic Regression Preprocessing
def preprocess_std(text) :

    # Lemmatizer
    lm = WordNetLemmatizer()

    # Checks if given word contains a special character
    def contains_special(word):
        for char in word:
            if char.isnumeric() or (not char.isalnum()):
                return True
        return False

    # Process sentences
    def process_sentences(sentence):
        # tokenize, lemmatize, and remove special characters
        processed = [lm.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)
                    # make sure no strings that contain only numeric characters
                    if not contains_special(word)]
        return ' '.join(processed)
    

    text_processed = process_sentences(text)
    text_vectorized = vectorizer.transform([text_processed])

    return text_vectorized

# Neural Networks Preprocessing
def preprocess_neural_networks(text):
    text = text.lower()
    tokens = word_tokenize(text)
    embeddings = []
    
    for token in tokens:
        if token in word2vec_vectorizer.wv:
            embeddings.append(word2vec_vectorizer.wv[token])
    
    max_sequence_length_feed_forward = 6000  # Change this to match the expected input size for the feed-forward model
    max_sequence_length_lstm = 30  # Change this to match the expected input size for the LSTM model
    max_sequence_length_bidirectional_lstm = 30  # Change this to match the expected input size for the bidirectional LSTM model
    embedding_size_lstm = 200
    embedding_size_bilstm = 200

    # Pad or truncate the embeddings for the feed-forward model
    if len(embeddings) >= max_sequence_length_feed_forward:
        embeddings = embeddings[:max_sequence_length_feed_forward]
    else:
        # Pad with zeros if the length is less than max_sequence_length_feed_forward
        embeddings.extend([np.zeros_like(embeddings[0])] * (max_sequence_length_feed_forward - len(embeddings)))

    # Reshape the array for the feed-forward model
    padded_embeddings_feed_forward = np.array(embeddings)

    reshaped_embeddings_feed_forward = padded_embeddings_feed_forward.flatten()[:max_sequence_length_feed_forward]
    reshaped_embeddings_feed_forward = reshaped_embeddings_feed_forward.reshape((1, max_sequence_length_feed_forward))


    # Pad or truncate the embeddings for LSTM model
    if len(embeddings) >= max_sequence_length_lstm:
        embeddings_lstm = embeddings[:max_sequence_length_lstm]
    else:
        # Pad with zeros if the length is less than max_sequence_length_lstm
        embeddings_lstm = embeddings + [np.zeros_like(embeddings[0])] * (max_sequence_length_lstm - len(embeddings))

    # Reshape the array for the LSTM model
    padded_embeddings_lstm = np.array(embeddings_lstm).reshape((1, max_sequence_length_lstm, embedding_size_lstm))

    # Pad or truncate the embeddings for Bidirectional LSTM model
    if len(embeddings) >= max_sequence_length_bidirectional_lstm:
        embeddings_bilstm = embeddings[:max_sequence_length_bidirectional_lstm]
    else:
        # Pad with zeros if the length is less than max_sequence_length_bidirectional_lstm
        embeddings_bilstm = embeddings + [np.zeros_like(embeddings[0])] * (max_sequence_length_bidirectional_lstm - len(embeddings))

    # Reshape the array for the Bidirectional LSTM model
    padded_embeddings_bidirectional_lstm = np.array(embeddings_bilstm).reshape((1, max_sequence_length_bidirectional_lstm, embedding_size_bilstm))

    return reshaped_embeddings_feed_forward, padded_embeddings_lstm, padded_embeddings_bidirectional_lstm

    
# Functions for Buttons
def get_classification(model, preprocess_function):
    user_input = entry.get('1.0', tk.END)

    if model == feed_forward_model :
        user_input_processed = preprocess_function(user_input)[0]
    elif model == lstm_model :
        user_input_processed = preprocess_function(user_input)[1]
    elif model == bidirectional_lstm_model :
        user_input_processed = preprocess_function(user_input)[2]
    else:
        user_input_processed = preprocess_function(user_input)

    classifier_result = model.predict(user_input_processed)[0]


    threshold = 0.5

    if classifier_result >= threshold :
        text_result = "Disaster"
    else :
        text_result = "Not a Disaster"


    result_label.config(text=f"Classification: {text_result}")

    if model in [feed_forward_model, lstm_model, bidirectional_lstm_model] :
        probability_label.config(text=f"Probability: {classifier_result}")
    else:
        probability_label.config(text='')



# Create buttons to invoke models
nbClassifyButton = Button(text='Naive Bayes', command=lambda: get_classification(naive_bayes_model, preprocess_std), width=12, bg = '#98F5FF', activebackground='#7AC5CD')
nbClassifyButton.grid(row=4, column=0,padx=5,pady=5)

lrClassifyButton = Button(text='Logistic Regression', command=lambda: get_classification(logistic_regression_model, preprocess_std), width=18, bg = '#7FFFD4', activebackground='#66CDAA')
lrClassifyButton.grid(row=4, column=1,padx=5,pady=5)

ffClassifyButton = Button(text='Feed-Forward Neural Network', command=lambda: get_classification(feed_forward_model, preprocess_neural_networks), width=25, bg = '#FF7256', activebackground='#CD5B45')
ffClassifyButton.grid(row=5, column=0,padx=5,pady=5)

lstmClassifyButton = Button(text='LSTM Neural Network', command=lambda: get_classification(lstm_model, preprocess_neural_networks), width=18, bg = '#AB82FF', activebackground='#8968CD')
lstmClassifyButton.grid(row=5, column=1,padx=5,pady=5)

bilstmClassifyButton = Button(text='Bidirectional LSTM Neural Network', command=lambda: get_classification(bidirectional_lstm_model, preprocess_neural_networks), width=28, bg = '#FFEC8B', activebackground='#CDBE70')
bilstmClassifyButton.grid(row=6, column=0,padx=5,pady=5)

window.mainloop()


