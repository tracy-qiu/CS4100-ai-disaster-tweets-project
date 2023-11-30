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

from keras.preprocessing.text import Tokenizer

import numpy as np

import os

# Get directory and full paths
script_directory = os.path.dirname(os.path.abspath(__file__))

### LOAD MODELS ###
# Original Dataset 
naive_bayes_full_path = os.path.join(script_directory, "naive_bayes_model_Original Dataset.pk1")
naive_bayes_model = joblib.load(naive_bayes_full_path)
logistic_regression_full_path = os.path.join(script_directory, "logistic_regression_model_Original Dataset.pk1")
logistic_regression_model = joblib.load(logistic_regression_full_path)
feed_forward_full_path = os.path.join(script_directory, "feed_forward_model_Original Dataset.pk1")
feed_forward_model = joblib.load(feed_forward_full_path)
lstm_full_path = os.path.join(script_directory, "lstm_model_Original Dataset.pk1")
lstm_model = joblib.load(lstm_full_path)
bidirectional_lstm_full_path = os.path.join(script_directory, "bidirectional_lstm_model_Original Dataset.pk1")
bidirectional_lstm_model = joblib.load(bidirectional_lstm_full_path)

# Oversampled Dataset
naive_bayes_oversampled_full_path = os.path.join(script_directory, "naive_bayes_model_Oversampling Dataset.pk1")
naive_bayes_oversampled_model = joblib.load(naive_bayes_oversampled_full_path)
logistic_regression_oversampled_full_path = os.path.join(script_directory, "logistic_regression_model_Oversampling Dataset.pk1")
logistic_regression_oversampled_model = joblib.load(logistic_regression_oversampled_full_path)
feed_forward_oversampled_full_path = os.path.join(script_directory, "feed_forward_model_Oversampling Dataset.pk1")
feed_forward_oversampled_model = joblib.load(feed_forward_oversampled_full_path)
lstm__oversampled_full_path = os.path.join(script_directory, "lstm_model_Oversampling Dataset.pk1")
lstm_oversampled_model = joblib.load(lstm__oversampled_full_path)
bidirectional_lstm_oversampled_full_path = os.path.join(script_directory, "bidirectional_lstm_model_Oversampling Dataset.pk1")
bidirectional_lstm_oversampled_model = joblib.load(bidirectional_lstm_oversampled_full_path)

# Undersampled Dataset
naive_bayes_undersampled_full_path = os.path.join(script_directory, "naive_bayes_model_Undersampling Dataset.pk1")
naive_bayes_undersampled_model = joblib.load(naive_bayes_undersampled_full_path)
logistic_regression_undersampled_full_path = os.path.join(script_directory, "logistic_regression_model_Undersampling Dataset.pk1")
logistic_regression_undersampled_model = joblib.load(logistic_regression_undersampled_full_path)
feed_forward_undersampled_full_path = os.path.join(script_directory, "feed_forward_model_Undersampling Dataset.pk1")
feed_forward_undersampled_model = joblib.load(feed_forward_undersampled_full_path)
lstm__undersampled_full_path = os.path.join(script_directory, "lstm_model_Undersampling Dataset.pk1")
lstm_undersampled_model = joblib.load(lstm__undersampled_full_path)
bidirectional_lstm_undersampled_full_path = os.path.join(script_directory, "bidirectional_lstm_model_Undersampling Dataset.pk1")
bidirectional_lstm_undersampled_model = joblib.load(bidirectional_lstm_undersampled_full_path)

# Load vectorizers - SAME VECTORIZER USED TO TRAIN MODELS MUST BE USED
vectorizer_full_path = os.path.join(script_directory, "vectorizer_original.pk1")
vectorizer = joblib.load(vectorizer_full_path)

vectorizer_oversampled_full_path = os.path.join(script_directory, "vectorizer_oversampling.pk1")
vectorizer_oversampled = joblib.load(vectorizer_oversampled_full_path)

vectorizer_undersampled_full_path = os.path.join(script_directory, "vectorizer_undersampling.pk1")
vectorizer_undersampled = joblib.load(vectorizer_undersampled_full_path)

word2vec_vectorizer_full_path = os.path.join(script_directory, "word2vec_vectorizer_original.pk1")
word2vec_vectorizer = joblib.load(word2vec_vectorizer_full_path)

word2vec_vectorizer_oversampled_full_path = os.path.join(script_directory, "word2vec_vectorizer_oversampling.pk1")
word2vec_vectorizer_oversampled = joblib.load(word2vec_vectorizer_oversampled_full_path)

word2vec_vectorizer_undersampled_full_path = os.path.join(script_directory, "word2vec_vectorizer_undersampling.pk1")
word2vec_vectorizer_undersampled = joblib.load(word2vec_vectorizer_undersampled_full_path)

# Create Tkinter Window
window = Tk()
window.title("Disaster Tweet Classification")
window.geometry('700x500')

tabControl = ttk.Notebook(window)
tabControl.pack()

tabOriginal = Frame(tabControl)
tabOversampling = Frame(tabControl)
tabUndersampling = Frame(tabControl)

tabOriginal.pack()
tabOversampling.pack()
tabUndersampling.pack()

tabControl.add(tabOriginal, text='Original Data')
tabControl.add(tabOversampling, text='Oversampling')
tabControl.add(tabUndersampling, text='Undersampling')

# Original Dataset Tab
labelOriginal = Label(tabOriginal, text='Enter Tweet to Classify', padx=5, pady=5, font=('Calibri', 20) )
labelOriginal.grid(row=1,column=0, columnspan=2, padx=5, pady=5)

entryOriginal = Text(tabOriginal, height=10)
entryOriginal.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

resultLabelOriginal = Label(tabOriginal, text='', padx=5, pady=5)
resultLabelOriginal.grid(row=3, column=0, columnspan=2)

probabilityLabelOriginal = Label(tabOriginal, text='', padx=5, pady=5)
probabilityLabelOriginal.grid(row = 7, column = 0, columnspan=2)

# Oversampled Dataset Tab
labelOversampling = Label(tabOversampling, text='Enter Tweet to Classify', padx=5, pady=5, font=('Calibri', 20) )
labelOversampling.grid(row=1,column=0, columnspan=2, padx=5, pady=5)

entryOversampling = Text(tabOversampling, height=10)
entryOversampling.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

resultLabelOversampling = Label(tabOversampling, text='', padx=5, pady=5)
resultLabelOversampling.grid(row=3, column=0, columnspan=2)

probabilityLabelOversampling = Label(tabOversampling, text='', padx=5, pady=5)
probabilityLabelOversampling.grid(row = 7, column = 0, columnspan=2)

# Undersampled Dataset Tab
labelUndersampling = Label(tabUndersampling, text='Enter Tweet to Classify', padx=5, pady=5, font=('Calibri', 20) )
labelUndersampling.grid(row=1,column=0, columnspan=2, padx=5, pady=5)

entryUndersampling = Text(tabUndersampling, height=10)
entryUndersampling.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

resultLabelUndersampling = Label(tabUndersampling, text='', padx=5, pady=5)
resultLabelUndersampling.grid(row=3, column=0, columnspan=2)

probabilityLabelUndersampling = Label(tabUndersampling, text='', padx=5, pady=5)
probabilityLabelUndersampling.grid(row = 7, column = 0, columnspan=2)

#Naive Bayes and Logistic Regression Preprocessing
def preprocess_std(text, df_type) :

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

    if (df_type == 'original') :
        text_vectorized = vectorizer.transform([text_processed])
    elif (df_type == 'oversampling') :
        text_vectorized = vectorizer_oversampled.transform([text_processed])
    elif (df_type == 'undersampling') :
        text_vectorized = vectorizer_undersampled.transform([text_processed])

    return text_vectorized

# Neural Networks Preprocessing
def preprocess_neural_networks(text, df_type):
    text = text.lower()

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

    tokens = word_tokenize(text_processed)

    embeddings = []

    for token in tokens:
        if (df_type == 'original') :
            if token in word2vec_vectorizer.wv:
                embeddings.append(word2vec_vectorizer.wv[token])
        elif (df_type == 'oversampling') :
            if token in word2vec_vectorizer_oversampled.wv:
                embeddings.append(word2vec_vectorizer_oversampled.wv[token])
        elif (df_type == 'undersampling') :
            if token in word2vec_vectorizer_undersampled.wv:
                embeddings.append(word2vec_vectorizer_undersampled.wv[token])

    if (df_type == 'undersampling') :
        max_sequence_length_feed_forward = 5800  # Change this to match the expected input size for the feed-forward model
        max_sequence_length_lstm = 29  # Change this to match the expected input size for the LSTM model
        max_sequence_length_bidirectional_lstm = 29  # Change this to match the expected input size for the bidirectional LSTM model
    else :
        max_sequence_length_feed_forward = 6000  # Change this to match the expected input size for the feed-forward model
        max_sequence_length_lstm = 30  # Change this to match the expected input size for the LSTM model
        max_sequence_length_bidirectional_lstm = 30  # Change this to match the expected input size for the bidirectional LSTM model

   
    embedding_size_lstm = 200
    embedding_size_bilstm = 200



     # Pad or truncate the embeddings for the feed-forward model
    if len(embeddings) >= max_sequence_length_feed_forward:
        embeddings_feed_forward = embeddings[:max_sequence_length_feed_forward]
    else:
        # Pad with zeros if the length is less than max_sequence_length_feed_forward
        embeddings_feed_forward = embeddings + ([np.zeros_like(embeddings[0])] * (max_sequence_length_feed_forward - len(embeddings)))


    # Reshape the array for the feed-forward model
    padded_embeddings_feed_forward = np.array(embeddings_feed_forward)

    reshaped_embeddings_feed_forward = padded_embeddings_feed_forward.flatten()[:max_sequence_length_feed_forward]
    reshaped_embeddings_feed_forward = reshaped_embeddings_feed_forward.reshape((1, max_sequence_length_feed_forward))


    # Pad or truncate the embeddings for LSTM model
    if len(embeddings) >= max_sequence_length_lstm:
        embeddings_lstm = embeddings[:max_sequence_length_lstm]
    else:
        # Pad with zeros if the length is less than max_sequence_length_lstm
        embeddings_lstm = embeddings + ([np.zeros_like(embeddings[0])] * (max_sequence_length_lstm - len(embeddings)))

    # Reshape the array for the LSTM model
    padded_embeddings_lstm = np.array(embeddings_lstm).reshape((1, max_sequence_length_lstm, embedding_size_lstm))

    # Pad or truncate the embeddings for Bidirectional LSTM model
    if len(embeddings) >= max_sequence_length_bidirectional_lstm:
        embeddings_bilstm = embeddings[:max_sequence_length_bidirectional_lstm]
    else:
        # Pad with zeros if the length is less than max_sequence_length_bidirectional_lstm
        embeddings_bilstm = embeddings + ([np.zeros_like(embeddings[0])]  * (max_sequence_length_bidirectional_lstm - len(embeddings)))

    # Reshape the array for the Bidirectional LSTM model
    padded_embeddings_bidirectional_lstm = np.array(embeddings_bilstm).reshape((1, max_sequence_length_bidirectional_lstm, embedding_size_bilstm))

    
    return reshaped_embeddings_feed_forward, padded_embeddings_lstm, padded_embeddings_bidirectional_lstm


    
# Functions for Buttons
def get_classification(model, preprocess_function, entry, df_type):
    user_input = entry.get('1.0', tk.END)


    if model in [feed_forward_model, feed_forward_oversampled_model, feed_forward_undersampled_model]:
        user_input_processed = preprocess_function(user_input, df_type)[0]
    elif model in [lstm_model, lstm_oversampled_model, lstm_undersampled_model] :
        user_input_processed = preprocess_function(user_input, df_type)[1]
    elif model in [bidirectional_lstm_model, bidirectional_lstm_oversampled_model, bidirectional_lstm_undersampled_model] :
        user_input_processed = preprocess_function(user_input, df_type)[2]
    else:
        user_input_processed = preprocess_function(user_input, df_type)

    classifier_result = model.predict(user_input_processed)[0]
    print(classifier_result)


    threshold = 0.5

    if classifier_result >= threshold :
        text_result = "Disaster"
    else :
        text_result = "Not a Disaster"

    if (df_type == 'original') :
        resultLabelOriginal.config(text=f"Classification: {text_result}")
    elif (df_type == 'oversampling') :
        resultLabelOversampling.config(text=f"Classification: {text_result}")
    elif (df_type == 'undersampling') :
        resultLabelUndersampling.config(text=f"Classification: {text_result}")


    if model in [feed_forward_model, lstm_model, bidirectional_lstm_model] :
        probabilityLabelOriginal.config(text=f"Probability: {classifier_result}")
    elif model in [feed_forward_oversampled_model, lstm_oversampled_model, bidirectional_lstm_oversampled_model] :
        probabilityLabelOversampling.config(text=f"Probability: {classifier_result}")
    elif model in [feed_forward_undersampled_model, lstm_undersampled_model, bidirectional_lstm_undersampled_model] :
        probabilityLabelUndersampling.config(text=f"Probability: {classifier_result}")
    else:
        probabilityLabelOriginal.config(text='')
        probabilityLabelOversampling.config(text='')
        probabilityLabelUndersampling.config(text='')




# Create buttons to invoke models: Original Dataset
nbClassifyButtonOriginal = Button(tabOriginal, text='Naive Bayes', command=lambda: get_classification(naive_bayes_model, preprocess_std, entryOriginal, 'original'), width=12, bg = '#98F5FF', activebackground='#7AC5CD')
nbClassifyButtonOriginal.grid(row=4, column=0,padx=5,pady=5)

lrClassifyButtonOriginal = Button(tabOriginal, text='Logistic Regression', command=lambda: get_classification(logistic_regression_model, preprocess_std, entryOriginal, 'original'), width=18, bg = '#7FFFD4', activebackground='#66CDAA')
lrClassifyButtonOriginal.grid(row=4, column=1,padx=5,pady=5)

ffClassifyButtonOriginal = Button(tabOriginal, text='Feed-Forward Neural Network', command=lambda: get_classification(feed_forward_model, preprocess_neural_networks, entryOriginal, 'original'), width=25, bg = '#FF7256', activebackground='#CD5B45')
ffClassifyButtonOriginal.grid(row=5, column=0,padx=5,pady=5)

lstmClassifyButtonOriginal = Button(tabOriginal, text='LSTM Neural Network', command=lambda: get_classification(lstm_model, preprocess_neural_networks, entryOriginal, 'original'), width=18, bg = '#AB82FF', activebackground='#8968CD')
lstmClassifyButtonOriginal.grid(row=5, column=1,padx=5,pady=5)

bilstmClassifyButtonOriginal = Button(tabOriginal, text='Bidirectional LSTM Neural Network', command=lambda: get_classification(bidirectional_lstm_model, preprocess_neural_networks, entryOriginal, 'original'), width=28, bg = '#FFEC8B', activebackground='#CDBE70')
bilstmClassifyButtonOriginal.grid(row=6, column=0,padx=5,pady=5)

# Create buttons to invoke models: Oversampling
nbClassifyButtonOversampling = Button(tabOversampling, text='Naive Bayes', command=lambda: get_classification(naive_bayes_oversampled_model, preprocess_std, entryOversampling, 'oversampling'), width=12, bg = '#98F5FF', activebackground='#7AC5CD')
nbClassifyButtonOversampling.grid(row=4, column=0,padx=5,pady=5)

lrClassifyButtonOversampling = Button(tabOversampling, text='Logistic Regression', command=lambda: get_classification(logistic_regression_oversampled_model, preprocess_std, entryOversampling, 'oversampling'), width=18, bg = '#7FFFD4', activebackground='#66CDAA')
lrClassifyButtonOversampling.grid(row=4, column=1,padx=5,pady=5)

ffClassifyButtonOversampling = Button(tabOversampling, text='Feed-Forward Neural Network', command=lambda: get_classification(feed_forward_oversampled_model, preprocess_neural_networks, entryOversampling, 'oversampling'), width=25, bg = '#FF7256', activebackground='#CD5B45')
ffClassifyButtonOversampling.grid(row=5, column=0,padx=5,pady=5)

lstmClassifyButtonOversampling = Button(tabOversampling, text='LSTM Neural Network', command=lambda: get_classification(lstm_oversampled_model, preprocess_neural_networks, entryOversampling, 'oversampling'), width=18, bg = '#AB82FF', activebackground='#8968CD')
lstmClassifyButtonOversampling.grid(row=5, column=1,padx=5,pady=5)

bilstmClassifyButtonOversampling = Button(tabOversampling, text='Bidirectional LSTM Neural Network', command=lambda: get_classification(bidirectional_lstm_oversampled_model, preprocess_neural_networks, entryOversampling, 'oversampling'), width=28, bg = '#FFEC8B', activebackground='#CDBE70')
bilstmClassifyButtonOversampling.grid(row=6, column=0,padx=5,pady=5)

# Create buttons to invoke models: Undersampling
nbClassifyButtonUndersampling = Button(tabUndersampling, text='Naive Bayes', command=lambda: get_classification(naive_bayes_undersampled_model, preprocess_std, entryUndersampling, 'undersampling'), width=12, bg = '#98F5FF', activebackground='#7AC5CD')
nbClassifyButtonUndersampling.grid(row=4, column=0,padx=5,pady=5)

lrClassifyButtonUndersampling = Button(tabUndersampling, text='Logistic Regression', command=lambda: get_classification(logistic_regression_undersampled_model, preprocess_std, entryUndersampling, 'undersampling'), width=18, bg = '#7FFFD4', activebackground='#66CDAA')
lrClassifyButtonUndersampling.grid(row=4, column=1,padx=5,pady=5)

ffClassifyButtonUndersampling = Button(tabUndersampling, text='Feed-Forward Neural Network', command=lambda: get_classification(feed_forward_undersampled_model, preprocess_neural_networks, entryUndersampling, 'undersampling'), width=25, bg = '#FF7256', activebackground='#CD5B45')
ffClassifyButtonUndersampling.grid(row=5, column=0,padx=5,pady=5)

lstmClassifyButtonUndersampling = Button(tabUndersampling, text='LSTM Neural Network', command=lambda: get_classification(lstm_undersampled_model, preprocess_neural_networks, entryUndersampling, 'undersampling'), width=18, bg = '#AB82FF', activebackground='#8968CD')
lstmClassifyButtonUndersampling.grid(row=5, column=1,padx=5,pady=5)

bilstmClassifyButtonUndersampling = Button(tabUndersampling, text='Bidirectional LSTM Neural Network', command=lambda: get_classification(bidirectional_lstm_undersampled_model, preprocess_neural_networks, entryUndersampling, 'undersampling'), width=28, bg = '#FFEC8B', activebackground='#CDBE70')
bilstmClassifyButtonUndersampling.grid(row=6, column=0,padx=5,pady=5)

window.mainloop()


