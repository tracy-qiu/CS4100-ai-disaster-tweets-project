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

import os

# Get directory and full paths
script_directory = os.path.dirname(os.path.abspath(__file__))

# Load models
naive_bayes_full_path = os.path.join(script_directory, "naive_bayes_model.pk1")
naive_bayes_model = joblib.load(naive_bayes_full_path)
logistic_regression_full_path = os.path.join(script_directory, "logistic_regression_model.pk1")
logistic_regression_model = joblib.load(logistic_regression_full_path)

# Load vectorizer - SAME VECTORIZER USED TO TRAIN MODELS MUST BE USED
vectorizer_full_path = os.path.join(script_directory, "vectorizer.pk1")
vectorizer = joblib.load(vectorizer_full_path)

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

# Functions for Buttons
def get_classification(model):
    user_input = entry.get('1.0', tk.END)

    
    ### The processing of the inputed text must be exactly the same as the training datasets of the models used here ###
    ### Begin Processing ###

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
    

    user_input_processed = process_sentences(user_input)
    user_input_vectorized = vectorizer.transform([user_input_processed])

    ### End Processing ###

    classifier_result = model.predict(user_input_vectorized)[0]

    if classifier_result == 0 :
        text_result = "Not a Disaster"
    if classifier_result == 1:
        text_result = "Disaster"


    result_label.config(text=f"Classification: {text_result}")

# Create buttons to invoke models
nbClassifyButton = Button(text='Naive Bayes', command=lambda: get_classification(naive_bayes_model), width=12, bg = '#98F5FF', activebackground='#7AC5CD')
nbClassifyButton.grid(row=4, column=0,padx=5,pady=5)

lrClassifyButton = Button(text='Logistic Regression', command=lambda: get_classification(logistic_regression_model), width=18, bg = '#7FFFD4', activebackground='#66CDAA')
lrClassifyButton.grid(row=4, column=1,padx=5,pady=5)

window.mainloop()


