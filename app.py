import streamlit as st 
import pickle
import pandas as pd
import requests 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template

pss = PorterStemmer()

def transform(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_sentence = [w for w in tokens if not w in stopwords.words('english')]
    filtered_sentence = [pss.stem(word) for word in filtered_sentence]
    lemmatizer = WordNetLemmatizer()
    filtered_sentence = [lemmatizer.lemmatize(word) for word in filtered_sentence]
    return filtered_sentence

tfidf =  pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('mnb_clf.pkl','rb'))

st.title("Email Classifier - Spam or not spam")

input = st.text_area("Enter the mail")

if st.button('predict'):
    transformed_text = transform(input)
    vectorize = tfidf.transform(transformed_text)
    prediction = model.predict(vectorize)[0]
    if prediction == 1:
      st.header("Spam")
    else:
      st.header("Not Spam")
    
