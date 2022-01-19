import streamlit as st
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import joblib

st.title('Cowid Tweet Sentiment Classifier')


spam_model = joblib.load('naive_model.joblib')
vectorizer = joblib.load('CountVectorizer.joblib')
inp_text = st.text_area('Enter the tweet  to determine whether it is Positive,Negative,Neutral',height=200)

vectorised_text = vectorizer.transform([inp_text])
pred = ''


def spam_predict(inp_text):
    prediction = spam_model.predict(inp_text)
    if prediction == 0:
        pred = 'Neutral'
    elif prediction == 1:
        pred = 'Positive'
    else:
        pred = 'Negative'
    return pred
if st.button('Submit'):
    st.write('The tweet you entered is:',spam_predict(vectorised_text))
