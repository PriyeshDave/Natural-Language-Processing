import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import *
from PIL import Image


model = joblib.load('Model/lemmatize_tfidf_naiveBayes.pkl')
tfidf_vectorizer = joblib.load('Model/tfidf_vectorizer.pkl')

st.set_page_config(page_title="Sentiment Analysis Appilcation",
                   page_icon="😄😠⌛", layout="wide")


#creating option list for dropdown menu


st.markdown("<h1 style='text-align: center;'>Sentiment Analysis Application 😄😠⌛</h1>", unsafe_allow_html=True)
image = Image.open('Sentiment-Analysis.jpg')
st.image(image)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter your text below.")
        text = st.text_area(label='Enter text here...')
        submit = st.form_submit_button("Predict")


    if submit:
       text = lemamtizeSentence(text)
       print('Lemmatized Text', text)
       pred = get_prediction(text, model, tfidf_vectorizer)

       if pred == 0:
              st.write('Ughhh!!! It seems like a negative text.😒')
              st.write('Always say positive words.🙂')
       else:
              st.write('Positive Sentiment. Stay Positive always!🙂')
       #st.write(f"The predicted severity is:  {pred}")

if __name__ == '__main__':
    main()