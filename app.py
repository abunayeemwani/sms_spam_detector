import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # remove non alpha numeric characters/words
    alpha_num = []
    for i in text:
        if i.isalnum():
            alpha_num.append(i)

    # remove stop words like [is, the, you, on...]
    clean_words = []
    for i in alpha_num:
        if i not in stopwords.words('english') and i not in string.punctuation:
            clean_words.append(i)

    result = []
    for i in clean_words:
        result.append(ps.stem(i))

    return " ".join(result)

st.title('Email/SMS Spam Detector')
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. preprocess
    transformed_text = transform_text(input_sms)

    # 2. vectorize
    vector_input = tfidf.transform([transformed_text])

    # 3. predict
    result = model.predict(vector_input)[0]

    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
