import nltk
import streamlit as st
import pickle

from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def tranform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    ## removing special char
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)  ## return string


# Load the saved model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

# Input text
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess the input text
    transformed_sms = tranform_text(input_sms)

    # 2. Vectorize the preprocessed text using the loaded TfidfVectorizer
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict using the loaded model
    result = model.predict(vector_input)[0]

    # 4. Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
