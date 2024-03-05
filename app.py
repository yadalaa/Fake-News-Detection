import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')  # You might also need to download the punkt tokenizer
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to preprocess text
def preprocess_text(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    return ' '.join(text)

# Load the model and CountVectorizer
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake["temp"] = 1
true["temp"] = 0
df = pd.concat([fake, true]).reset_index(drop=True)
X = df['title'].apply(preprocess_text)
y = df['temp'].values
countv = CountVectorizer(max_features=5000)
X = countv.fit_transform(X).toarray()
classifier = LogisticRegression(random_state=0)
classifier.fit(X, y)

# Streamlit app
st.title("Fake News Classifier")

# Text input for prediction
user_input = st.text_area("Enter the news title:")

if user_input:
    # Preprocess user input
    processed_input = preprocess_text(user_input)

    # Transform user input using CountVectorizer
    input_arr = countv.transform([processed_input]).toarray()

    # Make prediction
    prediction = classifier.predict(input_arr)
    confidence = classifier.predict_proba(input_arr)[:, 1]

    # Display prediction and confidence
    st.subheader("Prediction:")
    if prediction[0] == 1:
        st.write("News is Fake")
    else:
        st.write("News is Real")

    st.subheader("Confidence Level:")
    st.write(f"{confidence[0] * 100:.2f}%")


