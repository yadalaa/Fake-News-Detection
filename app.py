import streamlit as st
import pandas as pd
import re
import numpy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import string
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Target variable for fake news
fake['temp'] = 0

# Target variable for true news
true['temp'] = 1

# Concatenating and dropping for fake news
fake['news'] = fake['title'] + fake['text']
fake = fake.drop(['title', 'text'], axis=1)

# Concatenating and dropping for true news
true['news'] = true['title'] + true['text']
true = true.drop(['title', 'text'], axis=1)

# Rearranging the columns
fake = fake[['subject', 'date', 'news', 'temp']]
true = true[['subject', 'date', 'news', 'temp']]

frames = [fake, true]
news_dataset = pd.concat(frames)

clean_news = news_dataset.copy()

def review_cleaning(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

clean_news['news'] = clean_news['news'].apply(lambda x: review_cleaning(x))

stop = stopwords.words('english')
clean_news['news'] = clean_news['news'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
X = tfidf_vectorizer.fit_transform(clean_news['news'])
y = clean_news['temp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Streamlit app
st.title('Fake News Detector')
input_text = st.text_area('Enter news article')

def predict_fake_news(input_text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Perform stemming on the input text
    input_text = re.sub('[^a-zA-Z]', ' ', input_text)
    input_text = input_text.lower()
    input_text = input_text.split()
    input_text = [ps.stem(word) for word in input_text if word not in stop_words]
    input_text = ' '.join(input_text)

    # Vectorize the input text
    input_data = tfidf_vectorizer.transform([input_text])

    # Make prediction and get confidence levels
    prediction = knn.predict(input_data)
    confidence_levels = knn.predict_proba(input_data)[0]  # Probabilities for both classes

    return prediction[0], confidence_levels[0], confidence_levels[1]

if input_text:
    prediction, confidence_fake, confidence_true = predict_fake_news(input_text)
    st.subheader("Prediction:")
    if prediction == 1:
        st.write('The news is real.')
        st.subheader("Confidence Level for True News:")
        st.write(f"{confidence_true * 100:.2f}%")
        st.subheader("Confidence Level for Fake News:")
        st.write(f"{confidence_fake * 100:.2f}%")
    elif prediction == 0:
        st.write('The news is fake.')
        st.subheader("Confidence Level for True News:")
        st.write(f"{confidence_true * 100:.2f}%")
        st.subheader("Confidence Level for Fake News:")
        st.write(f"{confidence_fake * 100:.2f}%")
