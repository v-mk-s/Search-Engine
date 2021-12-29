import nltk
import pickle

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from utils import preprocess_title, preprocess_body

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

df = pd.read_csv('data/Questions.csv', encoding="ISO-8859-1",
                 usecols=['Title', 'Body', 'CreationDate', 'Score'])


print('Preprocessing title')
df['Title preprocessed'] = df['Title'].apply(preprocess_title)
print('Preprocessing body')
df['Body preprocessed'] = df['Body'].apply(preprocess_body)
df.to_csv('data/Preprocessed Questions.csv', index=False)


title = df['Title']
tfidf_title_vectorizer = TfidfVectorizer(stop_words=stopwords)
print('Fitting tfidf_title_vectorizer')
tfidf_title_vectorizer.fit(title)
pickle.dump(tfidf_title_vectorizer,
            open('models/tfidf_title_vectorizer.sav', 'wb'))

body = df['Body']
tfidf_body_vectorizer = TfidfVectorizer(stop_words=stopwords)
print('Fitting tfidf_body_vectorizer')
tfidf_body_vectorizer.fit(body)
pickle.dump(tfidf_body_vectorizer,
            open('models/tfidf_body_vectorizer.sav', 'wb'))


def build_inverted_index():
    inverted_index = defaultdict(list)
    rows, cols = df.shape
    for idx in range(1, rows):
        sample = df.iloc[idx]
        union = sample['Title']
        for word in set(union.split()):
            inverted_index[word].append(idx)

    filename = 'inverted_index.sav'
    pickle.dump(inverted_index, open(filename, 'wb'))
    inverted_index = pickle.load(open(filename, 'rb'))
    return inverted_index


print('Building inverted index')
inverted_index = build_inverted_index()

pickle.dump(inverted_index, open('data/inverted_index.sav', 'wb'))
