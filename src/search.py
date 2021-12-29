import pickle

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel
from utils import preprocess

print('Loading dataframe...')
df = pd.read_csv('data/Questions.csv', encoding="ISO-8859-1",
                 usecols=['Title', 'Body', 'CreationDate', 'Score'])

print('Loading inverted_index...')
inverted_index = pickle.load(open('data/inverted_index.sav', 'rb'))

print('Loading tfidf_title_vectorizer...')
tfidf_title_vectorizer = pickle.load(open('models/tfidf_title_vectorizer.sav', 'rb'))
print('Loading tfidf_body_vectorizer...')
tfidf_body_vectorizer  = pickle.load(open('models/tfidf_body_vectorizer.sav', 'rb'))


date_min = pd.to_datetime(df['CreationDate'].min()).timestamp()
date_max = pd.to_datetime(df['CreationDate'].max()).timestamp()

votes_min = df['Score'].min()
votes_max = df['Score'].max()

TITLE_WEIGHT = 0.7
BODY_WEIGHT = 0.3


def weighted_score(idx: int, query: str) -> float:
    doc = df.iloc[idx]

    date = pd.to_datetime(doc['CreationDate']).timestamp()
    DATE_WEIGHT = 1 + (date - date_min) / (date_max - date_min)

    votes = doc['Score']
    VOTE_WEIGHT = 1 + (votes - votes_min) / (votes_max - votes_min)

    title_tfidf = tfidf_title_vectorizer.transform([doc['Title']])
    query_title_tfidf = tfidf_title_vectorizer.transform([query])
    title_distance = euclidean_distances(query_title_tfidf, title_tfidf)

    body_tfidf = tfidf_body_vectorizer.transform([doc['Body']])
    query_body_tfidf = tfidf_body_vectorizer.transform([query])
    body_distance = euclidean_distances(query_body_tfidf, body_tfidf)
    return (TITLE_WEIGHT * title_distance + BODY_WEIGHT * body_distance) * DATE_WEIGHT * VOTE_WEIGHT

def traverse_inverted_index(query):
    """
    Returns intersection of documents
    in inverted index by query
    """
    query_tokens = query.split()
    pre_index = []
    for token in query_tokens:
        docs = inverted_index[token]
        pre_index.append(docs)
    docs = list(set.intersection(*map(set, pre_index)))
    return docs


def traverse_tfifd(col, tfidf_vectorizer, query, idxs, n_keep):
    """
    Calculates tfidf for col and
    returns top n_keep samples
    """
    feature = df[col][idxs]
    tfidf_query = tfidf_vectorizer.transform([query])
    tfidf_inverted_index_docs = tfidf_vectorizer.transform(feature)

    docs = linear_kernel(tfidf_query, tfidf_inverted_index_docs)
    docs = docs.flatten().argsort()[::-1][:n_keep]

    docs = np.array(idxs)[docs]
    return docs


def retrieve(query):
    if query == '':
        return []

    query = preprocess(query)

    inverted_index_docs = traverse_inverted_index(query)

    print(f'Found {len(inverted_index_docs)} using inverted index')

    if len(inverted_index_docs) == 0:
        return []

    docs = traverse_tfifd(col='Title', tfidf_vectorizer=tfidf_title_vectorizer,
                                   query=query, idxs=inverted_index_docs, n_keep=10)

    docs = [[doc, weighted_score(doc, query)[0, 0]] for doc in docs]
    docs.sort(key=lambda x: -x[1])

    print(f'Found {len(docs)} docs')

    index = []
    for doc, score in docs:
        CreationDate, Score, Title, Body = df.iloc[doc]
        index.append([CreationDate, Score, Title, f'{Body}', round(score, 3)])

    return index
