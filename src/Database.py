import pickle

from utils import remove_stop_words, lemmatize
from Document import Document


class Database:
    def __init__(self):
        self.body_inverted_index = None
        self.title_inverted_index = None

        self.body_tfidf_vectorizer = None
        self.title_tfidf_vectorizer = None

        self.index = dict()

        self.table_name = 'data/Preprocessed Questions.sav'
        self.tfidf_title_vectorizer_name = 'data/tfidf_title_vectorizer.sav'
        self.tfidf_body_vectorizer_name = 'data/tfidf_body_vectorizer.sav'

        self.title_inverted_index_name = 'data/title_inverted_index.sav'
        self.body_inverted_index_name = 'data/body_inverted_index.sav'

    def load_title_inverted_index(self):
        self.title_inverted_index = pickle.load(open(self.title_inverted_index_name, 'rb'))

    def load_body_inverted_index(self):
        self.body_inverted_index = pickle.load(open(self.body_inverted_index_name, 'rb'))

    def load_tfidf_title_vectorizer(self):
        self.title_tfidf_vectorizer = pickle.load(open(self.tfidf_title_vectorizer_name, 'rb'))

    def load_tfidf_body_vectorizer(self):
        self.body_tfidf_vectorizer = pickle.load(open(self.tfidf_body_vectorizer_name, 'rb'))

    def load_database(self):
        table = pickle.load(open(self.table_name, 'rb'))
        rows, cols = table.shape

        for idx in range(rows):
            id, title, body, date_score, votes_score, title_top_10_tfidf, body_top_30_tfidf = table.iloc[idx]
            self.index[id] = Document(id, title, body, date_score, votes_score,
                                      title_top_10_tfidf, body_top_30_tfidf)

    def get(self, id: int):
        return self.index[id]

    def find_in_title_inverted_index(self, query):
        query_tokens = query.split()
        pre_index = []
        for token in query_tokens:
            docs = self.title_inverted_index[token]
            pre_index.append(docs)

        if len(pre_index) == 0:
            return []

        docs = list(set.intersection(*map(set, pre_index)))
        return docs

    def find_n_best_docs(self, query, n=5):
        if query == '':
            return []

        query = remove_stop_words(query)
        query = lemmatize(query)

        docs_idxs = self.find_in_title_inverted_index(query)

        print(f'Found {len(docs_idxs)} using inverted index')

        if len(docs_idxs) == 0:
            return []

        index = [self.get(id) for id in docs_idxs]

        tfidf_query = self.title_tfidf_vectorizer.transform([query])

        index.sort(key=lambda x: -x.calculate_score(tfidf_query))

        return index[:n]
