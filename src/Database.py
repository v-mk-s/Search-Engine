import pickle

from utils import remove_stop_words, lemmatize
from Document import Document


class Database:
    """
    Database class

    used to store all
    documents of type Document

    Attributes:
        body_inverted_index: pretrained body_inverted_index
        title_inverted_index: pretrained title_inverted_index
        body_tfidf_vectorizer: pretrained body_tfidf_vectorizer
        title_tfidf_vectorizer: pretrained title_tfidf_vectorizer
        index: list that contains all of the documents
        table_name: name of the preprocessed .sav file from preprocess.ipynb
        tfidf_title_vectorizer_name: name of pretrained tfidf_title_vectorizer_name
        tfidf_body_vectorizer_name: name of pretrained tfidf_body_vectorizer_name
        title_inverted_index_name: name of pretrained title_inverted_index_name
        body_inverted_index_name: name of ptetrained body_inverted_index_name
    """

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
        """
        Retrieves documents from preprocessed file
        and saves them in self.index
        """

        table = pickle.load(open(self.table_name, 'rb'))
        rows, cols = table.shape

        for idx in range(rows):
            id, title, body, date_score, votes_score, title_top_10_tfidf, body_top_30_tfidf = table.iloc[idx]
            self.index[id] = Document(id, title, body, date_score, votes_score,
                                      title_top_10_tfidf, body_top_30_tfidf)

    def get(self, id: int) -> Document:
        """
        Returns single doc by id
        """

        return self.index[id]

    def find_in_title_inverted_index(self, query: str) -> list:
        """
        Retrieves documents by intersecting keywords
        from query and words in title inverted index
        """

        query_tokens = query.split()
        pre_index = []
        for token in query_tokens:
            docs = self.title_inverted_index[token]
            pre_index.append(docs)
        if len(pre_index) == 0:
            return []
        docs = list(set.intersection(*map(set, pre_index)))
        return docs

    def find_in_body_inverted_index(self, query: str) -> list:
        """
        Retrieves documents by intersecting keywords
        from query and words in body inverted index
        """

        query_tokens = query.split()
        pre_index = []
        for token in query_tokens:
            docs = self.body_inverted_index[token]
            pre_index.append(docs)
        if len(pre_index) == 0:
            return []
        docs = list(set.intersection(*map(set, pre_index)))
        return docs

    def find_n_best_docs(self, query, n=5):
        """
        Retrieves best n documents by firstly applying
        find_in_title_inverted_index and then calculating
        tf-idf vector distance between query tf-idf vector
        and all documents acquired in inverted index, then
        sort them with respect to score
        """

        if query == '':
            return []

        query = remove_stop_words(query)
        query = lemmatize(query)

        docs_body_idxs = self.find_in_body_inverted_index(query)
        docs_title_idxs = self.find_in_title_inverted_index(query)

        docs_idxs = set(docs_body_idxs) | set(docs_title_idxs)

        if len(docs_idxs) == 0:
            return []

        index = [self.get(id) for id in docs_idxs]

        tfidf_query = self.title_tfidf_vectorizer.transform([query])

        index.sort(key=lambda x: -x.calculate_score(tfidf_query))

        return index[:n]
