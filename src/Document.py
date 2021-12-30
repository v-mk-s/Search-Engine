from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Document:
    """
    Document class

    Parameters
    ----------
    id: int
        Unique index of document

    title : title
        Date when question was posted

    body : body
        Number of votes for question

    date_score : date_score
        Question title

    votes_score : votes_score
        Question title

    title_top_10_tfidf: matrix
        Title preprocessed top 10 keywords TF-IDF vector used
        for calculating distance between TF-IDF query and TF-IDF of document

    body_top_30_tfidf: matrix
        Body preprocessed top 30 keywords TF-IDF vector used
        for calculating distance between TF-IDF query and TF-IDF of document
    """
    def __init__(self, id, title, body, date_score, votes_score, title_top_10_tfidf, body_top_30_tfidf):
        self.id = id
        self.title = title
        self.body = body
        self.date_score = date_score
        self.votes_score = votes_score
        self.title_top_10_tfidf = title_top_10_tfidf
        self.body_top_30_tfidf = body_top_30_tfidf
        self.score = None

    def calculate_tfidf_title_distance(self, tfidf_query):
        cosine_similarities = cosine_similarity(self.title_top_10_tfidf.reshape(1, -1), tfidf_query.reshape(1, -1))
        return cosine_similarities[0][0]

    def calculate_tfidf_body_distance(self, tfidf_query):
        cosine_similarities = cosine_similarity(self.title_top_10_tfidf.reshape(1, -1), tfidf_query.reshape(1, -1))
        return cosine_similarities[0][0]

    def calculate_score(self, tfidf_query) -> float:
        """
        Perform a scoring based on cosine similarity of
        tf-idf vectors between document and seach query.

        Parameters
        ----------
        tfidf_query : matrix
            The search query

        calculate_tfidf_title_distance : TfidfVectorizer
            TfidfVectorizer

        calculate_tfidf_body_distance : TfidfVectorizer
            TfidfVectorizer
        Returns
        -------
        score : float
             Score is between 0 and 10
        """
        TITLE_WEIGHT = 0.7
        BODY_WEIGHT = 0.3

        title_distance = self.calculate_tfidf_title_distance(tfidf_query)
        body_distance  = self.calculate_tfidf_body_distance(tfidf_query)
        
        self.score = (TITLE_WEIGHT * title_distance + BODY_WEIGHT * body_distance) * self.date_score * self.votes_score

        self.score = round(self.score, 3)

        return self.score
