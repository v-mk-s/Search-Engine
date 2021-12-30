from sklearn.metrics.pairwise import cosine_similarity


class Document:
    """
    Document class

    Used for store each document and calculate
    its score based on tf-idf vector similarity

    Attributes:
        id: Unique index of document
        title: document title
        body: document body
        date_score: the newer document, the higher this value, from 1 to 2
        votes_score: the more upvoted document have the more this value, from 1 to 2
        title_top_10_tfidf: top 10 highest tfidf title values
        body_top_30_tfidf: top 30 highest tfidf title values
        score: score of document, calculates after calling calculate_score
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
        tf-idf vectors between document and search query.

        Args:
            tfidf_query: scipy sparse matrix, acquiring from transforming query by trained TfidfVectorizer

        Returns:
            Score is between 0 and 8
        """

        TITLE_WEIGHT = 0.7
        BODY_WEIGHT = 0.3

        title_distance = self.calculate_tfidf_title_distance(tfidf_query)
        body_distance  = self.calculate_tfidf_body_distance(tfidf_query)
        
        self.score = (TITLE_WEIGHT * title_distance + BODY_WEIGHT * body_distance) * self.date_score * self.votes_score

        self.score = round(self.score, 3)

        return self.score
