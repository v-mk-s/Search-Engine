import nltk

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

nltk.download('stopwords')
nltk.download("punkt")

stopwords = set(nltk.corpus.stopwords.words('english'))

lemmatizer = WordNetLemmatizer()


def remove_stop_words(text):
    text = text.lower()
    text_tokens = word_tokenize(text)
    tokens_without_sw = [w for w in text_tokens if not w in stopwords and w.isalnum()]
    return ' '.join(tokens_without_sw)


def lemmatize(text):
    text = text.lower()
    text_tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in text_tokens]
    return ' '.join(lemmatized_tokens)
