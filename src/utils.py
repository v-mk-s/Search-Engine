import nltk
import re

from nltk.stem import WordNetLemmatizer

stopwords = set(nltk.corpus.stopwords.words('english'))

lemmatizer = WordNetLemmatizer()


def preprocess(sentence):
    sentence = sentence.lower()
    word_list = nltk.word_tokenize(sentence)
    word_list = [word for word in word_list if not word in stopwords]
    word_list = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return word_list


def preprocess_title(sentence):
    sentence = preprocess(sentence)
    return sentence


def preprocess_body(sentence):
    CLEANR = re.compile('<(.*?)>')
    sentence = re.sub(CLEANR, '', sentence)
    sentence = preprocess(sentence)
    return sentence
