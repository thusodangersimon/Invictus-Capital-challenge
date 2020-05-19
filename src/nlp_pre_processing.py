"""
Functions for NPL

Thuso Simon
"""
import re

import inflect
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import TransformerMixin


def to_lowercase(words: list) -> list:
    """
    Convert all characters to lowercase from list of tokenized words
    :param words: a list of words
    :return: list of words
    """
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words: list) -> list:
    """
    Remove punctuation from list of tokenized words
    :param words: a list of words
    :return: list of words
    """
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words: list) -> list:
    """
    Replace all integer occurrences in list of tokenized words with textual representation
    :param words: list of words
    :return: list of words
    """
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words: list) -> list:
    """
    Remove stop words from list of tokenized words
    :param words:
    :return:
    """
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words: list, language: str = 'english') -> list:
    """
    Stem words in list of tokenized words
    :param words:
    :param language: language library to get for stemming
    :return:
    """
    stemmer = SnowballStemmer(language)
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def normalize(words: list, language: str = 'english') -> list:
    """
    Run all cleaning steps
    :param words: list
    :param language: language to use
    :return: list
    """
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = stem_words(words, language)
    return words


class TextNormalization(TransformerMixin):
    """
    Does text normalization.

    Usage, give text column that needs to be cleaned.

    clean_df = TextNormalization.fit_transform(df.text, language='english')

    """

    def fit(self, df, y=None):
        return self

    def transform(self, df, language='english', **trans_params):
        output_df = df.str.split(' ').apply(normalize).apply(lambda s: ' '.join(s))
        return output_df
