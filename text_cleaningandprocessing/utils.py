''' This module has all the basic nlp functions'''


# import all libraries

import re
import unicodedata
import string
import regex as re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import LineTokenizer
nltk.download('stopwords')
nltk.download('punkt')


def load_data(data_path):
    ''' Load your data by specifying the path.'''
    data = pd.read_csv(data_path)

    return data


def normalize_text(data):
    ''' Lower text and remove punctuation, articles and extra white space.'''
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(data))))


def strip_html_tags(data):
    '''Strip out html tags'''
    strip = re.compile('<.*?>')
    return re.sub(strip, '', data)


def remove_accented_chars(data):
    ''' Remove accent '''
    data = unicodedata.normalize('NFKD', data).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')
    return data


def expand_contractions(phrase):
    ''' Remove contractions like you've to you have '''
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def remove_special_characters(data, remove_digits):
    ''' Remove_digits will have either True or False value '''
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    data = re.sub(pattern, '', data)
    return data


def word_tokenizer(data):
    ''' Tokenize the text on word level '''
    return nltk.word_tokenize(data)


def line_tokenizer(data, blanklines):
    ''' Tokenize the text on line level ie, after \n'''
    tokenizer = LineTokenizer(blanklines=blanklines)
    return tokenizer.tokenize(data)


def sentence_tokenizer(data):
    ''' Tokenize the text on sentence level ie, after \n'''
    return nltk.sent_tokenize(data)


def remove_stopwords(data, language):
    ''' Initialize stopwords attribute. Language is specified within the quotes example: 'french' and 'english' '''
    word_tokens = word_tokenizer(data)
    stop_word = stopwords.words(language)
    filtered_sentence = [w for w in word_tokens if not w in stop_word]
    return filtered_sentence


def porter_stemming(data):
    ''' Stems the English text using PorterStemmer. '''
    porter = PorterStemmer()
    data = word_tokenizer(data)
    stemmed = [porter.stem(word) for word in data]
    return stemmed


def lancaster_stemming(data):
    ''' Stems the English text using LancasterStemmer. '''
    lancaster = LancasterStemmer()
    data = word_tokenizer(data)
    stemmed = [lancaster.stem(word) for word in data]
    return stemmed


def snowball_stemming(data, language):
    ''' Stems the English text using SnowballStemmer. 
    It is an iterative algorithm and 
    has stemming in multiple languages like : English, Fench, Danish
    Dutch, German, Hungarian, Italian, Norwegian, Spanish, Swedish etc. '''
    snowball = SnowballStemmer(language)
    data = word_tokenizer(data)
    stemmed = [snowball.stem(word) for word in data]
    return stemmed


def wordnet_lemmatizer(data):
    ''' Lemmatizes the English text '''
    wordnet = WordNetLemmatizer()
    data = word_tokenizer(data)
    lemmetized = [wordnet.lemmatize(word)
                  for word in data]  # Can define the POS here
    return lemmetized


def tf_idf(data, encoding, analyzer, stop, ngram_range, max_df, min_df, max_features):
    ''' Convert a collection of raw documents to a matrix of TF-IDF features   '''
    vectorizer = TfidfVectorizer(encoding=encoding, analyzer=analyzer, stop_words=stop,
                                 ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)
    transformed_data = vectorizer.fit_transform(data)
    return transformed_data
