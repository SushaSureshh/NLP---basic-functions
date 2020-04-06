''' This module has all the basic nlp functions for keyword extraction'''


# import all libraries

from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
from nltk.corpus import stopwords


def get_top_n_words(data, num_words):
    ''' Takes an nlp corpus as input
        Returns n words along with their frequency based on word occurance
    '''
    # Initialize the count vectorizer
    vectorizer = CountVectorizer(
        ngram_range=(2, 2), stop_words=stopwords)
    # Fit the vectorizer
    bag_of_words = vectorizer.fit_transform(data)

    # Fit the bag of words matrix
    sum_words = bag_of_words.sum(axis=0)

    # Count the frequency of each word in the vocabulary
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vectorizer.vocabulary_.items()]

    # Sort the words based on the frequency
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[num_words]


def get_keyphrases(data, language):
    ''' Takes the data and returns the most importnat keywords in the text using RAKE algorithm
    '''
    rake = Rake(language)
    rake.extract_keywords_from_text(data)
    return rake.get_ranked_phrases_with_scores()
