''' This program extracts the most common phrases from
English as well as French text'''


# import all libraries

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import seaborn as sns
from pandas_ods_reader import read_ods
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')


class MostFrequentBigrams():
    ''' Define a model, train the model'''

    def __init__(self, language):
        self.language = language
        self.init_stopwords()
        self.load_data()
        self.top_df = self.top_df
        self.data = self.data
        self.num_words = 20  # Can be any number of bi-grams you want to print

    def init_stopwords(self):
        '''
        Initialize stopwords attribute
        '''
        self.stopwords = stopwords.words(self.language)
        self.stopwords.append(',')
        self.stopwords.append('  ')
        self.stopwords.append(' ')

    def load_data(self):
        '''
        Load your data by specifying the path
        '''
        sheet_idx = 1
        self.data_frame = read_ods('input.ods', sheet_idx)
        self.data = self.data_frame['LeçonsApprises']
        return self.data
        # self.data_frame = read_csv('file name')
        # self.dataframe = read_csv('path of the file)

    def clean_data(self):
        '''
        Preprocessing the data
        Return the nlp corpus
        '''
        # To lower case
        self.data = self.data.str.lower(
        )

        # Strip out HTML
        self.data = self.data.str.replace(
            r'<.*?>', ' ')

        # Remove special characters
        self.data = self.data.str.replace(
            r'[^\w]', ' ')

        # Removes numbers
        self.data = self.data.str.replace(
            r'\b\d+\b', ' ')

        # Remove single letter words
        self.data = self.data.str.replace(
            r'\b\w\b', ' ')

        # Replace any series of white space with a single space character
        self.data = self.data.str.replace(
            r'\s+', ' ')

        # create the corpus
        nlp_corpus = [''.join(x) for x in self.data]

        return nlp_corpus

    def get_top_n_words(self, nlp_corpus):
        ''' Takes an nlp corpus as input
            Returns n words along with their frequency
        '''
        # Initialize the count vectorizer
        vectorizer = CountVectorizer(
            ngram_range=(2, 2), stop_words=self.stopwords)
        # Fit the vectorizer
        bag_of_words = vectorizer.fit_transform(nlp_corpus)

        # Fit the bag of words matrix
        sum_words = bag_of_words.sum(axis=0)

        # Count the frequency of each word in the vocabulary
        words_freq = [(word, sum_words[0, idx]) for word, idx in
                      vectorizer.vocabulary_.items()]

        # Sort the words based on the frequency
        words_freq = sorted(words_freq, key=lambda x: x[1],
                            reverse=True)
        return words_freq[:self.num_words]

    def visualize(self, words_freq):
        ''' Visualize the bi-grams'''
        self.top_df = pd.DataFrame(words_freq)
        self.top_df.columns = ["Bi-gram", "Freq"]
        print(self.top_df)
        # Barplot of most freq Bi-grams
        sns.set(rc={'figure.figsize': (13, 8)})
        graph = sns.barplot(x="Bi-gram", y="Freq", data=self.top_df)
        graph.set_xticklabels(graph.get_xticklabels(), rotation=45)

    def run(self):
        '''
        Train and upload the model
        '''
        nlp_corpus = self.clean_data()
        words_freq = self.get_top_n_words(nlp_corpus)
        self.visualize(words_freq)


def main():
    '''
    Main fucntion
    for training english and french models
    '''
    # most_frequent_bigrams("english").run()
    MostFrequentBigrams("french").run()


if __name__ == '__main__':
    main()
