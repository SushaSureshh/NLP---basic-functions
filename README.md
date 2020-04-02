# NLP---basic-functions
For cleaning the data I have the following functions: 

1.  Replace any series of white space with a single space character
2.  Strip out HTML
3.  Remove special characters and numbers
4.  Convert all the upper case words to lowercase - This will change for Part Of Speech (POS) tagging and Named Entity Recognition (NER)
5.  Remove single letter words
6. Remove contractions- Example you've to you have
7. Normalize data by removing extra white spaces, punctuations, and article.

For pre-processing I have the following:

1.  Tokenize the text : Convert sentences in the document to words and treat them as single tokens. We have both word and line tokenization.
2.  Removes stop words for both French and English : Stop words are words like "the", "a", etc. Stop words make your data noisy, removing stopwords is important in any NLP tasks.
3. Stemming of the text : Words " swim " and "swimming" mean the same, in this package we have porter stemmer, lancaster stemmer, snowball stemmer which supports multiple languages like French, Danish, Dutch, German, Hungarian, Italian, Norwegian, Spanish, Swedish, etc. 
4. Lemmatization of the text :  Lemmatization is process of grouping together words which has similar meanings. Example Cars, car's, car --> car. This package has wordnet lemmatizer.
4.  Count vectorizer: Vectorization help represent the words in a text as vectors so that Machine Learning algorithms can further process them. This package has tf-idf vectorizer.

