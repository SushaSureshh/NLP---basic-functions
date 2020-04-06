''' This module has all the basic entity extraction nlp functions'''


# import all libraries
import nltk
import spacy
import en_core_web_sm  # download the model for English NER
spacy.load('fr_core_news_sm')  # download the model for French NER


def named_entity_recognition(data):
    ''' 
    Using Spacy.
    Named Entity recognition is an entity extraction technique,
    used to identify Name, Organization etc. This function seeks to locate and classify named entities in text 
    into pre-defined categories such as the names of persons, organizations, locations, 
    expressions of times, quantities, monetary values, percentages, etc.  
    '''
    # load model package "en_core_web_sm" for English and "fr_core_web_sm" for French
    nlp_en = en_core_web_sm.load()
    #nlp_fr = spacy.load("fr_core_news_sm")

    # Find the named entity using the model that has been loaded
    ner = nlp_en(data)
    #ner = nlp_fr(data)

    return([(X.text, X.label_) for X in ner.ents])


def pos_tagging(data):
    ''' 
    Find the POS Tags for all the word in the corpus

    - Noun (N)- Daniel, London, table, dog, teacher, pen, city, happiness, hope
    - Verb (V)- go, speak, run, eat, play, live, walk, have, like, are, is
    - Adjective (ADJ)- big, happy, green, young, fun, crazy, three
    - Adverb (ADV)- slowly, quietly, very, always, never, too, well, tomorrow
    - Preposition (P)- at, on, in, from, with, near, between, about, under
    - Conjunction (CON)- and, or, but, because, so, yet, unless, since, if
    - Pronoun (PRO)- I, you, we, they, he, she, it, me, us, them, him, her, this
    - Interjection (INT)- Ouch! Wow! Great! Help! Oh! Hey! Hi!
    '''
    token = nltk.word_tokenize(data)
    #  POS Tags for each word in the document
    return nltk.pos_tag(token)


def chunking(data):
    ''' 
    Group together words that are specific types of POS
    '''
    # Define pattern for chunking
    pattern = """mychunk:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""  # define the grammar here

    # Initialise the chunker
    chunker = nltk.RegexpParser(pattern)

    data = pos_tagging(data)

    # Chunk and parse the documents based on the defined grammar
    data = chunker.parse(data)

    return data
