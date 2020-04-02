
'''This has unittest for the text_cleaning process'''

import unittest
import HtmlTestRunner
from utils import *


class TestUtils(unittest.TestCase):

    def test_word_tokenizer(self):
        ''' Test for tokenization of the text on word level '''
        sentence = 'hello world'
        self.assertEqual(word_tokenizer(sentence), ['hello', 'world'])

    def test_normalize_text(self):
        ''' Test to check the text to lower case and remove punctuation, articles and extra whitespace.'''
        sentence = 'hello a    World!'
        self.assertEqual(normalize_text(sentence), 'hello world')

    def test_strip_html_tags(self):
        ''' Test to check the stripping of the html tags'''
        sentence = " I am loving <html>"
        self.assertEqual(strip_html_tags(
            sentence), " I am loving ")

    def test_remove_special_characters(self):
        ''' Test to check the removal of the special characters '''
        sentence = "This script tests the removal of special characters like 1#@#@3224"
        self.assertEqual(remove_special_characters(
            sentence, True), 'This script tests the removal of special characters like ')

    def test_expand_contractions(self):
        ''' Test to check contractions'''
        sentence = "This script tests the removal of contractions that I've"
        self.assertEqual(expand_contractions(
            sentence), 'This script tests the removal of contractions that I have')

    def test_remove_accented_chars(self):
        ''' Test to evaluate the results of accent removal method'''
        sentence = "Comment ça va ?"
        self.assertEqual(remove_accented_chars(
            sentence), 'Comment ca va ?')

    def test_line_tokenizer(self):
        ''' Test to evaluate line tokenizer'''
        sentence = "This script is to test\n The line tokenizer."
        self.assertEqual(line_tokenizer(
            sentence, 'discard'), ['This script is to test', ' The line tokenizer.'])

    def test_sentence_tokenizer(self):
        ''' Test to evaluate sentence tokenizer'''
        sentence = "This script is to test. The sentence tokenizer !"
        self.assertEqual(sentence_tokenizer(
            sentence), ['This script is to test.', 'The sentence tokenizer !'])

    def test_remove_stopwords(self):
        ''' Test to evaluate stopword removal'''
        sentence = "This script is to test stopwords in a sentence"
        self.assertEqual(remove_stopwords(
            sentence, 'english'), ['This', 'script', 'test', 'stopwords', 'sentence'])

    def test_porter_stemming(self):
        ''' Test to evaluate porter stemming'''
        sentence = "This is a scripting to test stemming on agreed condetions"
        self.assertEqual(porter_stemming(
            sentence), ['thi', 'is', 'a', 'script', 'to', 'test', 'stem', 'on', 'agre', 'condet'])

    def test_lancaster_stemming(self):
        ''' Test to evaluate lancaster stemming'''
        sentence = "This is a scripting to test stemming on agreed condetions"
        self.assertEqual(lancaster_stemming(
            sentence), ['thi', 'is', 'a', 'scripting', 'to', 'test', 'stem', 'on', 'agree', 'condet'])

    def test_snowball_stemming(self):
        ''' Test to evaluate snowball stemming in French'''
        sentence = "voudrais non animaux yeux dors couvre"
        self.assertEqual(snowball_stemming(
            sentence, 'french'), ['voudr', 'non', 'animal', 'yeux', 'dor', 'couvr'])

    def test_wordnet_lemmatizer(self):
        ''' Test to evaluate wordnet lemmatizing'''
        sentence = "This is a scrit to test cats using abaci and cacti"
        self.assertEqual(wordnet_lemmatizer(
            sentence), ['This', 'is', 'a', 'scrit', 'to', 'test', 'cat', 'using', 'abacus', 'and', 'cactus'])


if __name__ == '__main__':
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(
        output='text_cleaningandprocessing/Test_report'))
