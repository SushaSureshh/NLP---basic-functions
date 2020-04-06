
'''This has unittest for the text_cleaning process'''

import unittest
import HtmlTestRunner
from utils import *


class TestUtils(unittest.TestCase):

    def test_named_entity_recognition(self):
        ''' Test for named entity recognition '''
        sentence = 'Susha Suresh is CEO of Apple'
        self.assertEqual(named_entity_recognition(sentence), [
                         ('Susha Suresh', 'PERSON'), ('Apple', 'ORG')])

    def test_pos_tagging(self):
        ''' Test for POS Tagging'''
        sentence = 'This test script checks the effeciency of POS  tagging'
        self.assertEqual(pos_tagging(sentence), [('This', 'DT'), ('test', 'NN'), ('script', 'NN'), (
            'checks', 'VBZ'), ('the', 'DT'), ('effeciency', 'NN'), ('of', 'IN'), ('POS', 'NNP'), ('tagging', 'VBG')])


if __name__ == '__main__':
    unittest.main(testRunner=HtmlTestRunner.HTMLTestRunner(
        output='entity_extraction/Test_report'))
