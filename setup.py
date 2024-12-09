#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 19:35:52 2021

@author: mrw
"""

import nltk

def setup():
    # Obtain NLTK resources
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('vader_lexicon')

#-----------Main----------------
if __name__ == "__main__":
    setup()