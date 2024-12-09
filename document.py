#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import codecs
import nltk

class Document:
    def __init__(self, content = "", file_name = None, user_stopwords = [], pos = ['N', 'V', 'J', 'R']):
        self.content = self.read_from_file(file_name) if file_name else content
        self.stopwords = self.read_stopwords()
        self.stopwords.extend(user_stopwords)
        self.sentences = self.create_sentences(pos)
        self.tokens = self.create_tokens()
    
    # Read content from file
    def read_from_file(self, file_name):
        f = codecs.open(file_name, 'r', 'utf-8')
        content = f.read()
        f.close()
        return content
    
    # Read user-defined stopwords
    def read_stopwords(self):
        stopwords = nltk.corpus.stopwords.words('english')
        for line in codecs.open('./noise/stopwords.txt', 'r', 'utf-8'):
            stopwords.append(line.strip())
        symbols = ["'", '"', '“', '”', '`', '’', '.', ',', '-', '!', '?', ':', ';', '(', ')', '[', ']', '&', '0', '%', '...', '--', '``', "''"]
        return stopwords + symbols

    # Divide into sentences
    def create_sentences(self, pos):
        return [self.create_tokens_from(s, pos) for s in nltk.tokenize.sent_tokenize(self.content)]

    # Divide a string into tokens
    def create_tokens_from(self, s, pos, lemmatized=True, strip_stopwords=True):
        tokens = [t.lower() for t in nltk.tokenize.word_tokenize(s)]
        tokens = self.lemmatize(tokens, pos) if lemmatized else tokens
        return self.strip_stopwords(tokens) if strip_stopwords else tokens
    
    # Strip stopwords and symbols from tokens
    def strip_stopwords(self, tokens):
        return [t for t in tokens if not t in self.stopwords and len(t) > 1]
        
    # Divide into tokens
    def create_tokens(self):
        return [t for s in self.sentences for t in s]

    # Lemmatize words
    def lemmatize(self, tokens, pos):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return [lemmatizer.lemmatize(w, self.wordnet_pos(t)) 
                for w, t in nltk.pos_tag(tokens) if t[0] in pos]
    
    # Lookup WordNet POS
    # https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    def wordnet_pos(self, tag):
        tags = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
        # tag example: 'VBD' for verb
        return tags.get(tag[0], nltk.corpus.wordnet.NOUN)
   
    # Count word frequencies
    def freq_count(self):
        result = {}
        for t in self.tokens:
            if t in result:
                result[t] += 1
            else:
                result[t] = 1
        return result
    
     
