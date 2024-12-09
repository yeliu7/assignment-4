import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from document import Document
from keygraph import KeyGraph

import os.path

@st.cache_data()
def load_corpus(file):
    documents = pd.read_csv(file)
    return documents

def show_corpus(corpus):
    st.dataframe(corpus)

def fit_document(corpus, user_stopwords, pos):
    document = get_saved_document()
    if document:
        return document
    document = Document(" ".join(corpus["content"]), 
        user_stopwords=user_stopwords.split('\n'), pos=pos)
    set_saved_document(document)
    return document

def get_saved_document():
    return st.session_state.get('document')

def set_saved_document(document):
    st.session_state['document'] = document

def fit_keygraph(doc, M=30, K=12):
    keygraph = get_saved_keygraph(M, K)
    if keygraph:
        return keygraph
    keygraph = KeyGraph(doc, M, K)
    set_saved_keygraph(M, K, keygraph)
    return keygraph

# Get saved state from st.session_state
def get_saved_keygraph(M, K):
	return st.session_state.get(f'keygraph_{M}_{K}')

# Save state using st.session_state
def set_saved_keygraph(M, K, keygraph):
	st.session_state[f'keygraph_{M}_{K}'] = keygraph

def show_keygraph(keygraph, corpus_file, M, K):
    file_name_base = os.path.splitext(corpus_file.name)[0]
    G = keygraph.draw()
    keygraph.write_to_file(G, file_name_base, M, K)
    with st.container():
        components.html(open(f"graphs/{file_name_base}_{M}_{K}.html", 'r', 
            encoding='utf-8').read(), height=625)

def find_concordance(corpus, word, window=5):
    concordance = []
    for index, row in corpus.iterrows():
        tokens = row['content'].split()
        for i, token in enumerate(tokens):
            if token == word:
                concordance.append(' '.join(tokens[i-window:i+window]))
    return concordance

st.sidebar.title("Chance Discovery")

corpus_file = st.sidebar.file_uploader("Corpus", type="csv",
	on_change=lambda: st.session_state.clear())
user_stopwords = st.sidebar.text_area("Stopwords (one per line)",
	on_change=lambda: st.session_state.clear())
# pos = st.sidebar.multiselect("Parts of Speech (at least one)", ["N", "V", "J", "R"], ["N"],
#     on_change=lambda: st.session_state.clear())
# if pos == []:
#     st.sidebar.error("Please select at least one part of speech.")
#     pos = ["N"] # default to nouns
pos = ["N", "V", "J", "R"]

if corpus_file is not None:
    corpus = load_corpus(corpus_file)

    # Drop NA
    # corpus = corpus.dropna()

    st.subheader("Corpus")
    show_corpus(corpus)

    M = st.sidebar.slider("M", 1, 30, 30)
    K = st.sidebar.slider("K", 1, 30, 12)

    doc = fit_document(corpus, user_stopwords, pos)  
    keygraph = fit_keygraph(doc, M=M, K=K)

    st.subheader("Key Graph")
    st.sidebar.write("Number of unique words:", len(keygraph.words))
    st.write(", ".join(keygraph.top_m_words))
    st.caption("Top-M unique words")

    show_keygraph(keygraph, corpus_file, M, K)
    st.caption("Keygraph")

    st.subheader("Concordance")
    word = st.text_input("Word")
    if word:
        concordance = find_concordance(corpus, word)
        for c in concordance:
            st.write(c)
        st.caption("Concordance")
