# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:46:37 2023

@author: iru-ra2
"""

import pandas as pd
import re, spacy, gensim
from nltk.corpus import stopwords
#Gensim
import pyLDAvis
import pyLDAvis.lda_model
import pyLDAvis.gensim
from gensim import models
from gensim.models import LdaModel, Phrases
import gensim.corpora as corpora

#----------Base
#display maximum table
pd.set_option('display.max_column',10)
pd.set_option('display.max_rows',10)
# Load spaCy language model
nlp = spacy.load('en_core_web_sm',disable = ['parser','ner'])

#file selection to open
excel_file_path = "C:/Users/chie_/Downloads/twitter_training.csv/twitter_training.csv"
df = pd.read_csv(excel_file_path)
df = df.sample(n=4000,random_state=42) #trucate the sample size due to computation limitation
print(df.head())
#selection
num_topics = 5
minwordfreq = 3
#selection of excel
start_row   = 0
start_column= 3

#---------Date cleaning
#covert to list
selected_column = df.iloc[start_row:, start_column] 
#removing blank and Nil
data = [str(x) for x in selected_column if pd.notna(x) and not isinstance(x, float) and 
        'Nil' not in str(x) and 'nil' not in str(x) and 'NIL' not in str(x)]
#removing symbols
datas = [re.split(r'[\.+\,+\!+\?+\+\_+\-"]',sent) for sent in data]

#Simple pre-processing > remove punctuation, splitting sentence to word and convert to lowercase
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True)) #deacc=True removes punctuations
data_words = list(sent_to_words(datas))

stoplist = ['the', 'I','m','self','our','who','where','after','but','t','am','their','what',
            'me','are','as','about','on','over','once','myself','off','ma','having','because','whom','herself'
            ,'ourselves','d','very','a','i','than','by','how','few','all','now','these','which','was','be','why',
            'some','so','she','is','below','itself','above','ve','had','or','were','only','each','his',
            'he','it','doing','when','until','o','from','them','other','into','with','same','can','your',
            'to','my','here','its','themselves','you','ll','before','re','y','himself','there','own','did','between',
            'that','for','her','yourselves','do','in','they','yours','through','this','while','been','and','at',
            'just','nor','if','has','those','feel','felt','think','we','us','let','of','please','will'
            ,'an','would','could','whether','yet','make']
data_stop = [
    [word for word in sentence if word not in stoplist]
    for sentence in data_words]

#lemmatising words
def lemmatization(texts, allowed_postags =['NOUN', 'ADJ', 'VERB', 'ADV']):#'NOUN', 'ADJ', 'VERB', 'ADV'
    texts_out = []
    for sent in texts:
        doc=nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out
data_lemmatized = lemmatization(data_stop, allowed_postags=['NOUN','VERB','ADV','ADJ'])

#tokenisation
def preprocess(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if token.text.lower()]
    return tokens
tokens = [preprocess(text) for text in data_lemmatized]

#Bag-of-word; min-count > frequent of the 2 words joining together; #threshold is the strength of relevance of the 2 words
bigram_phraser = Phrases(tokens, min_count=2, threshold =2)
tokens_with_bigrams = [bigram_phraser[doc] for doc in tokens]

#building dictonary and word count
dictionary = corpora.Dictionary(tokens_with_bigrams)
id2token = dictionary.id2token
bow_corpus = [dictionary.doc2bow(doc) for doc in tokens_with_bigrams] #or use tokens

tfidf = models.TfidfModel(bow_corpus)
tfidf_corpus = tfidf[bow_corpus]

#-------------LDA model
lda_gensim = LdaModel(corpus=tfidf_corpus,
                      num_topics=num_topics,
                      id2word=dictionary,
                      alpha='auto',
                      eta='auto',
                      iterations = 100,
                      random_state=42)

#-------------pyLDAvis 
pyLDAvis.enable_notebook()
p = pyLDAvis.gensim.prepare(lda_gensim,tfidf_corpus,dictionary)
# Save the visualization as an HTML file
html_file_path = "C:/Users/chie_/Desktop/Python/visualization4.html"
pyLDAvis.save_html(p, html_file_path)
