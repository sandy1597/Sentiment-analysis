#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow.keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, SimpleRNN, Activation, Dropout, Conv1D
from tensorflow.keras.layers import Embedding, Flatten, LSTM, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import spacy
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,auc


data = pd.read_csv("data.csv", header=None, encoding='latin-1')
data.columns = ["index","label","tweet"]
data = data.iloc[1: , :]
#data.head()



import spacy
from spacy.cli.download import download
download(model="en_core_web_sm")



def glove(files):
    model = {}
    with open(files, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embeddings = [float(val) for val in split_line[1:]]
            model[word] = embeddings
    print("[INFO] Done...{} words loaded!".format(len(model)))
    return model
# adopted from utils.py
# nlp = spacy.load("en")
nlp = spacy.load("en_core_web_sm")

def stopwords_remover(input_str):
    '''
    function to remove stopwords
        input: sentence - string of sentence
    '''
    new_lis = []
    # tokenize sentence
    input_str = nlp(input_str)
    for i in input_str:
        if (i.is_stop == False) & (i.pos_ !="PUNCT"):
            new_lis.append(i.string.strip())
    # convert back to sentence string
    ans = " ".join(str(x) for x in new_lis)
    return ans


def lemmatization(input_str):
    '''
    function to do lemmatization
        input: sentence - string of sentence
    '''
    input_str = nlp(input_str)
    ans = ""
    for w in input_str:
        ans +=" "+w.lemma_
    return nlp(ans)

def vectorizer(sentence, model):
    '''
    sentence vectorizer using the pretrained glove model
    '''
    ans_vector = np.zeros(200)
    num_w = 0
    for w in sentence.split():
        try:
            # add up all token vectors to a sent_vector
            ans_vector = np.add(ans_vector, model[str(w)])
            num_w += 1
        except:
            pass
    return ans_vector
data_X = data[data.columns[2]].to_numpy()
data_y = data[data.columns[1]]
data_y = pd.get_dummies(data_y).to_numpy()
glovemodel = glove("glove.twitter.27B.200d.txt")
mxvocab = 18000
mxlen = 15
token = Tokenizer(num_words=mxvocab)
token.fit_on_texts(data_X)
sequences = token.texts_to_sequences(data_X)
wordindex = token.word_index
data_keras = pad_sequences(sequences, maxlen=mxlen, padding="post")
from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(data_keras, data_y, test_size = 0.1, random_state=42)
# calcultaete number of words
nb_words = len(token.word_index) + 1
print(nb_words)

# obtain the word embedding matrix
matrix = np.zeros((nb_words, 200))
for word, i in wordindex.items():
    embeddingvector = glovemodel.get(word)
    if embeddingvector is not None:
        matrix[i] = embeddingvector
print('Null word embeddings: %d' % np.sum(np.sum(matrix, axis=1) == 0))
# adopted from sent_tran_eval.py
'''
changes made here Modell is new class we created

'''
class Modell:
  def model_build(self,nb_words, rnn_model="SimpleRNN",matrix=None):
    model = Sequential()
    if matrix is not None:
        model.add(Embedding(nb_words, 
                        200, 
                        weights=[matrix], 
                        input_length= mxlen,
                        trainable = False))
    else:
        model.add(Embedding(nb_words, 
                        200, 
                        input_length= mxlen,
                        trainable = False))
        

    if rnn_model == "SimpleRNN":
        model.add(SimpleRNN(200))
    elif rnn_model == "GRU":
        model.add(GRU(200))
    else:
        model.add(LSTM(200))

    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
    return model







