from gensim.models import KeyedVectors
from keras.preprocessing.text import *

import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
word2vec = KeyedVectors.load_word2vec_format('WordEmbeddings/Word2Vec.bin',binary=True)

def TokenCreator(sentences,tokenizer=None):
    
    temp_wordlist = None
    if(tokenizer == None):
        tokenizer2 = Tokenizer()
        tokenizer2.fit_on_texts(sentences)
        
        temp_wordlist = tokenizer2.texts_to_sequences(sentences)
        return (temp_wordlist,tokenizer2)
    
    else:
        temp_wordlist = tokenizer.texts_to_sequences(sentences)
        return temp_wordlist
    
def find_vocab(dataset):
    lst=[]
    for i in dataset:
        for j in i:
            lst.append(j)
    st = set(lst)
    return st


def padding(X_train,Y_train,X_test,Y_test,X_val,Y_val):
    X_padded_train = pad_sequences(X_train, maxlen=100, padding="pre", truncating="post")
    Y_padded_train = pad_sequences(Y_train, maxlen=100, padding="pre", truncating="post")

    X_padded_test = pad_sequences(X_test, maxlen=100, padding="pre", truncating="post")
    Y_padded_test = pad_sequences(Y_test, maxlen=100, padding="pre", truncating="post")

    X_padded_val = pad_sequences(X_val, maxlen=100, padding="pre", truncating="post")
    Y_padded_val = pad_sequences(Y_val, maxlen=100, padding="pre", truncating="post")

    return X_padded_train,Y_padded_train,X_padded_test,Y_padded_test,X_padded_val,Y_padded_val


def categorize_labels(Y_train,Y_test,Y_val):
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    Y_val = to_categorical(Y_val)
    return Y_train,Y_test,Y_val

def word2vectorize(X_train,Y_train,X_test,Y_test,X_val,Y_val):
    
    #creating vocab
    st1 = find_vocab(X_train)
    vocab_ate  = list(st1)
    vocab_ate.append("#UK")

    #appending unknown word token
    for i in X_test:
        for j in range(0,len(i)):
            if i[j] not in st1:
                i[j]="#UK"

    for i in X_val:
        for j in range(0,len(i)):
            if i[j] not in st1:
                i[j]="#UK"

    #generating tokens
    X_train_ate_tokenized, toke = TokenCreator(X_train)
    X_test_ate_tokenized = TokenCreator(X_test, tokenizer = toke)
    X_val_ate_tokenized = TokenCreator(X_val, tokenizer = toke)
    Y_train_ate_tokenized,toke2 = TokenCreator(Y_train)
    Y_test_ate_tokenized = TokenCreator(Y_test, tokenizer = toke2)
    Y_val_ate_tokenized = TokenCreator(Y_val, tokenizer = toke2)

    #generating weights
    vocab_size_ate = len(vocab_ate) 
    weights = np.zeros((vocab_size_ate, 300))
    mapping = toke.word_index 
    for word, index in mapping.items():
        try:
            weights[index:] = word2vec[word]
        except KeyError:
            pass

    return X_train_ate_tokenized,Y_train_ate_tokenized,X_test_ate_tokenized,Y_test_ate_tokenized,X_val_ate_tokenized,Y_val_ate_tokenized,weights,vocab_ate

