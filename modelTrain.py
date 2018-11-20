# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:48:28 2018

@author: AkshayJk
"""

import jokes_csv_extractor as jokes_csv
import re
import plaidml.keras
import random
import math

plaidml.keras.install_backend()


from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.layers import Bidirectional
from keras import regularizers
from keras.layers import Dropout
import numpy as np

"""
completetext = '\n'.join(text_set)
chars = sorted(list(set(completetext)))
print('total chars:', len(chars))
"""


def process_data():
    text_set = jokes_csv.csvToTxt()
    
    text_set = list(text_set)
    rem_words = ['what', 'do', 'you', 'call', '', 'a', 'the']    
    print(text_set[0])
    lettercount = 0
    lettermax = 0
    lettermin = len(next(iter(text_set)))
    for line in text_set:
        lettercount += len(line)
        if len(line) > lettermax:
            lettermax = len(line)
            if len(line) < lettermin:
                lettermin = len(line)
    lettercount = lettercount/len(text_set)
    word_list = []
    word_count = 0
    wordmax = 0
    wordmin = len(next(iter(text_set)))

    for line in text_set:
        words = re.split(' ', line)
        for word in words:
            if '?' in word:
                word = word.strip('?')
            if 'q:'in word:
                word = word.strip('q:')
            if 'a:' in word:
                word = word.strip('a:')
            if word in rem_words:
                continue
            word_list.append(word)
        word_count += len(words)
        if len(words) > wordmax:
            wordmax = len(words)
        if len(words) < wordmin:
            wordmin = len(words)
    word_count = word_count/len(text_set)

    print("Total words = ", len(word_list))

    word_dict = dict()
    for word in word_list:
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1

    min_word_count = 1
    low_freq_words = set()
    for word in word_dict:
        if word_dict[word] < min_word_count:
            low_freq_words.add(word)
    words = sorted(set(word_list) - low_freq_words)
    print("Total unique words = ", len(word_dict))
    print("Count of words with high frequency = ", len(words))

    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))
    sentences = []
    next_words = []
    seq_length = int(word_count)
    ignored = 0
    for i in range(0, len(word_list) - seq_length, 1):
        if len(set(word_list[i: i + seq_length + 1]).intersection(low_freq_words)) == 0:
            sentences.append(word_list[i: i + seq_length])
            next_words.append(word_list[i + seq_length])
        else:
            ignored += 1
    print("No. of ignored sentences = ", ignored)
    print("No. of input sentences = ", len(sentences))
    return (sentences, next_words, seq_length, word_indices, indices_word, words)

def shuffle_set(sentences, next_words):
    complete_set = sentences
    train_next_words = []
    random.shuffle(sentences)
    for word in sentences:
         i = complete_set.index(word)
         train_next_words.append(next_words[i])
    return (sentences, train_next_words)


def split_train_test_set(sentences, next_words):
    complete_set = sentences
    train_next_words = []
    validation_next_words = []
    random.shuffle(sentences)
    split = int(0.9 * len(sentences))
    train_set = sentences[:split]
    for word in train_set:
         i = complete_set.index(word)
         train_next_words.append(next_words[i])
    validation_set = sentences[split:]
    for word in validation_set:
         i = complete_set.index(word)
         validation_next_words.append(next_words[i])
    return (train_set, train_next_words, validation_set, validation_next_words)

def generator(sentences, next_words, batch_size, word_indices, seq_len):
    index = 0
    while True:
        x = np.zeros((batch_size, seq_len, len(words)), dtype = np.bool)
        y = np.zeros((batch_size, len(words)), dtype = np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentences[index % len(sentences)]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_words[i % len(sentences)]]] = 1
            index += 1
        yield x, y

def sample(predictions, temperature = 1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions)/temperature
    ep_predictions = np.exp(predictions)
    predictions = ep_predictions / np.sum(ep_predictions)
    prob = np.random.multinomial(1, predictions, 1)
    return np.argmax(prob)

def create_model(dropout = 0.2):
    model = Sequential()
    model.add(Bidirectional(LSTM(8), input_shape = (seq_length, len(words))))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model

def on_epoch_end(epoch, logs):
    print("-------------------Generating text after epoch ", epoch)
    
    seed_index = np.random.randint(len(sentences))
    seed = sentences[seed_index]
    
    for diversity in [0.3, 0.4, 0.5, 0.6]:
        sentence = seed
        print("------------Diversity", diversity)
        print("------------Generated with seed: ",sentence)
        
        for i in range(2):
            x_pred = np.zeros((1, seq_length, len(words)))
            for t, w in enumerate(sentence):
                x_pred[0, t, word_indices[w]] = 1    
            predictions = model.predict(x_pred, verbose = 0)[0]
            next_index = sample(predictions, diversity)
            next_word = indices_word[next_index]
            sentence = sentence[1:]
            sentence.append(next_word)
            print(next_word)


batch_size = 64
print("Processing Data...")
sentences, next_words, seq_length, word_indices, indices_word, words = process_data()
print(sentences[0])
"""
print("Shuffling and Splitting Dataset...")
sentences_train, next_words_train, sentences_test, next_words_test = split_train_test_set(sentences, next_words)
print("Creating and compiling LSTM model...")
model = create_model()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print("Evaluating callbacks...")
checkpoint = ModelCheckpoint('pg_weights.hdf5', monitor='val_acc', save_best_only=True)
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
early_stopping = EarlyStopping(monitor='val_acc', patience=5)
callbacks_list = [checkpoint, print_callback, early_stopping]
print("Summarizing model...")
model.summary()
print("Fitting model to training set and validating on testing set...")
model.fit_generator(generator(sentences_train, next_words_train, batch_size, word_indices, seq_length), steps_per_epoch=int(len(sentences_train)/batch_size) + 1, epochs=10, callbacks=callbacks_list, validation_data=generator(sentences_test, next_words_test, batch_size, word_indices, seq_length), validation_steps=int(len(sentences_test)/batch_size) + 1)
"""