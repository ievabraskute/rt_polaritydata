#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
import os.path

import sklearn.model_selection as ms
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, GlobalMaxPool1D, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[34]:


# params
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 10000


# In[3]:


# get embeddings

embeddings_index = {}
with open(
    os.path.join('glove.twitter.27B', f'glove.twitter.27B.{EMBEDDING_DIM}d.txt'), 'r', encoding='utf-8'
) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print(f'Found {len(embeddings_index)} word vectors.')


# In[35]:


# load the separate positive and negative reviews datasets, 
# concatenate, shuffle them

with open('rt-polaritydata/rt-polarity.pos') as f:
    lines = f.readlines()
    pos = pd.DataFrame({'review': lines, 'target': [1]*len(lines)})
    
with open('rt-polaritydata/rt-polarity.neg') as f:
    lines = f.readlines()
    neg = pd.DataFrame({'review': lines, 'target': [0]*len(lines)})
    
input_data = pd.concat([pos, neg]).sample(frac=1)


# In[36]:


# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(input_data.review.values)
sequences = tokenizer.texts_to_sequences(input_data.review.values)

word_index = tokenizer.word_index
print(f'Found {len(word_index)} unique tokens.')

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = input_data.target.values

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[37]:


# split the data into a training set, validation set and test set

X_train_val, X_test, y_train_val, y_test = ms.train_test_split(
    data, labels, test_size = 0.2, shuffle=True, random_state=42
)

X_train, X_val, y_train, y_val = ms.train_test_split(
    X_train_val, y_train_val, test_size = 0.2, shuffle=True, random_state=42
)


# In[38]:


# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[42]:


EPOCHS = 10
BATCH_SIZE = 128
# STEPS_PER_EPOCH = np.ceil(X_train.shape[0]/BATCH_SIZE)


def build_model(embedding_matrix):
    sequence = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = Embedding(
        num_words,
        EMBEDDING_DIM,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False
    )(sequence)
    x = Bidirectional(
        LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)
    )(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.25)(x)
    result = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=sequence, outputs=result)
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  metrics=['acc'])

    return model

          
model = build_model(embedding_matrix)
model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    validation_data=(X_val, y_val)
)       


# In[43]:


# evaluate the model on the test set
model.evaluate(X_test, y_test, verbose=2)


# In[ ]:




