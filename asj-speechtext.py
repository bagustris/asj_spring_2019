# CSL Paper: Dimensional speech emotion recognition from acoustic and text
# Changelog:
# 2019-09-01: initial version
# 2019-10-06: optimizer MTL parameters with linear search (in progress)

import numpy as np
import pickle
import pandas as pd

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Flatten, \
                         Embedding, Dropout, Flatten, BatchNormalization, \
                         RNN, concatenate, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

#from keras_self_attention import SeqSelfAttention
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from attention_helper import AttentionDecoder
#from read_csv import load_features

import random as rn
import tensorflow as tf

rn.seed(123)
np.random.seed(99)
tf.set_random_seed(1234)

# load feature and labels
feat = np.load('../IEMOCAP-Emotion-Detection/X_egemaps.npy')
#vad = np.load('../IEMOCAP-Emotion-Detection/y_egemaps.npy')


# load text data
# loat output/label data
path = '../IEMOCAP-Emotion-Detection/'
pickle_path = path + 'data_collected_10039.pickle'

with open(pickle_path, 'rb') as handle:
    data = pickle.load(handle)
len(data)

v = [v['v'] for v in data]
a = [a['a'] for a in data]
d = [d['d'] for d in data]

vad = np.array([v, a, d])
vad = vad.T
print(vad.shape)

text = [t['transcription'] for t in data]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

MAX_SEQUENCE_LENGTH = len(max(text, key=len))
token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

import codecs
EMBEDDING_DIM = 300

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

file_loc = path + 'glove.840B.300d.txt'
print (file_loc)

gembeddings_index = {}
with codecs.open(file_loc, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        gembedding = np.asarray(values[1:], dtype='float32')
        gembeddings_index[word] = gembedding

f.close()
print('G Word embeddings:', len(gembeddings_index))

nb_words = len(word_index) +1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('nb words: {}'.format(nb_words))
print('G Null word embeddings: {}'.format(np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0)))

# remove outlier, < 1, > 5
vad = np.where(vad==5.5, 5.0, vad)
vad = np.where(vad==0.5, 1.0, vad)

print('Feature shape: ', feat.shape)
print('Label shape: ', vad.shape)

# standardization
scaled_feature = False

# set Dropout
do = 0.4

if scaled_feature == True:
    scaler = StandardScaler()
    scaler = scaler.fit(feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaler.transform(feat.reshape(feat.shape[0]*feat.shape[1], feat.shape[2]))
    scaled_feat = scaled_feat.reshape(feat.shape[0], feat.shape[1], feat.shape[2])
    feat = scaled_feat
else:
    feat = feat

scaled_vad = True

# standardization
if scaled_vad:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    scaled_vad = scaler.transform(vad) #.reshape(vad.shape[0]*vad.shape[1], vad.shape[2]))
    vad = scaled_vad 
else:
    vad = vad

# Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
def ccc(gold, pred):
    #gold       = K.squeeze(gold, axis=-1)
    #pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1,  keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    return ccc


def ccc_loss(gold, pred):  
    # input (num_batches, seq_len, 1)
    ccc_loss   = K.constant(1.) - ccc(gold, pred)
    return ccc_loss


# API model, if use RNN, first two rnn layer must return_sequences=True
def api_model(alpha, beta, gamma):
    # speech network
    input_speech = Input(shape=(feat.shape[1], feat.shape[2]), name='speech_input')
    net_speech = BatchNormalization()(input_speech)
    net_speech = CuDNNLSTM(64, return_sequences=True)(net_speech)
    net_speech = CuDNNLSTM(64, return_sequences=True)(net_speech)
    net_speech = Flatten()(net_speech)
    model_speech = Dropout(0.5)(net_speech)
    
    # text network
    input_text = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    model_text = Embedding(nb_words,
                           EMBEDDING_DIM,
                           weights = [g_word_embedding_matrix],
                           trainable = True)(input_text)
    model_text = CuDNNLSTM(64, return_sequences=True)(model_text)
    model_text = CuDNNLSTM(64, return_sequences=False)(model_text)
    model_text = Dropout(0.4)(model_text)
    
    # combined model
    model_combined = concatenate([model_speech, model_text])
    model_combined = Dense(32, activation='relu')(model_combined)
    model_combined = Dense(16, activation='relu')(model_combined)
    model_combined = Dropout(0.4)(model_combined)
#    target_names = ('v', 'a', 'd')
#    model_combined = [Dense(1, name=name)(model_combined) for name in target_names]
    model_combined = Dense(3, activation='linear')(model_combined)
    
    model = Model([input_speech, input_text], model_combined) 
    model.compile(loss=ccc_loss, optimizer='rmsprop', metrics=[ccc])
    #    model.compile(loss=ccc_loss, #{'v': ccc_loss, 'a': ccc_loss, 'd': ccc_loss},
    #                  loss_weights={'v': alpha, 'a': beta, 'd': gamma},
    #                  optimizer='rmsprop', metrics=[ccc])
    return model

def main(alpha, beta, gamma):
    model = api_model(alpha, beta, gamma)
    model.summary()

    earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    hist = model.fit([feat[:8000], x_train_text[:8000]], vad[:8000], batch_size=32  , #best:8
                  validation_split=0.2, epochs=50, verbose=1, shuffle=True,
                  callbacks=[earlystop])
    metrik = model.evaluate([feat[8000:], x_train_text[8000:]], vad[8000:])
    print(metrik)

if __name__ == '__main__':
    main(0.1, 0.5, 0.4)
