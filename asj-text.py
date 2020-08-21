# OCOCOSDA 2019: dimensional speech emotion recognition from text feature

# uncomment these to run on CPU only
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Masking, CuDNNLSTM, TimeDistributed, Bidirectional, Embedding, Dropout, Flatten, concatenate, CuDNNGRU
from keras.utils import to_categorical
from sklearn.preprocessing import label_binarize

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping

np.random.seed(99)

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

# loat output/label data
path = '/home/s1820002/IEMOCAP-Emotion-Detection/'
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

#MAX_SEQUENCE_LENGTH = len(max(text, key=len))

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []
MAX_SEQUENCE_LENGTH = len(max(token_tr_X, key=len))
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

nb_words = len(word_index) + 1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))

# split train/test
split = 8500
earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min')

# model: GRU
def text_model1():
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ))
    #net = Embedding(2737, 128, input_length=500)(inputs)
    net = Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    trainable = True)(inputs)
    net = CuDNNLSTM(32, return_sequences=True)(net)
    net = CuDNNLSTM(32, return_sequences=False)(net)
                                                    
    #net = Dense(32)(net)
    net = Dropout(0.3)(net)
    net = Dense(3)(net) #linear activation
    model = Model(inputs=inputs, outputs=net) #[out1, out2, out3]
    model.compile(optimizer='rmsprop', loss=ccc_loss, metrics= [ccc])
    
    return model

model1 = text_model1()
hist1 = model1.fit(x_train_text[:split], vad[:split], epochs=30, batch_size=32, verbose=1, validation_split=0.2, callbacks=[earlystop])
#acc1 = hist1.history['val_mean_absolute_percentage_error']
#print('max: {:.4f}, min:{:.4f}, avg:{:.4f}'.format(max(acc1), min(acc1), np.mean(acc1)))
eval_metrik1 = model1.evaluate(x_train_text[split:], vad[split:])
print(eval_metrik1)
