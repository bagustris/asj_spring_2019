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
                         RNN, SimpleRNN, Activation
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE

# from attention_helper import AttentionDecoder
#from read_csv import load_features

np.random.seed(99)

# load feature and labels
feat = np.load('../IEMOCAP-Emotion-Detection/X_egemaps.npy')
vad = np.load('../IEMOCAP-Emotion-Detection/y_egemaps.npy')

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
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
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
    inputs = Input(shape=(feat.shape[1], feat.shape[2]), name='feat_input')
    net = BatchNormalization()(inputs)
    net = CuDNNLSTM(64, return_sequences=True)(net)
    net = CuDNNLSTM(64, return_sequences=True)(net)
    net = CuDNNLSTM(64, return_sequences=True)(net)
    net = Flatten()(net)
    net = Dropout(0.4)(net)
    target_names = ('v', 'a', 'd')
    outputs = [Dense(1, name=name)(net) for name in target_names]

    model = Model(inputs=inputs, outputs=outputs) #=[out1, out2, out3])
    model.compile(loss=ccc_loss, #{'v': ccc_loss, 'a': ccc_loss, 'd': ccc_loss},
                  loss_weights={'v': alpha, 'a': beta, 'd': gamma},
                  optimizer='rmsprop', metrics=[ccc])
    return model

def main(alpha, beta, gamma):
    model = api_model(alpha, beta, gamma)
    model.summary()

    earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10, 
                              restore_best_weights=True)
    hist = model.fit(feat[:8000], vad[:8000].T.tolist(), batch_size=32, #best:8
                  validation_split=0.2, epochs=100, verbose=1, shuffle=True,
                  callbacks=[earlystop])
    metrik = model.evaluate(feat[8000:], vad[8000:].T.tolist())
    print(metrik)

if __name__ == '__main__':
    #alpha=0.1
    #beta=0.5
    #gamma=1-(alpha+beta)
    main(1, 1, 1)
