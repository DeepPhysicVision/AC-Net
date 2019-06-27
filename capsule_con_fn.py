import keras
import numpy as np
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.engine import Layer
from keras.layers import Activation, Add, Bidirectional, Conv1D, Dense, Dropout, Embedding, TimeDistributed,Flatten
from keras.layers import concatenate,GRU,Input,K,LSTM,Lambda,MaxPooling1D,merge,GlobalAveragePooling1D,GlobalMaxPooling1D,SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from capsule_fn import CapsuleLayer

class MyFlatten(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MyFlatten, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask==None:
            return mask
        return K.batch_flatten(mask)

    def call(self, inputs, mask=None):
        return K.batch_flatten(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))

EMBEDDING_DIM = 300
POSITION_EMBEDDING_DIM = 50
MAX_LEN = 82
dropout_p = 0.3
rate_drop_dense = 0.3

def reduce_dimension(x, length, mask):
    res = K.reshape(x, [-1, length])  # n*30   x=n*30*1
    res = K.softmax(res)
    res = res * K.cast(mask, dtype='float32')  # n*30
    temp = K.sum(res, axis=1, keepdims=True)  # n*1
    temp = K.repeat_elements(temp, rep=length, axis=1)  #n*30
    return res / temp
def reduce_dimension_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    return [shape[0], shape[1]]

def attention(x, dim):
    res = K.batch_dot(x[0], x[1], axes=[1, 1])
    #x = [[1, 2], [3, 4]] 和 y = [[5, 6], [7, 8]]， batch_dot(x, y, axes=1) = [[17], [53]]
    return K.reshape(res, [-1, dim])
def attention_output_shape(input_shape):
    shape = list(input_shape[1])
    assert len(shape) == 3
    return [shape[0], shape[2]]

def no_change(input_shape):
    return input_shape

def liter(x, length):
    res = K.repeat(x, length)  # (?, 82, 300)
    return res
def liter_output_shape(input_shape):
    shape = list(input_shape)
    return [shape[0], MAX_LEN, shape[1]]

def build_model(max_len, aspect_max_len, embedding_matrix=[],position_embedding_matrix=[], class_num=3, num_words=10000):
    MAX_LEN = max_len #82
    #MAXLEN = maxlen #30

    sentence_input = Input(shape=(max_len,), dtype='int32', name='sentence_input')  # n*82
    position_input = Input(shape=(max_len,), dtype='int32', name='position_input')  # n*82
    aspect_input = Input(shape=(aspect_max_len,), dtype='int32', name='aspect_input')  # n*82

    sentence_embedding_layer = Embedding(num_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],input_length=max_len, trainable=False, mask_zero=True)
    sentence_embedding = sentence_embedding_layer(sentence_input)  # n*82*300
    position_embedding = Embedding(max_len * 2, POSITION_EMBEDDING_DIM, weights=[position_embedding_matrix],
                                   input_length=max_len, trainable=True, mask_zero=True)(position_input)  # n*82*50
    input_embedding = keras.layers.concatenate([sentence_embedding, position_embedding])  # n*82*350
    aspect_embedding_layer = Embedding(num_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],input_length=aspect_max_len, trainable=False, mask_zero=True)
    aspect_embedding = aspect_embedding_layer(aspect_input)  # n*9*300

    #encode_x = Bidirectional(GRU(300, activation="relu",return_sequences=True, recurrent_dropout=0.5, dropout=0.5))(input_embedding)  # n*82*600
    #aspect_embedding = Bidirectional(GRU(300, activation="relu", return_sequences=True,recurrent_dropout=0.5, dropout=0.5))(aspect_embedding)  #n*9*600

    #xcapsule = Capsule(num_capsule=32,dim_capsule=600,routings=3,share_weights=True)(encode_x)    #n*30*600

    aspect_attention = TimeDistributed(Dense(1, activation='tanh'))(aspect_embedding)  #n*9*1
    aspect_attention = Lambda(reduce_dimension,
                              output_shape=reduce_dimension_output_shape,
                              arguments={'length': aspect_max_len},
                              mask=aspect_embedding_layer.get_output_mask_at(0),
                              name='aspect_attention')(aspect_attention)       #n*9
    aspect_embedding_x = Lambda(attention, output_shape=attention_output_shape,arguments={'dim': 300})([aspect_attention, aspect_embedding])  #n*600
    aspect_embedding = Lambda(liter,
                              output_shape=liter_output_shape,
                              arguments={'length': max_len})(aspect_embedding_x)  # n*82*300
    temp = keras.layers.concatenate([input_embedding,aspect_embedding])  #n*82*650
    xcapsule = CapsuleLayer(num_capsule=32, dim_capsule=256,routings=3,name='digitcaps')(temp)
    x = MyFlatten()(xcapsule)

    x = Dropout(rate=0.5)(x)
    #x = keras.layers.multiply([aspect_embedding_x, x]) #n*600
    #x = keras.layers.concatenate([aspect_embedding_x, x])   #n*1200

    predictions = Dense(class_num, activation='softmax')(x)  #n*3
    model = Model(inputs=[sentence_input, position_input, aspect_input], outputs=predictions)
    model.compile(loss=['categorical_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])  #error
    print(model.summary())
    return model

def train_model(sentence_inputs=[], position_inputs=[], aspect_input=[], labels=[], model=None):
    model.fit({'sentence_input': sentence_inputs,
               'position_input': position_inputs,
               'aspect_input': aspect_input},
               labels,
               epochs=3,
               batch_size=64,
               verbose=2,
               validation_split=0.1
               )
    return model

def get_predict(sentence_inputs=[], position_inputs=[], aspect_input=[], model=None):
    results = model.predict({'sentence_input': sentence_inputs, 'position_input': position_inputs, 'aspect_input': aspect_input}, batch_size=64, verbose=0)
    return results
