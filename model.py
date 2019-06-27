import keras
from keras.layers import *
import numpy as np
import pandas as pd
pd.set_option('max_colwidth',400)
from keras.models import Model, load_model
from keras import backend as K
from keras.engine import InputSpec, Layer

EMBEDDING_DIM = 300
POSITION_EMBEDDING_DIM = 50
MAX_LEN = 78
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


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x

class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.supports_masking = True
        self.output_dim = nb_head * size_per_head


    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


class myMaxPooling1D(Layer):
    def __init__(self, pool_size=2, strides=None,padding='valid', data_format='channels_last', **kwargs):
        super(myMaxPooling1D, self).__init__(pool_size, strides,padding, data_format, **kwargs)
        self.supports_masking = True
        self.pool_size = pool_size
        self.strides = strides
        self.padding= padding
        self.data_format = data_format

    def _pooling_function(self, inputs):
        output = K.pool2d(inputs, self.pool_size, self.strides,
                          self.padding, self.data_format, pool_mode='max')
        return output

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=4, share_weights=True,activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        self.supports_masking = True
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)
    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1] # ?*82*600 = 600
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size  1*600*(32*300)
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,input_dim_capsule,self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)   #none*82*600   1*600*(32*300)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]  #82*600, [0]=none
        input_num_capsule = K.shape(u_vecs)[1]  #[1]=82
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))   #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule] 32*82

        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule] 82*32
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))  #[32*82]
            b = K.permute_dimensions(b, (0, 2, 1))  #[32*82]
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])    #[..*32*[128]] [?*32*82*[128]]=[32*82]
        return outputs
    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def build_model(max_len, aspect_max_len, embedding_matrix=[],position_embedding_matrix=[],class_num=3, num_words=10000):
    MAX_LEN = max_len #82

    sentence_input = Input(shape=(max_len,), dtype='int32', name='sentence_input')  # n*82
    position_input = Input(shape=(max_len,), dtype='int32', name='position_input')  # n*82
    aspect_input = Input(shape=(aspect_max_len,), dtype='int32', name='aspect_input')  # n*9

    sentence_embedding_layer = Embedding(num_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],input_length=max_len, trainable=False, mask_zero=True)
    sentence_embedding = sentence_embedding_layer(sentence_input)  # n*82*300
    position_embedding = Embedding(max_len * 2, POSITION_EMBEDDING_DIM, weights=[position_embedding_matrix],
                                   input_length=max_len, trainable=True, mask_zero=True)(position_input)  # n*82*50

    aspect_embedding_layer = Embedding(num_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],input_length=aspect_max_len, trainable=False, mask_zero=True)
    aspect_input_embedding = aspect_embedding_layer(aspect_input)  # n*9*300


    multi_aspect = Attention(3,50)([aspect_input_embedding,aspect_input_embedding,aspect_input_embedding]) #*9*64
    multi_aspect = Dropout(0.5)(multi_aspect)

    multi_aspect_con = GlobalAveragePooling1D()(multi_aspect)

    aspect_embedding = Lambda(liter,
                              output_shape=liter_output_shape,
                              arguments={'length': max_len})(multi_aspect_con)  # n*82*64

    input_embedding = keras.layers.concatenate([sentence_embedding, position_embedding])  # n*82*350
    multi_context = Attention(8,16)([input_embedding,input_embedding,input_embedding])  #n*82*128
    multi_context_con = Dropout(0.5)(multi_context)
    multi_context_con = GlobalAveragePooling1D()(multi_context_con)

    temp = keras.layers.concatenate([multi_context, aspect_embedding])
    capsule = Capsule(num_capsule=78, dim_capsule=150, routings=4, share_weights=True)(temp) #n*82*64
    capsule = Dropout(0.5)(capsule)

    final = Attention(9,16)([capsule,multi_aspect,multi_aspect]) #n*82*72
    final = Dropout(0.5)(final)
    final_con = GlobalAveragePooling1D()(final) #n*144

    final_connect = keras.layers.concatenate([multi_context_con,multi_aspect_con,final_con])
    final_connect = Dropout(0.5)(final_connect)

    predictions = Dense(class_num, activation='softmax')(final_connect)  # n*3

    model = Model(inputs=[sentence_input, position_input, aspect_input], outputs=predictions)
    model.compile(loss=['categorical_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model

def train_model(sentence_inputs=[], position_inputs=[], aspect_input=[], labels=[], model=None):
    model.fit({'sentence_input': sentence_inputs, 'position_input': position_inputs, 'aspect_input': aspect_input}, labels, epochs=2, batch_size=128, verbose=2)
    return model


def get_predict(sentence_inputs=[], position_inputs=[], aspect_input=[], model=None):
    results = model.predict({'sentence_input': sentence_inputs, 'position_input': position_inputs, 'aspect_input': aspect_input}, batch_size=128, verbose=0)
    return results
