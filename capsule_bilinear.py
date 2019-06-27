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

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
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
                                     # shape=self.kernel_size  1*600*（82*128）
                                     initializer='glorot_uniform',
                                     trainable=True)
            self.U = self.add_weight(name='capsuel_first',
                                     shape=(),
                                     initializers='glorot_uniform',
                                     trainable=True
            )
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs, target):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]  #u_vec即输入，82*600, [0]=none
        input_num_capsule = K.shape(u_vecs)[1]  #[1]=82
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))   #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, :, 0])  # shape = [None, num_capsule, input_num_capsule,dim_capsule]] 32*82*128

        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1 ,3))  # shape = [None, input_num_capsule, num_capsule] 82*32*128
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1,3))  #[32*82]*128
            b = K.permute_dimensions(b, (0, 2, 1,3))  #[32*82]*128

            temp1 = K.batch_dot(target, c, [2, 3])  #none*32*82
            temp2 = K.batch_dot(temp1, u_hat_vecs, [2,2]) #[?*32*[82]]*[?*32*[82]*128=[?*32*128]深层更新参数
            outputs = self.activation(temp2)    #none*32*128]

            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])    #[..*32*[+]] [?*32*82*[128]]=[32*82]
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

EMBEDDING_DIM = 300
POSITION_EMBEDDING_DIM = 50
MAX_LEN = 82
MAXLEN = 32
dropout_p = 0.3
rate_drop_dense = 0.3

def reduce_dimension(x, length, mask):
    res = K.reshape(x, [-1, length])  # (?, 78)
    res = K.softmax(res)
    res = res * K.cast(mask, dtype='float32')  # (?, 78)
    temp = K.sum(res, axis=1, keepdims=True)  # (?, 1)
    temp = K.repeat_elements(temp, rep=length, axis=1)  # (?, 78)
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
    return [shape[0], MAXLEN, shape[1]]

def build_model(max_len, maxlen, aspect_max_len, embedding_matrix=[],position_embedding_matrix=[], class_num=3, num_words=10000):
    MAX_LEN = max_len
    MAXLEN = maxlen

    sentence_input = Input(shape=(max_len,), dtype='int32', name='sentence_input')  # (?, 78)
    position_input = Input(shape=(max_len,), dtype='int32', name='position_input')  # (?, 78)
    aspect_input = Input(shape=(aspect_max_len,), dtype='int32', name='aspect_input')  # (?, 78)

    sentence_embedding_layer = Embedding(num_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                         input_length=max_len, trainable=False, mask_zero=True)
    sentence_embedding = sentence_embedding_layer(sentence_input)  # (?, 78, 300)
    position_embedding = Embedding(max_len * 2, POSITION_EMBEDDING_DIM, weights=[position_embedding_matrix],
                                   input_length=max_len, trainable=True, mask_zero=True)(position_input)  # (?, 78, 50)
    aspect_embedding_layer = Embedding(num_words + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                       input_length=aspect_max_len, trainable=False, mask_zero=True)
    aspect_embedding = aspect_embedding_layer(aspect_input)  # (?, 9, 300)
    input_embedding = keras.layers.concatenate([sentence_embedding, position_embedding])  # (?, 78, 350)

    encode_x = Bidirectional(GRU(300, activation="relu",
                                 return_sequences=True, recurrent_dropout=0.5, dropout=0.5))(input_embedding)  # (?, 82, 600)
    aspect_embedding = Bidirectional(GRU(300, activation="relu", return_sequences=True,
                                         recurrent_dropout=0.5, dropout=0.5))(aspect_embedding)   ##(?,9,600)

    aspect_attention = TimeDistributed(Dense(1, activation='tanh'))(aspect_embedding)  # (?, 9, 1) .TimeDistributed(layer)
    aspect_attention = Lambda(reduce_dimension,
                              output_shape=reduce_dimension_output_shape,
                              arguments={'length': aspect_max_len},
                              mask=aspect_embedding_layer.get_output_mask_at(0),
                              name='aspect_attention')(aspect_attention)       # (?, 9)
    aspect_embedding_x0 = Lambda(attention, arguments={'dim': 128})([aspect_attention, aspect_embedding])  # (?, 128)
    aspect_embedding_x = Lambda(liter,
                              output_shape=liter_output_shape,
                              arguments={'length': maxlen})(aspect_embedding_x0)  # none*32*128

    xcapsule = Capsule(num_capsule=32,dim_capsule=256,routings=3,share_weights=True)([encode_x,aspect_embedding_x])    #？ *32*128

    attention_c = TimeDistributed(Dense(100, activation='tanh'))(xcapsule)   # (?, 82, 300)  32*100
    attention_c = TimeDistributed(Dense(1, activation='tanh'))(attention_c)  # (?, 82, 1)  32*1
    attention_x = Lambda(reduce_dimension,
                         output_shape=reduce_dimension_output_shape,
                        arguments={'length': maxlen},
                        mask=sentence_embedding_layer.get_output_mask_at(0),
                         name='attention_x')(attention_c)  # (?, 82)          none*32
    x = Lambda(attention, output_shape=attention_output_shape, arguments={'dim': 128})([attention_x, xcapsule])  # (?, 128)
    finall_x = Dropout(rate=0.5)(x)
    x = keras.layers.concatenate([aspect_embedding_x, finall_x])   #co-attention

    predictions = Dense(class_num, activation='softmax')(x)  # (?, 3)
    model = Model(inputs=[sentence_input, position_input, aspect_input], outputs=predictions)
    model.compile(loss=['categorical_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model

def train_model(sentence_inputs=[], position_inputs=[], aspect_input=[], labels=[], model=None):
    model.fit({'sentence_input': sentence_inputs, 'position_input': position_inputs, 'aspect_input': aspect_input}, labels,epochs=1, batch_size=64, verbose=2)
    return model

def get_predict(sentence_inputs=[], position_inputs=[], aspect_input=[], model=None):
    results = model.predict({'sentence_input': sentence_inputs, 'position_input': position_inputs, 'aspect_input': aspect_input}, batch_size=64, verbose=0)
    return results
