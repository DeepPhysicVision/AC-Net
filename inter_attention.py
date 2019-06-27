from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer
import tensorflow as tf

class InteractiveAttention(Layer):
    def __init__(self, return_attend_weight=False, initializer='orthogonal', regularizer=None,
                 constraint=None, **kwargs):
        self.return_attend_weight = return_attend_weight

        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.supports_masking = True
        super(InteractiveAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, asp_text_shape = input_shape

        self.context_w = self.add_weight(shape=(context_shape[-1], context_shape[-1]), initializer=self.initializer,
                                         regularizer=self.regularizer, constraint=self.constraint,
                                         name='{}_context_w'.format(self.name))
        self.context_b = self.add_weight(shape=(context_shape[1],), initializer='zero', regularizer=self.regularizer,
                                         constraint=self.constraint, name='{}_context_b'.format(self.name))

        super(InteractiveAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        if mask is not None:
            context_mask, asp_text_mask = mask
        else:
            context_mask = None
            asp_text_mask = None
        context, asp_text = inputs

        context_avg = K.mean(context, axis=1)
        asp_text_avg = K.mean(asp_text, axis=1)

        # attention over context with aspect_text over three parameter
        a_c = K.tanh(K.batch_dot(asp_text_avg, K.dot(context, self.context_w), axes=[1, 2]) + self.context_b)
        a_c = K.exp(a_c)
        if context_mask is not None:
            a_c *= K.cast(context_mask, K.floatx())
        a_c /= K.cast(K.sum(a_c, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attend_context = K.sum(context * K.expand_dims(a_c), axis=1)

        attend_concat = attend_context

        if self.return_attend_weight:
            return [attend_concat, a_c]
        else:
            return attend_concat

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, asp_text_shape = input_shape
        if self.return_attend_weight:
            return [(context_shape[0], context_shape[-1]+asp_text_shape[-1]), (context_shape[0], context_shape[1]),
                    (asp_text_shape[0], asp_text_shape[1])]
        else:
            return context_shape[0], context_shape[-1]+asp_text_shape[-1]