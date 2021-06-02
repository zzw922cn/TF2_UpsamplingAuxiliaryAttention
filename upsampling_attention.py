#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import numpy as np
import tensorflow as tf

class UpsamplingAttention(tf.keras.layers.Layer):

    def __init__(self, input_dim, P=2, dropout=0.):
        """ input_dim: dimension of duration predictor
        """

        super().__init__()
        self.conv1d = ConvBatchNorm(8, 3, dropout, name_idx="upsampling_attention")
        self.dense_layer1 = tf.keras.layers.Dense(16, "swish")
        self.dense_layer2 = tf.keras.layers.Dense(16, "swish")
        self.dense_layer3 = tf.keras.layers.Dense(1)

        self.aux_dense_layer1 = tf.keras.layers.Dense(P, "swish")
        self.aux_dense_layer2 = tf.keras.layers.Dense(P, "swish")

        self.proj_layer = tf.keras.layers.Dense(input_dim, use_bias=False)
        self.M = input_dim
        self.P = P

    def _build(self):
        fake_V = tf.random.uniform(shape=[1, 100, 512], dtype=tf.float32)
        fake_duration_outputs = tf.random.uniform(shape=[1, 100], dtype=tf.float32)
        fake_phone_mask = tf.random.uniform(shape=[1, 100], dtype=tf.float32)
        self(fake_V, fake_duration_outputs, fake_phone_mask)


    def call(self, V, duration_outputs, phone_mask, training=True):
        """ 上采样注意力过程(Upsampling and Auxiliary Attention)

        V: convolutional output of duration predictor
        duration_outputs: predicted value of duration predictor
        """
        N = tf.shape(V)[0]
        K = tf.shape(V)[1]

        durations = tf.nn.relu(tf.math.exp(duration_outputs) - 1.0)
        round_durations = tf.math.maximum(tf.cast(tf.math.round(durations), tf.int32), tf.ones_like(durations, dtype=tf.int32))

        mel_length = tf.cast(tf.reduce_sum(round_durations, axis=-1), tf.int32)
        max_mel_length = tf.cast(tf.math.reduce_max(mel_length), tf.int32)
        mel_mask = tf.sequence_mask(mel_length, max_mel_length, dtype=tf.int32)

        #  phone_mask: [N, K]
        #  mel_mask: [N, T]
        #  attention_mask: [N, T, K]
        attention_mask = tf.matmul(tf.expand_dims(tf.cast(mel_mask, tf.float32), -1), tf.expand_dims(tf.cast(phone_mask, tf.float32), 1))
        bool_attention_mask = tf.cast(attention_mask, tf.bool)

        #  [N, K]
        S = tf.cumsum(durations, exclusive=True, axis=1)

        #  [N, K]
        E = S + durations

        #  [N, T]
        T = tf.cast(tf.tile(tf.meshgrid(tf.range(1, max_mel_length+1)), [N, 1]), tf.float32)

        #  [N, T, K]
        S = tf.tile(tf.expand_dims(T, -1), [1, 1, K]) - tf.transpose(tf.tile(tf.expand_dims(S, -1), [1, 1, max_mel_length]), [0, 2, 1])
        S = S*attention_mask

        #  [N, T, K]
        E = tf.transpose(tf.tile(tf.expand_dims(E, -1), [1, 1, max_mel_length]), [0, 2, 1]) - tf.tile(tf.expand_dims(T, -1), [1, 1, K])
        E = E*attention_mask

        #  [N, K, 8]
        conv_V = self.conv1d(V, training=training)

        #  [N, T, K, 8+1+1]
        W_input = tf.concat([tf.expand_dims(S, -1), tf.expand_dims(E, -1), tf.tile(tf.expand_dims(conv_V, 1), [1, max_mel_length, 1, 1])], axis=-1)

        #  [N, T, K]
        W = tf.nn.softmax(-1000000*attention_mask*tf.squeeze(self.dense_layer3(self.dense_layer2(self.dense_layer1(W_input)))), axis=-1)

        #  [N, T, K, P]
        C = self.aux_dense_layer2(self.aux_dense_layer1(W_input))

        #  [N, T, M]
        proj_C = self.proj_layer(tf.squeeze(tf.reduce_sum(tf.tile(tf.expand_dims(W, -1), [1, 1, 1, self.P])*C, axis=2)))

        # [N, T, M]
        O = tf.linalg.matmul(W, V) + proj_C

        return O, mel_mask


class ConvBatchNorm(tf.keras.layers.Layer):
    """ 卷积+批归一化+Swish+Dropout """

    def __init__(
        self, filters, kernel_size, dropout_rate, name_idx=None
    ):
        super().__init__()
        self.conv1d = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="same",
            name="conv_._{}".format(name_idx),
        )

        self.batch_norm = tf.keras.layers.BatchNormalization(
            name="batch_norm_._{}".format(name_idx)
        )

        self.dropout = tf.keras.layers.Dropout(
            rate=dropout_rate, name="dropout_._{}".format(name_idx)
        )

    def call(self, inputs, training=True):

        outputs = self.conv1d(inputs)
        outputs = self.batch_norm(outputs, training=training)
        outputs = tf.keras.activations.swish(outputs)
        outputs = self.dropout(outputs, training=training)

        return outputs
