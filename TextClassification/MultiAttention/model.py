'''
@Project:NLPAction

@Author:lincoln

@File:model

@Time:2020-05-07 13:21:17

@Description:
'''

import tensorflow as tf
from tensorflow.keras import layers, Model, activations
import math

class WAN(layers.Layer):
    def __init__(self, SETTINGS):
        super(WAN, self).__init__()
        wan_query_init = tf.random_normal_initializer()
        wan_key_init = tf.random_normal_initializer()
        wan_query_b_init = tf.random_normal_initializer()
        wan_key_b_init = tf.random_normal_initializer()
        wan_weights_b_init = tf.random_normal_initializer()
        self.SETTINGS = SETTINGS
        self.wan_query = tf.Variable(initial_value=wan_query_init(shape=(SETTINGS.EMB_DIM, SETTINGS.EMB_DIM), dtype='float32'),
                               trainable=True)
        self.wan_key = tf.Variable(initial_value=wan_key_init(shape=(SETTINGS.EMB_DIM, SETTINGS.EMB_DIM), dtype="float32"),
                               trainable=True)
        self.wan_query_b = tf.Variable(initial_value=wan_query_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.EMB_DIM), dtype='float32'),
                               trainable=True)
        self.wan_key_b = tf.Variable(initial_value=wan_key_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.EMB_DIM), dtype='float32'),
                                     trainable=True)
        self.wan_weights_b = tf.Variable(initial_value=wan_weights_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.MAX_LEN), dtype='float32'),
                                         trainable=True)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()
    def call(self, V, training):
        query = tf.matmul(V, self.wan_query)+self.wan_query_b
        query = self.bn1(query,training = training)
        query = activations.elu(query)

        key = tf.matmul(V, self.wan_key)+self.wan_key_b
        key = self.bn2(key, training=training)
        key = tf.transpose(key, [0, 2, 1])
        key = activations.elu(key)



        weights = tf.matmul(query, key) + self.wan_weights_b
        weights = self.bn3(weights, training = training)
        #weights = layers.ELU(weights)
        weights = tf.nn.softmax(weights,axis=1)

        m1 = tf.matmul(weights,V)
        m1 = self.bn4(m1, training=training)
        return activations.elu(m1)

class LAN(layers.Layer):
    def __init__(self, SETTINGS):
        super(LAN, self).__init__()
        lan_f_init = tf.random_normal_initializer()
        lan_query_init = tf.random_normal_initializer()
        lan_key_init = tf.random_normal_initializer()
        lan_query_b_init = tf.random_normal_initializer()
        lan_key_b_init = tf.random_normal_initializer()
        lan_weights_b_init = tf.random_normal_initializer()
        lan_f_b_init = tf.random_normal_initializer()

        self.SETTINGS= SETTINGS

        self.f = tf.Variable(initial_value=lan_f_init(shape=(SETTINGS.WINDOW_SIZE*SETTINGS.EMB_DIM,SETTINGS.EMB_DIM), dtype="float32"),
                             trainable=True)
        self.lan_query = tf.Variable(initial_value=lan_query_init(shape=(SETTINGS.EMB_DIM, SETTINGS.EMB_DIM), dtype='float32'),
                                     trainable=True)
        self.lan_key = tf.Variable(initial_value=lan_key_init(shape=(SETTINGS.EMB_DIM, SETTINGS.EMB_DIM), dtype="float32"),
                                   trainable=True)
        self.lan_query_b = tf.Variable(initial_value=lan_query_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.EMB_DIM), dtype='float32'),
                                       trainable=True)
        self.lan_key_b = tf.Variable(initial_value=lan_key_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.EMB_DIM), dtype='float32'),
                                     trainable=True)
        self.lan_weights_b = tf.Variable(initial_value=lan_weights_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.MAX_LEN), dtype='float32'),
                                         trainable=True)
        self.lan_f_b = tf.Variable(initial_value=lan_f_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.EMB_DIM), dtype='float32'),
                                         trainable=True)

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()
        self.bn5 = layers.BatchNormalization()

    def call(self, m1, training):
        d = tf.zeros([self.SETTINGS.BATCH_SIZE, 1, self.SETTINGS.WINDOW_SIZE*self.SETTINGS.EMB_DIM])
        s = math.floor(self.SETTINGS.WINDOW_SIZE/2)
        pad = tf.zeros([self.SETTINGS.BATCH_SIZE, s, self.SETTINGS.EMB_DIM])
        m1 = tf.concat([pad,m1,pad], axis=1)

        for i in range(s, self.SETTINGS.MAX_LEN+s):
            t = tf.reshape(m1[:, i-s:i+s+1], [self.SETTINGS.BATCH_SIZE, 1, self.SETTINGS.WINDOW_SIZE*self.SETTINGS.EMB_DIM])
            d = tf.concat([d, t], axis=1)
        d = d[:, 1:]

        l = tf.matmul(d, self.f)+ self.lan_key_b
        l = self.bn1(l, training = training)
        l = activations.elu(l)

        query_2 = tf.matmul(l, self.lan_query) + self.lan_query_b
        query_2 = self.bn2(query_2, training = training)
        query_2 = activations.elu(query_2)

        key_2 = tf.matmul(l, self.lan_key) + self.lan_key_b
        key_2 = self.bn3(key_2, training=training)
        key_2 = activations.elu(key_2)

        weights = tf.matmul(query_2, tf.transpose(key_2, [0, 2, 1])) + self.lan_weights_b
        weights = self.bn4(weights, training = training)
        #weights = layers.ELU(weights)
        weights = tf.nn.softmax(weights, axis=1)

        l = tf.matmul(weights, l)
        l = self.bn5(l, training=training)
        return activations.elu(l)

class CAN(layers.Layer):
    def __init__(self, SETTINGS):
        super(CAN, self).__init__()
        q_init = tf.random_normal_initializer()
        k_init = tf.random_normal_initializer()
        q_b_init = tf.random_normal_initializer()
        k_b_init = tf.random_normal_initializer()
        v_b_init = tf.random_normal_initializer()
        w_b_init = tf.random_normal_initializer()

        self.q = tf.Variable(initial_value=q_init(shape=(SETTINGS.EMB_DIM, SETTINGS.EMB_DIM),dtype="float32"),
                             trainable=True)
        self.k = tf.Variable(initial_value=k_init(shape=(SETTINGS.EMB_DIM, SETTINGS.EMB_DIM),dtype="float32"),
                             trainable=True)
        self.v1 = tf.Variable(initial_value=k_init(shape=(SETTINGS.EMB_DIM, SETTINGS.EMB_DIM), dtype="float32"),
                             trainable=True)
        self.v2 = tf.Variable(initial_value=k_init(shape=(SETTINGS.EMB_DIM, SETTINGS.EMB_DIM), dtype="float32"),
                             trainable=True)
        self.q_b = tf.Variable(initial_value=q_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.EMB_DIM),dtype="float32"),
                               trainable=True)
        self.k_b = tf.Variable(initial_value=k_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.EMB_DIM),dtype="float32"),
                               trainable=True)
        self.m3_b = tf.Variable(initial_value=v_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.EMB_DIM),dtype="float32"),
                               trainable=True)
        self.weights_b = tf.Variable(initial_value=w_b_init(shape=(SETTINGS.MAX_LEN, SETTINGS.MAX_LEN),dtype="float32"),
                                     trainable=True)

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        self.bn4 = layers.BatchNormalization()

    def call(self, m1, m2, training):
        query = tf.matmul(m1,self.q)+self.q_b
        query = self.bn1(query, training = training)
        query = activations.elu(query)

        key = tf.matmul(m2,self.k)+self.k_b
        key = self.bn2(key, training = training)
        key = activations.elu(key)

        weights = tf.matmul(query, tf.transpose(key,[0,2,1]))+self.weights_b
        weights = self.bn3(weights, training = training)
        #weights = tf.nn.relu(weights)
        weights = tf.nn.softmax(weights,axis=1)

        m3 = tf.matmul(weights, tf.matmul(m2,self.v2) + tf.matmul(weights,tf.matmul(m1,self.v1)) + self.m3_b)
        m3 = self.bn4(m3, training=training)
        return activations.elu(m3)

class MANN(Model):
    def __init__(self, wan, lan, can, EmbeddingMatrix, SETTINGS):
        super(MANN,self).__init__()
        self.SETTINGS = SETTINGS
        self.embedding = layers.Embedding(SETTINGS.VOC_SIZE, SETTINGS.EMB_DIM,
                                input_length=SETTINGS.MAX_LEN,
                                embeddings_initializer=tf.keras.initializers.Constant(value=EmbeddingMatrix))

        self.wan = wan
        self.lan = lan
        self.can = can
        self.dp = tf.keras.layers.Dropout(rate=0.5)
        #self.conv1 = layers.Conv1D(filters=640, kernel_size=6, strides=1, padding="valid", activation="relu")
        self.conv2 = layers.Conv1D(filters=320, kernel_size=5, strides=5, padding="valid")

        self.conv3 = layers.Conv1D(filters=128, kernel_size=5, strides=5, padding="valid")
        #self.conv4 = layers.Conv1D(filters=64, kernel_size=5, strides=5, padding="valid", activation="relu")
        self.pooling = layers.MaxPool1D(pool_size=12, strides=1, padding="valid")
        self.flatten = layers.Flatten()
        #self.Dense1 = layers.Dense(320,activation="relu")
        #self.Dense2 = layers.Dense(128,activation="relu")
        self.Dense3 = layers.Dense(64,activation="elu")
        self.Dense4 = layers.Dense(2,activation="softmax")

        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()

    def call(self, inputs, training):
        x = self.embedding(inputs)

        x1 = self.wan(x, training)
        #x1 = self.dp(x1)
        x2 = self.lan(x1, training)
        #x2 = self.dp(x2)
        x = self.can(x1,x2, training)
        #x = self.dp(x)

        for _ in range(self.SETTINGS.LAYERS):
            # MANN layer
            x1 = self.wan(x, training)
            x2 = self.lan(x1, training)
            x = self.can(x1, x2, training)

        #x = self.pooling(x)
        #x = self.conv1(x)
        x = self.conv2(x)
        x = self.dp(x)
        x = self.bn1(x, training=training)
        x = activations.elu(x)
        #x = self.bn3(x, training=training)
        x = self.conv3(x)
        x = self.dp(x)
        x = self.bn2(x, training=training)
        x = activations.elu(x)
        #x = self.bn4(x, training=training)
        #x = self.conv4(x)
        #x = self.bn5(x, training=training)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.dp(x)
        #x = self.Dense1(x)
        #x = self.dp(x)
        #x = self.Dense2(x)
        #x = self.dp(x)
        x = self.Dense3(x)
        x = self.dp(x)
        x = self.bn3(x, training)
        x = activations.elu(x)
        #x = self.dp(x)
        x = self.Dense4(x)
        return x










