# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .tf_util import Layers, lrelu, linear

class _network(Layers):
    def __init__(self, name_scopes, layer_channels):
        assert(len(name_scopes) == 1)
        super().__init__(name_scopes)
        self.layer_channels = layer_channels

    def set_model(self, inputs, is_training = True, reuse = False):

        h  = inputs
        # fully connect
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i, s in enumerate(self.layer_channels):
                lin = linear(i, h, s)
                h = lrelu(lin)

        return tf.nn.softmax(lin)

class NeuralNetwork(object):
    
    def __init__(self, input_dim):
        self.network_layer = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 10]
        self.input_dim = input_dim
        self.network = _network(['NN'], self.network_layer)
        
    def set_model(self, lr):
        
        self.lr = tf.Variable(
            name = "learning_rate",
            initial_value = lr,
            trainable = False)

        self.lr_op = tf.assign(self.lr, 0.95 * self.lr)
        
        # -- place holder ---
        self.input = tf.placeholder(tf.float32, [None, self.input_dim])
        self.target_val = tf.placeholder(tf.float32, [None, 10])

        # -- set network ---
        self.v_s = self.network.set_model(self.input, is_training = True, reuse = False)

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target_val * tf.log(self.v_s), reduction_indices=[1]))
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy, var_list = self.network.get_variables())

        # -- for test --
        self.v_s_wo_train = self.network.set_model(self.input, is_training = False, reuse = True)


    def train(self, sess, input_data, target_val):
        feed_dict = {self.input: input_data,
                     self.target_val: target_val}
        _ = sess.run([self.train_op], feed_dict = feed_dict)
        
        return _

    def get_value(self, sess, input_data):
        v_s = sess.run([self.v_s_wo_train], feed_dict = {self.input: input_data})
        return v_s
    
    def decay_lr(self, sess):
        sess.run(self.lr_op)