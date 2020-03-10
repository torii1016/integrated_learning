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

    def set_model(self, inputs, output_layer, is_training = True, reuse = False):

        h  = inputs
        # fully connect
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i in range(output_layer):
                lin = linear(i, h, self.layer_channels[i])
                h = lrelu(lin)
        return tf.nn.softmax(lin), lin


class NeuralNetwork(object):
    
    def __init__(self, input_dim, integrate_layper, scope_name):
        #self.network_layer = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 10]
        self.network_layer = [20, 20, 10]
        self.input_dim = input_dim
        self.integrate_layper = integrate_layper
        self.network = _network([scope_name], self.network_layer)
        
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
        self.v_s, _ = self.network.set_model(self.input, len(self.network_layer), is_training = True, reuse = False)

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target_val * tf.log(self.v_s), reduction_indices=[1]))
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy, var_list = self.network.get_variables())

        # -- for test --
        self.v_s_wo_train, _ = self.network.set_model(self.input, len(self.network_layer), is_training = False, reuse = True)
        self.correct_prediction = tf.equal(tf.argmax(self.v_s_wo_train,1), tf.argmax(self.target_val,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # -- for output feature --
        self.v_s_, self.v_s_feature = self.network.set_model(self.input, self.integrate_layper, is_training = False, reuse = True)
    

    def train(self, sess, input_data, target_val):
        feed_dict = {self.input: input_data,
                     self.target_val: target_val}
        _, error = sess.run([self.train_op, self.cross_entropy], feed_dict = feed_dict)
        return _, error

    def validation(self, sess, input_data, target_val):
        feed_dict = {self.input: input_data,
                     self.target_val: target_val}
        _ = sess.run([self.accuracy], feed_dict = feed_dict)
        return _

    def test(self, sess, input_data, target_val):
        feed_dict = {self.input: input_data,
                     self.target_val: target_val}
        _ = sess.run([self.accuracy], feed_dict = feed_dict)
        return _

    def get_value(self, sess, input_data):
        v_s = sess.run([self.v_s_wo_train], feed_dict = {self.input: input_data})
        return v_s

    def get_feature(self, sess, input_data):
        v_s_feature = sess.run([self.v_s_feature], feed_dict = {self.input: input_data})
        return v_s_feature
    
    def decay_lr(self, sess):
        sess.run(self.lr_op)
