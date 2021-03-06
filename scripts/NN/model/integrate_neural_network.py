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

    def set_model_feature(self, inputs, output_layer, reuse = True):
        h  = inputs
        # fully connect
        with tf.variable_scope(self.name_scopes[0], reuse = reuse):
            for i in range(output_layer):
                lin = linear(i, h, self.layer_channels[i])
                h = lrelu(lin)

        return lin

class IntegrateNeuralNetwork(object):
    
    def __init__(self, input_dim, integrate_layper, scope_names):
        #self.network_layer = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 10]
        self.network_layer = [20, 20, 10]
        self.input_dim = input_dim
        self.integrate_layper = integrate_layper
        self.network_1 = _network([scope_names[0]], self.network_layer)
        self.network_2 = _network([scope_names[1]], self.network_layer)
        self.network_3 = _network([scope_names[2]], self.network_layer)
        self.network_4 = _network([scope_names[3]], self.network_layer)
        
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
        self.v_s_1 = self.network_1.set_model_feature(self.input, self.integrate_layper, reuse = False)
        self.v_s_2 = self.network_2.set_model_feature(self.input, self.integrate_layper, reuse = False)
        self.v_s_3 = self.network_3.set_model_feature(self.input, self.integrate_layper, reuse = False)
        self.integrate_1_2 = tf.concat([self.v_s_1, self.v_s_2], 1)
        self.integrated_input = tf.concat([self.integrate_1_2, self.v_s_3], 1)
        self.v_s = self.network_4.set_model(self.integrated_input, is_training = True, reuse = False)

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target_val * tf.log(self.v_s), reduction_indices=[1]))
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cross_entropy, var_list = self.network_4.get_variables())

        # -- for test --
        self.v_s_wo_train = self.network_4.set_model(self.integrated_input, is_training = False, reuse = True)
        self.correct_prediction = tf.equal(tf.argmax(self.v_s_wo_train,1), tf.argmax(self.target_val,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    
    def set_test_model(self):

        self.input = tf.placeholder(tf.float32, [None, self.input_dim])
        self.target_val = tf.placeholder(tf.float32, [None, 10])

        self.v_s_1 = self.network_1.set_model_feature(self.input, self.integrate_layper, reuse = False)
        self.v_s_2 = self.network_2.set_model_feature(self.input, self.integrate_layper, reuse = False)
        self.v_s_3 = self.network_3.set_model_feature(self.input, self.integrate_layper, reuse = False)
        self.integrate_1_2 = tf.concat([self.v_s_1, self.v_s_2], 1)
        self.integrated_input = tf.concat([self.integrate_1_2, self.v_s_3], 1)

        self.v_s_test = self.network_4.set_model(self.integrated_input, is_training = False, reuse = True)
        self.correct_prediction = tf.equal(tf.argmax(self.v_s_test,1), tf.argmax(self.target_val,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


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
    
    def decay_lr(self, sess):
        sess.run(self.lr_op)
