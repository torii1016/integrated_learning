# -*- coding:utf-8 -*-

import os
import sys
import math

import numpy as np
import tensorflow as tf

from .model.neural_network import NeuralNetwork

class Test(object):

    def __init__(self, test_data, test_label, scope_name, model_name):
        self.test_data = test_data
        self.test_label = test_label

        self.nn = NeuralNetwork(784, 2, scope_name)
        self.nn.set_model(0.5)

        self.model_name = model_name
        
    
    def __call__(self):

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, self.model_name)

        accuracy = self.nn.test(sess, self.test_data, self.test_label)

        print( "accuracy: {}".format(accuracy) )

        #feature = self.nn.get_feature(sess, [self.test_data[0]])
        #print(feature)