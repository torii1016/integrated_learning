# -*- coding:utf-8 -*-

import os
import sys
import math

import numpy as np
import tensorflow as tf

from .model.neural_network import NeuralNetwork

class TrainMulti(object):

    def __init__(self, lr_rate, episode_num):

        self.lr_rate = lr_rate

        self.episode_num = episode_num
        self.step = self.episode_num/10

        self.accuracy_list = []
        self.loss_list = []
    
    def set_dataset_1(self, input_data, input_label, scope_name, save_name):
        self.train_data_1 = input_data
        self.train_label_1 = input_label
        self.nn_1 = NeuralNetwork(784, scope_name)
        self.nn_1.set_model(self.lr_rate)
        self.save_name_1 = save_name
    
    def set_dataset_2(self, input_data, input_label, scope_name, save_name):
        self.train_data_2 = input_data
        self.train_label_2 = input_label
        self.nn_2 = NeuralNetwork(784, scope_name)
        self.nn_2.set_model(self.lr_rate)
        self.save_name_2 = save_name
    
    def set_dataset_3(self, input_data, input_label, scope_name, save_name):
        self.train_data_3 = input_data
        self.train_label_3 = input_label
        self.nn_3 = NeuralNetwork(784, scope_name)
        self.nn_3.set_model(self.lr_rate)
        self.save_name_3 = save_name
    
    def get_learning_curve(self):
        return self.loss_list, self.accuracy_list

    def train(self, train_data, train_label, nn, save_name):
        # -- begin training --
        with tf.Session() as sess:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            try:
                for i in range(1, self.episode_num+1):

                    choice_id = np.random.choice(train_data.shape[0], 100, replace=False)
                    batch_data = train_data[choice_id]
                    batch_label = train_label[choice_id]

                    _, error = nn.train(sess, batch_data, batch_label)
                    self.loss_list.append(error)
                    #print( "episode: {},  error: {}".format(i, error))

                    choice_id = np.random.choice(train_data.shape[0], 100, replace=False)
                    val_data = train_data[choice_id]
                    val_label = train_label[choice_id]

                    accuracy = nn.validation(sess, val_data, val_label)
                    self.accuracy_list.append(accuracy)
                    #print( "episode: {},  accuracy: {}".format(i, accuracy))

                saver.save(sess, save_name)
        
            except KeyboardInterrupt:
                pass
    
    def __call__(self):
        self.train(self.train_data_1, self.train_label_1, self.nn_1, self.save_name_1)
        self.train(self.train_data_2, self.train_label_2, self.nn_2, self.save_name_2)
        self.train(self.train_data_3, self.train_label_3, self.nn_3, self.save_name_3)