# -*- coding:utf-8 -*-

import os
import sys
import math

import numpy as np
import tensorflow as tf

from .model.integrate_neural_network import IntegrateNeuralNetwork

class TrainIntegrate(object):

    def __init__(self, input_data, input_label, lr_rate, episode_num, scope_names, model_names, save_name):

        self.lr_rate = lr_rate

        self.train_data = input_data
        self.train_label = input_label
        self.nn = IntegrateNeuralNetwork(784, 2, [scope_names[0], scope_names[1], scope_names[2], scope_names[3]])
        self.nn.set_model(self.lr_rate)
        self.model_names = model_names
        self.save_name = save_name

        self.episode_num = episode_num
        self.step = self.episode_num/10

        self.accuracy_list = []
        self.loss_list = []
    
    
    def get_learning_curve(self):
        return self.loss_list, self.accuracy_list

    def __call__(self):
        # -- begin training --
        with tf.Session() as sess:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)

            try:
                for i in range(1, self.episode_num+1):

                    choice_id = np.random.choice(self.train_data.shape[0], 100, replace=False)
                    batch_data = self.train_data[choice_id]
                    batch_label = self.train_label[choice_id]

                    _, error = self.nn.train(sess, batch_data, batch_label)
                    self.loss_list.append(error)
                    #print( "episode: {},  error: {}".format(i, error))

                    choice_id = np.random.choice(self.train_data.shape[0], 100, replace=False)
                    val_data = self.train_data[choice_id]
                    val_label = self.train_label[choice_id]

                    accuracy = self.nn.validation(sess, val_data, val_label)
                    self.accuracy_list.append(accuracy)
                    #print( "episode: {},  accuracy: {}".format(i, accuracy))

                saver.save(sess, self.save_name)
        
            except KeyboardInterrupt:
                pass