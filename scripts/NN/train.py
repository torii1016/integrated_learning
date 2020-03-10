# -*- coding:utf-8 -*-

import os
import sys
import math

import numpy as np
import tensorflow as tf

from .model.neural_network import NeuralNetwork

class Train(object):

    def __init__(self, train_data, train_label, lr_rate, episode_num, scope_name, save_name):
        self.train_data = train_data
        self.train_label = train_label

        self.nn = NeuralNetwork(784, 1, scope_name)
        self.nn.set_model(lr_rate)

        self.episode_num = episode_num
        self.save_name = save_name
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
                    """
                    if i%self.step==0: 
                        bar = ('=' * int(i/self.step) ) + (' ' * (int(self.episode_num/self.step-int(i/self.step)))) 
                        print('\r### begin episode[{0}] {1}% ({2}/{3})'.format(bar, int((i/self.step)/(self.episode_num/self.step)*100), i, self.episode_num), end='') 
                    elif i==1:
                        bar = ('=' * 0 ) + (' ' * (int(self.episode_num/self.step))) 
                        print('\r### begin episode[{0}] {1}% ({2}/{3})'.format(bar, int((i/self.step)/(self.episode_num/self.step)*100), 0, self.episode_num), end='') 
                    """

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