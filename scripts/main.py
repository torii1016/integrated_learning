from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from NN.train import Train

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train = Train(mnist.train.images, mnist.train.labels, 0.5, 10000, "save.dump")
train()