import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from NN.train import Train
from NN.test import Test
from NN.train_multi import TrainMulti
from NN.train_integrate import TrainIntegrate
#from show_data import ShowData

def train_single(data, label, scope_name, model_name):
    train = Train(data, label, 0.5, 100, scope_name, model_name)
    train()

    #loss_list, accuracy_list = train.get_learning_curve()


def train_multi(datas, labels, scope_names, model_names):
    train_multi = TrainMulti(0.5, 100)
    train_multi.set_dataset_1(datas[0], labels[0], scope_names[0], model_names[0])
    train_multi.set_dataset_2(datas[1], labels[1], scope_names[1], model_names[1])
    train_multi.set_dataset_3(datas[2], labels[2], scope_names[2], model_names[2])
    train_multi()

def train_integrate(data, label, scope_names, model_names):
    train_integrate = TrainIntegrate(data, label, 0.5, 100, scope_names, model_names, "save_4.dump")
    train_integrate()

def test_single(data, label, scope_name, model_name):
    test = Test(data, label, scope_name, model_name)
    test()

def main():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    train_data = np.concatenate([mnist.train.images, mnist.validation.images])
    train_label = np.concatenate([mnist.train.labels, mnist.validation.labels])

    train_data_1, train_data_2, train_data_3, train_data_4 = np.split(train_data, 4)
    train_label_1, train_label_2, train_label_3, train_label_4 = np.split(train_label, 4)
    
    #train_single(train_data_1, train_label_1, "NN_1", "save_1.dump")
    #train_single(train_data_2, train_label_2, "NN_2", "save_2.dump")
    #train_single(train_data_3, train_label_3, "NN_3", "save_3.dump")

    #train_multi([train_data_1, train_data_2, train_data_3], [train_label_1, train_label_2, train_label_3],
    #    ['NN_1', 'NN_2', 'NN_3'], ["save_1.dump", "save_2.dump", "save_3.dump"])
    #train_integrate(train_data_4, train_label_4, ['NN_1', 'NN_2', 'NN_3', 'NN_4'], ["save_1.dump", "save_2.dump", "save_3.dump"])


    #test_single(train_data, train_label, "NN", "save.dump")
    test_single(train_data, train_label, "NN_1", "save_1.dump")
    #test_single(train_data, train_label, "NN_2", "save_2.dump")
    #test_single(train_data, train_label, "NN_3", "save_3.dump")


    #show_data = ShowData()
    #show_data.show_accuracy_curve(accuracy_list, "accuracy_curve.png")
    #show_data.show_loss_curve(loss_list, "loss_curve.png")
        
if __name__ == u'__main__':
    main()