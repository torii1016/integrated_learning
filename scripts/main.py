from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

from NN.train import Train
from show_data import ShowData

def main():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    train = Train(mnist.train.images, mnist.train.labels, 0.5, 1000, "save.dump")
    train()

    loss_list, accuracy_list = train.get_learning_curve()

    show_data = ShowData()
    show_data.show_accuracy_curve(accuracy_list, "accuracy_curve.png")
    show_data.show_loss_curve(loss_list, "loss_curve.png")
        
if __name__ == u'__main__':
    main()