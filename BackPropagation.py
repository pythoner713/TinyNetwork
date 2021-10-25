import numpy as np

import os
import random
import struct
from functools import reduce

PATH = os.getcwd()

def s(x):
    return 1 / (1 + np.exp(-x))

def LoadMnist(path = PATH, kind = 'train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype = np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)
    labels_ = np.array([np.zeros(10) for x in labels])
    for i in range(len(labels)):
        labels_[i][labels[i]] = 1
    return images, labels_

class Neuron(object):

    def __init__(self, layer_id, node_id):
        self.layer_id = layer_id
        self.node_id = node_id
        self.down = []
        self.up = []
        self.y = 0
        self.d = 0

    def set_output(self, y):
        self.y = y

    def add_down_con(self, con):
        self.down.append(con)

    def add_up_con(self, con):
        self.up.append(con)

    def calc_output(self):
        up_w = [con.w for con in self.up]
        up_y = [con.up_node.y for con in self.up]
        y = np.dot(up_w, up_y)
        self.y = s(y)

    def calc_hidden_d(self):
        down_w = [con.w for con in self.down]
        down_D = [con.down_node.d for con in self.down]
        down_d = np.dot(down_D, down_w)
        self.d = self.y * (1 - self.y) * down_d

    def calc_output_d(self, label):
        self.d = self.y * (1 - self.y) * (label - self.y)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        down_str = reduce(lambda res, con: res + '\n\t' + str(con), self.down, '')
        up_str = reduce(lambda res, con: res + '\n\t' + str(con), self.up, '')
        return node_str + '\n\tdownstream:' + down_str + '\n\tupstream:' + up_str 


class ConstNeuron(object):

    def __init__(self, layer_id, node_id):
        self.layer_id = layer_id
        self.node_id = node_id
        self.down = []
        self.y = 1

    def add_down_con(self, con):
        self.down.append(con)

    def calc_hidden_d(self):
        down_w = [con.w for con in self.down]
        down_D = [con.down_node.d for con in self.down]
        down_d = np.dot(down_D, down_w)
        self.d = self.y * (1 - self.y) * down_d


class Layer(object):
    
    def __init__(self, layer_id, neuron_num):
        self.layer_id = layer_id
        self.neurons = []
        for i in range(neuron_num):
            self.neurons.append(Neuron(layer_id, i))
        self.neurons.append(ConstNeuron(layer_id, neuron_num))

    def set_output(self, data):
        for i in range(len(data)):
            self.neurons[i].set_output(data[i])

    def calc_output(self):
        for neuron in self.neurons[:-1]:
            neuron.calc_output()

    def dump(self):
        for neuron in self.neurons:
            print(neuron)


class Connection(object):
    
    def __init__(self, up_node, down_node):
        self.up_node = up_node
        self.down_node = down_node
        self.w = random.uniform(-0.1, 0.1)
        self.gradient = 0

    def calc_gradient(self):
        self.gradient = self.down_node.d * self.up_node.y

    def update_w(self, rate):
        self.calc_gradient()
        self.w += rate * self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index, 
            self.upstream_node.node_index,
            self.downstream_node.layer_index, 
            self.downstream_node.node_index, 
            self.weight)


class Connections(object):

    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for con in self.connections:
            print(con)


class Network(object):
    
    def __init__(self, layers):
        self.connections = Connections()
        self.layers = []
        layer_num = len(layers)
        neuron_num = 0
        for i in range(layer_num):
            self.layers.append(Layer(i, layers[i]))
        for i in range(layer_num - 1):
            connections = [Connection(up_node, down_node)
                           for up_node in self.layers[i].neurons
                           for down_node in self.layers[i + 1].neurons[:-1]]
            for con in connections:
                self.connections.add_connection(con)
                con.down_node.add_up_con(con)
                con.up_node.add_down_con(con)

    def train(self, Y, X, rate, epoch):
        for i in range(epoch):
            for j in range(len(X)):
                self.train_one_sample(Y[j], X[j], rate)

    def train_one_sample(self, y, x, rate):
        self.predict(x)
        self.calc_d(y)
        self.update_w(rate)

    def calc_d(self, y):
        output_neurons = self.layers[-1].neurons
        for i in range(len(y)):
            output_neurons[i].calc_output_d(y[i])
        for layer in self.layers[-2::-1]:
            for neuron in layer.neurons:
                neuron.calc_hidden_d()

    def update_w(self, rate):
        for layer in self.layers[:-1]:
            for neuron in layer.neurons:
                for con in neuron.down:
                    con.update_w(rate)

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return [u.y for u in self.layers[-1].neurons[:-1]]

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def test(self, sample):
        res = self.predict(sample)
        image = sample.reshape((28, 28))
        for i in range(28):
            for j in range(28):
                print(' .*#'[image[i][j] // 64], end = '')
            print()
        for i in range(10):
            print('PROBABILITY of %d: %f' % (i, res[i]))
        print('PREDICTION: %d' % np.argmax(res))


if __name__ == '__main__':
    net = Network([784, 16, 16, 10])
    X, Y = LoadMnist()

