import numpy as np

import os
import random
import struct

PATH = os.getcwd()

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
    images = [m.reshape(784, 1) for m in images]
    labels_ = [m.reshape(10, 1) for m in labels_]
    return images, labels_, labels


def s(z):
    return .5 * (1 + np.tanh(.5 * z))
    #return np.arctan(z)

def s_(z):
    return z * (1 - z)
    #return 1 / (1 + z * z)


class FCLayer(object):

    def __init__(self, n, m, f, f_):
        self.n = n
        self.m = m
        self.f = f
        self.f_ = f_
        self.w = np.random.uniform(-1, 1, (m, n))
        self.b = np.zeros((m, 1))
        self.y = np.zeros((m, 1))

    def forward(self, x):
        self.x = x
        self.y = self.f(np.dot(self.w, x) + self.b)

    def backward(self, d):
        self.d = self.f_(self.x) * np.dot(self.w.T, d)
        self.wd = np.dot(d, self.x.T)
        self.bd = d

    def update(self, rate):
        self.w += rate * self.wd
        self.b += rate * self.bd

    def __str__(self):
        s = '%d %d\n' % (self.n, self.m)
        for i in range(self.m):
            for j in range(self.n):
                s += '%f ' % self.w[i][j]
            s += '\n'
        for i in range(self.m): s += '%f ' % self.b[i][0]
        s += '\n'
        return s
        

class Network(object):

    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FCLayer(layers[i], layers[i + 1], s, s_))

    def __str__(self):
        s = ''
        for layer in self.layers:
            s += str(layer)
        return s

    def predict(self, sample):
        y = sample
        for layer in self.layers:
            layer.forward(y)
            y = layer.y
        return y

    def train(self, Y, X, rate, epoch = 1):
        for i in range(epoch):
            for j in range(len(X)):
                self.single_train(Y[j], X[j], rate)
            net.evaluate()

    def single_train(self, y, x, rate):
        self.predict(x)
        self.calc_gradient(y)
        self.update(rate)

    def calc_gradient(self, y):
        d = self.layers[-1].f_(self.layers[-1].y) * (y - self.layers[-1].y)
        for layer in self.layers[::-1]:
            layer.backward(d)
            d = layer.d
        return d

    def update(self, rate):
        for layer in self.layers:
            layer.update(rate)

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

    def evaluate(self):
        correct = 0
        for i in range(len(XX)):
            if np.argmax(self.predict(XX[i])) == int(ZZ[i]):
                correct += 1
        print('RATE: %f' % (correct / len(XX)))

    def dump(self):
        f = open('param.txt', 'w')
        f.write(self.__str__())
        f.close()

    def read(self):
        f = open('param.txt', 'r')
        l = f.readlines()
        f.close()
        now = 0
        for layer in self.layers:
            w = []
            n, m = list(map(eval, l[now].split()))
            for i in range(now + 1, now + m + 1):
                w += list(map(eval, l[i].split()))
            b = list(map(eval, l[now + m + 1].split()))
            layer.w = np.array(w).reshape(m, n)
            layer.b = np.array(b).reshape(m, 1)
            now += m + 2            
        

if __name__ == '__main__':
    net = Network([784, 28, 28, 10])
    X, Y, Z = LoadMnist()
    XX, YY, ZZ = LoadMnist(kind = 't10k')
    #net.train(Y, X, .005, 10)
