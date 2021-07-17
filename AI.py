import os
import struct
import numpy as np

N = 50000
PATH = os.getcwd() + '\\data'

def s(x):
    return .5 * (1 + np.tanh(.5 * x))

def s_(x):
    return s(x) * (1 - s(x))

def load_mnist(path = PATH, kind = 'train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype = np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)
    return images, labels

class Network(object):
    
    def __init__(self, sizes):
        self.L = len(sizes)
        self.sizes = sizes[1:]
        self.X, self.Y = load_mnist()
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def forward_propogation(self, a):
        for b, w in zip(self.biases, self.weights):
            a = s(np.dot(w, a) + b)
        return a

    def cost(self, output, y):
        output = output[:]
        output[y] -= 1
        return output

    def back_propogation(self, x, y):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        act = x
        acts = [np.array(x)]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, act) + b
            zs.append(z)
            act = s(z)
            acts.append(act)
        d = self.cost(acts[-1], y) * s_(zs[-1])
        db[-1] = d
        dw[-1] = np.dot(d.reshape(self.sizes[-1], 1), acts[-2].reshape(1, len(acts[-2])))
        for i in range(2, self.L):
            z = zs[-i]
            d = np.dot(self.weights[1 - i].transpose(), d) * s_(z)
            db[-i] = d
            dw[-i] = np.dot(d.reshape(len(d), 1), acts[-i - 1].reshape(1, len(acts[-i - 1])))
        return db, dw

    def update(self, mini_batch, r):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            ddb, ddw = self.back_propogation(x, y)
            db = [u + v for u, v in zip(db, ddb)]
            dw = [u + v for u, v in zip(dw, ddw)]
        self.biases = [u - (r / len(mini_batch)) * v for u, v in zip(self.biases, db)]
        self.weights = [u - (r / len(mini_batch)) * v for u, v in zip(self.weights, dw)]

    def test(self, n):
        res = self.forward_propogation(self.X[n])
        image = self.X[n].reshape((28, 28))
        for i in range(28):
            for j in range(28):
                print(' .*#'[image[i][j] // 64], end = '')
            print()
        for i in range(10):
            print('PROBABILITY of %d: %f' % (i, res[i]))
        print('PREDICTION: %d' % np.argmax(res))
        print('ANSWER: %d' % self.Y[n])

    def evaluate(self):
        cnt = 0
        tot = 1000
        idx = np.random.permutation(np.arange(len(self.X)))
        self.X = self.X[idx]
        self.Y = self.Y[idx]
        for i in range(tot):
            res = self.forward_propogation(self.X[i])
            cnt += int(np.argmax(res) == self.Y[i])
        print('ACCURACY:', cnt / tot)

    def learn(self, period, size, rate):
        for now in range(period):
            idx = np.random.permutation(np.arange(len(self.X)))
            self.X = self.X[idx]
            self.Y = self.Y[idx]
            T = [[[u, v] for u, v in zip(self.X[k: k + size], self.Y[k: k + size])] for k in range(0, N, size)]
            for t in T:
                self.update(t, rate)
            self.evaluate()
                
if __name__ == '__main__':
    net = Network([784, 16, 16, 10])
    net.learn(10, 3, 0.05)
