import os
import struct
import numpy as np

PATH = r'C:\Users\wuxic\Desktop\AI\data'

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

def write(x, y):
    now = 1
    for X in x:
        t = X.reshape((28, 28))
        f = open(PATH + r'\images\%d.txt' % now, 'w+')
        for i in range(28):
            for j in range(28):
                f.write(str(t[i][j]) + ' ')
            f.write('\n')
        f.write(str(y[now - 1]))
        now += 1

if __name__ == '__main__':
    x, y = load_mnist()
    write(x, y)
