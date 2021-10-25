import numpy as np

train_X = np.array([1.43, 1.56, 1.59, 1.72, 1.65, 1.71, 1.77, 1.61, 1.64, 1.6])
train_Y = np.array([41, 49, 61, 79, 68, 69, 74, 69, 68, 54])

class Neuron(object):
    
    def __init__(self, n, f):
        self.f = f
        self.w = np.zeros(n)
        self.b = 0

    def __str__(self):
        return 'w = %s\nb = %f\n' % (self.w, self.b)

    def predict(self, x):
        return self.f(np.dot(self.w, x) + self.b)

    def train(self, X, Y, times, rate):
        for i in range(times):
            self.iterate(X, Y, rate)

    def iterate(self, X, Y, rate):
        samples = zip(X, Y)
        for (x, y) in samples:
            t = self.predict(x)
            self.w -= rate * x * (t - y)
            self.b -= rate * (t - y)

def NeuronTraining():
    u = Neuron(1, lambda x: x)
    u.train(train_X, train_Y, 1000, 0.1)
    return u

if __name__ == '__main__':
    u = NeuronTraining()
    print(u)
            
