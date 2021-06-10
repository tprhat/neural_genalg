import time

import numpy as np
import csv
import sys
import operator


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SingleNetwork:
    def __init__(self, layers_sizes):
        self.layers = []
        self.err = None
        for layer_size in layers_sizes:
            layer_weights = np.random.normal(0, 0.01, size=layer_size)
            self.layers.append(layer_weights)

    def calculate_error(self, X, y):
        length = len(y)
        first = True
        for layer in self.layers[:-1]:
            if first:
                predictions = np.dot(X, layer)
                first = False
            else:
                predictions = np.dot(predictions, layer)
            predictions = sigmoid(predictions)
            ones = np.ones((len(predictions), 1))
            predictions = np.append(ones, predictions, axis=1)

        predictions = np.dot(predictions, self.layers[-1])
        predictions = np.reshape(predictions, (1, length))
        y = np.reshape(y, (1, length))

        err = np.square(y - predictions)
        err = np.sum(err) / length

        self.err = err

def selection(network_list):
    max = sum([1 / c.err for c in network_list])
    pick1, pick2 = np.random.uniform(0, max, size=2)
    p1 = -1
    p2 = -1
    current = 0
    for c, chromosome in enumerate(network_list):
        current += 1 / chromosome.err
        if p1 == -1 and current > pick1:
            p1 = c
        if p2 == -1 and current > pick2:
            p2 = c
    return p1, p2


popsize = 0
elitism = 0
p = 0.0
K = 0.0
iters = 0
train = ''
test = ''
network_type = ''

for n, i in enumerate(sys.argv):
    if i == '--popsize':
        popsize = int(sys.argv[n + 1])
    if i == '--iter':
        iters = int(sys.argv[n + 1])
    if i == '--train':
        train = sys.argv[n + 1]
    if i == '--test':
        test = sys.argv[n + 1]
    if i == '--elitism':
        elitism = int(sys.argv[n + 1])
    if i == '--p':
        p = float(sys.argv[n + 1])
    if i == '--K':
        K = float(sys.argv[n + 1])
    if i == '--nn':
        network_type = sys.argv[n + 1]

X = []  # input
y = []  # expected output
with open(train) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        y.append(float(row.pop()))
        floats = [float(x) for x in row]
        X.append(floats)

X_test = []
y_test = []
with open(test) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        y_test.append(float(row.pop()))
        floats = [float(x) for x in row]
        X_test.append(floats)
ones = np.ones((len(X), 1))
X = np.append(ones, X, axis=1)
ones = np.ones((len(X_test), 1))
X_test = np.append(ones, X_test, axis=1)

network = list(map(int, network_type.split('s')[:-1]))
layers = [(len(X[0]), network[0])]

for n in network[:-1]:
    layers.append((n+1, n))

layers.append((network[-1]+1, 1))
nn = SingleNetwork(layers)

current_gen = []
# pocetna generacija
for i in range(popsize):
    nn = SingleNetwork(layers)
    current_gen.append(nn)

# ovdje krecu iteracije
for i in range(1, iters + 1):

    for nn in current_gen:
        nn.calculate_error(X, y)

    current_gen.sort(key=operator.attrgetter('err'))  # sortiranje prema greskama, manja greska bolji roditelj

    if i % 2000 == 0:  # ispis svakih 2000 iteracija
        print(f"[Train error @ {i}]: " + str(current_gen[0].err))
    new_gen = []
    for j in range(elitism):
        new_gen.append(current_gen[j])

    # ovdje krece stvaranje nove populacije
    for j in range(popsize - elitism):
        # selekcija
        p1, p2 = selection(current_gen)
        layers_parent1 = current_gen[p1]
        layers_parent2 = current_gen[p2]
        nn = SingleNetwork(layers)

        # krizanje
        for layer in range(len(layers_parent1.layers)):
            nn.layers[layer] = layers_parent1.layers[layer].copy()
            nn.layers[layer] += layers_parent2.layers[layer]
            nn.layers[layer] /= 2

        for k in range(len(nn.layers)):
            for m in range(len(nn.layers[k])):
                for n in range(len(nn.layers[k][0])):
                    if np.random.uniform() < p:
                        nn.layers[k][m][n] += np.random.normal(scale=K)

        new_gen.append(nn)
    if i == iters:
        for nn in new_gen:
            nn.calculate_error(X, y)
        new_gen.sort(key=operator.attrgetter('err'))
        new_gen[0].calculate_error(X_test, y_test)
        print(f"[Test error]: " + str(new_gen[0].err))

    current_gen = new_gen
