import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Kohonen_2D(object):
    def __init__(self, params):
        self.eta_0 = params['eta_0']
        self.sigma_0 = params['sigma_0']
        self.tau = params['tau']
        self.n_input = params['n_input']
        self.output_dims = params['output_dims']
        self.n_output = self.output_dims[0] * self.output_dims[1]       

        self.weights = self.init_weights()
    
    def init_weights(self):
        """Initializes weights randomly"""
        weights = []
        for _ in range(self.n_output):
            vec = [random.uniform(0, 1) for _ in range(self.n_input)]
            weights.append(np.array(vec))
        return np.array(weights)

    def update_sigma(self, t):
        self.sigma = self.sigma_0 * math.exp(-(t / self.tau))
    
    def update_eta(self, t):
        self.eta = self.eta_0 * math.exp(-(t / self.tau))
    
    def lambda_list(self, winner):
        """Returns a list of neighborhood coefficients"""
        distances_square = []
        for i in range(self.output_dims[0]):
            for j in range(self.output_dims[1]):
                dist = np.sqrt((i - winner[0])**2+(j - winner[1])**2)
                distances_square.append(np.square(dist))
        return np.exp(-np.array(distances_square) / (2 * self.sigma**2))
   
    def classify(self, pattern):
        return self.winning_neuron(pattern)
 
    def feed(self, pattern):
        """Feed a pattern and update weights"""
        winner = self.winning_neuron(pattern)
        delta_w = []
        lambdas = self.lambda_list(winner)
        for i in range(self.n_output):
            diff = pattern - self.weights[i]
            delta_w.append(self.eta * lambdas[i] * diff)
        self.weights += np.array(delta_w)

    def winning_neuron(self, pattern):
        """Returns the position of the winning neuron"""
        winning_index = 0
        min_length = np.linalg.norm(pattern - self.weights[0])
        for i in range(self.n_output):
            len = np.linalg.norm(pattern - self.weights[i])
            if len < min_length:
                min_length = len
                winning_index = i
        row = winning_index // self.output_dims[0]
        col = winning_index % self.output_dims[1]
        return (row, col)

def read_data(file):
    lines = None
    with open(file, 'r') as f:
        lines = f.readlines()
    split = map(lambda a: a.split(','), lines)
    return split

def pretty_data(raw):
    classifications, data = [], []
    for l in raw:
        classifications.append(int(l[0]))
        data.append(np.array(list(map(float, l[1:]))))
    return classifications, data

def normalize(data):
    mean = np.average(data, axis=0)
    std = np.std(data, axis=0)
    normalized = list(map(lambda x: np.divide((x-mean), std), data))
    return normalized

if __name__ == '__main__':
    targets, inputs = pretty_data(read_data('wine.data.txt'))
    inputs = normalize(inputs)
    
    # model parameters
    params = {
        'eta_0': 0.1,
        'sigma_0': 30,
        'tau': 300,
        'n_input': 13,
        'output_dims': (20, 20) }
    
    eta_conv = 0.01
    sigma_conv = 0.9
    t_order = 1000
    t_conv = 20000
    
    net = Kohonen_2D(params)
    # ordering phase
    for i in range(1, t_order):
        net.update_sigma(i)
        net.update_eta(i)
        net.feed(random.choice(inputs)) 
    
    # convergence phase
    net.sigma = sigma_conv
    net.eta = eta_conv
    for i in range(1, t_conv):
        net.feed(random.choice(inputs))

    # training done, classify input 
    fig = plt.figure()
    get_color = lambda t: ['r', 'g', 'b'][t-1]

    for (t, i) in zip(targets, inputs):
        winner = net.classify(i)
        plt.scatter(winner[0], winner[1], color=get_color(t))
    plt.axis([-1, 20, -1, 20])
    plt.savefig('2_class.png', bbox_inches='tight') 
