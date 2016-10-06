import numpy as np
import math
import random
import matplotlib.pyplot as plt

class Kohonen(object):
    def __init__(self, params):
        self.eta_0 = params['eta_0']
        self.sigma_0 = params['sigma_0']
        self.tau = params['tau']
        self.n_input = params['n_input']
        self.n_output = params['n_output']
            
        self.weights = self.init_weights()
    
    def init_weights(self):
        """Initializes weights randomly"""
        weights = []
        for _ in range(self.n_output):
            x = random.uniform(0,1)
            y = random.uniform(0,1)
            weights.append(np.array([x, y]))
        return np.array(weights)

    def update_sigma(self, t):
        self.sigma = self.sigma_0 * math.exp(-(t / self.tau))
    
    def update_eta(self, t):
        self.eta = self.eta_0 * math.exp(-(t / self.tau))
    
    def lambda_list(self, winner):
        """Returns a list of neighborhood coefficients"""
        indices = np.array(range(self.n_output))
        distances_square = np.square(indices - winner)
        return np.exp(-distances_square / (2 * self.sigma**2))
    
    def feed(self, pattern):
        """Feed a pattern and update weights"""
        winner = self.winning_neuron_index(pattern)
        delta_w = []
        lambdas = self.lambda_list(winner)
        for i in range(self.n_output):
            diff = pattern - self.weights[i]
            delta_w.append(self.eta * lambdas[i] * diff)
        self.weights += np.array(delta_w)

    def winning_neuron_index(self, pattern):
        """Returns the index of the winning neuron"""
        winning_index = 0
        min_length = np.linalg.norm(pattern - self.weights[0])
        for i in range(self.n_output):
            len = np.linalg.norm(pattern - self.weights[i])
            if len < min_length:
                min_length = len
                winning_index = i
        return winning_index

def triangle_points(n_points):
    """Draws points from the uniform, triangular distribution"""
    x = np.array([1, 0])
    y = np.array([1, np.sqrt(3)])
    y = y / np.linalg.norm(y)
    
    points = []
    for _ in range(n_points):
        a1 = np.random.uniform()
        a2 = np.random.uniform()
        # transform points in upper triangle to lower
        if (a1 + a2 > 1):
            a1 = 1 - a1
            a2 = 1 - a2
        p = a1*x + a2*y    
        points.append(a1*x + a2*y)
    return points

def draw_triangle():
    """Draws the equilateral triangle"""
    plt.plot([0, 1], [0, 0], color='b', linestyle='-')
    plt.plot([0, 0.5], [0, np.sqrt(3)/2], color='b', linestyle='-')
    plt.plot([0.5, 1], [np.sqrt(3)/2, 0], color='b', linestyle='-')

if __name__ == '__main__':
    # model parameters
    params = {
        'eta_0': 0.1,
        'sigma_0': 100,
        'tau': 200,
        'n_input': 2,
        'n_output': 100 }
    
    eta_conv = 0.01
    sigma_conv = 0.9
    t_order = 1000
    t_conv = 50000
    
    # generate data points
    data = triangle_points(1000)
    
    for run in range(1,2):
        if run == 1:
            params['sigma_0'] = 5
        net = Kohonen(params)

        # ordering phase
        for i in range(1, t_order):
            net.update_sigma(i)
            net.update_eta(i)
            net.feed(random.choice(data)) 
        # create plot
        plt.figure() 
        plt.plot(*zip(*net.weights), marker='*')
        draw_triangle()
        plt.savefig('1{}_order.png'.format('a' if run == 0 else 'b'), 
                    bbox_inches='tight')
        
        # set parameters for convergence phase
        net.sigma = sigma_conv
        net.eta = eta_conv
        # convergence phase
        for i in range(1, t_conv):
            net.feed(random.choice(data))  
        
        # create plot
        plt.figure()
        plt.plot(*zip(*net.weights), marker='*')
        draw_triangle()
        plt.savefig('1{}_conv.png'.format('a' if run == 0 else 'b'), 
                    bbox_inches='tight')
