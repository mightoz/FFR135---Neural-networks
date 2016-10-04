import numpy as np
import math
import random
import matplotlib.pyplot as plt

class Kohonen(object):
    # (x,y) for output dimensions, necessary?
    def __init__(self, params):
        self.eta_0 = params['eta_0']
        self.sigma_0 = params['sigma_0']
        self.tau = params['tau']
        self.n_input = params['n_input']
        self.n_output = params['n_output']
            
        self.weights = self.init_weights() 
    
    def init_weights(self):
        """Initializes weights randomly from the distribution"""
        return np.array(triangle_points(self.n_output))
    
    def update_sigma(self, t):
        self.sigma = self.sigma_0 * math.exp(-(t / self.tau))
    
    def update_eta(self, t):
        self.eta = self.eta_0 * math.exp(-(t / self.tau))

    # BEWARE, BUGS BE HERE
    def lambda_fn(self, winner, j):
        """Calculate the neighborhood coefficient for output j"""
        distance_square = np.linalg.norm(j - winner)
        return math.exp(-distance_square / (2 * self.sigma**2))
    
    # OR HERE FOR THAT MATTER
    def lambda_vector(self, winner):
        """Returns a vector of neighborhood coefficients"""
        lambdas = []
        for w in self.weights:
            lambdas.append(self.lambda_fn(winner, w))
        return np.array(lambdas)
    
    def feed(self, pattern):
        """Feed a pattern and update weights"""
        winner = self.winning_neuron_index(pattern)
        #lambdas = self.lambda_vector(self.weights[winner])
        #delta_w = []
        for j in range(self.n_output):
            diff = pattern - self.weights[winner]
            #delta_w = self.eta * lambdas.item(j) * diff
            delta_w = self.eta * diff
            self.weights.put(j, self.weights.item(j) + delta_w)
            
        #self.weights += np.array(delta_w)

    def winning_neuron_index(self, pattern):
        """Returns the index of the winning neuron"""
        min = np.linalg.norm(self.weights[0])
        for i in range(self.n_output):
            if np.linalg.norm(pattern - self.weights[i]) < min:
                min = i
        return min

def triangle_points(n_points):
    x = np.array([1, 0])
    y = np.array([1, np.sqrt(3)])
    y = y / np.linalg.norm(y)
    
    points = []
    for _ in range(n_points):
        a1 = np.random.uniform()
        a2 = np.random.uniform()
        if (a1 + a2 > 1):
            a1 = 1 - a1
            a2 = 1 - a2
        p = a1*x + a2*y    
        points.append(a1*x + a2*y)
    return points

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
    
    net = Kohonen(params)
    plt.scatter(*zip(*net.weights), marker='*', color='b')
    
    for i in range(1, t_order):
        net.update_sigma(i)
        net.update_eta(i)
        net.feed(random.choice(data))
 
    plt.scatter(*zip(*net.weights), marker='*', color='g')
    net.sigma = sigma_conv
    net.eta = eta_conv
    for i in range(1, t_conv):
        net.feed(random.choice(data))  
 
    plt.scatter(*zip(*net.weights), marker='*', color='r')
    #plt.scatter(*zip(*data))
    plt.show()
