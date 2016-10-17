import numpy as np
import random
import matplotlib.pyplot as plt

class Hybrid(object):
    def __init__(self, k, eta_unsupervised, eta_supervised, beta):
        self.n_input = 2
        self.k = k
        self.n_output = 1
        self.eta_unsupervised = eta_unsupervised
        self.eta_supervised = eta_supervised
        self.beta = beta

        self.init_weights()

    def init_weights(self):
        # initialize weights to middle layer
        weights_input = []
        for _ in range(self.k):
            vec = [random.uniform(-1, 1) for _ in range(self.n_input)]
            weights_input.append(np.array(vec))  
        self.weights_input = weights_input

        # initialize weights to output layer
        weights_output = []
        for _ in range(self.n_output):
            vec = [random.uniform(-1, 1) for _ in range(self.k)]
            weights_output.append(np.array(vec))
        self.weights_output = weights_output

        # init threshold for output
        self.threshold = random.uniform(-1, 1)

    def g_list(self, pattern):
        differences = pattern - self.weights_input
        distances = np.linalg.norm(differences, axis=1)
        distances = np.square(distances) / -2
        exps = np.exp(distances)
        return exps / np.sum(exps)
    
    def classify(self, pattern):
        hidden_output = self.g_list(pattern)
        output = np.tanh(self.beta * (np.dot(self.weights_output, hidden_output) - self.threshold))
        return np.sign(output)
        
    def train_unsupervised(self, pattern):
        gs = self.g_list(pattern)
        winner = np.argmax(gs)
        # calculate delta w for winner
        d_w = self.eta_unsupervised * (pattern - self.weights_input[winner])
        self.weights_input[winner] += d_w

    def train_supervised(self, pattern, target):
        activation_d = lambda b: self.beta * (1 - np.square(np.tanh(self.beta * b)))
        hidden_output = np.array(self.g_list(pattern))
        b = np.dot(self.weights_output, hidden_output) - self.threshold
        output = np.tanh(self.beta * b) # no threshold, included in b
        error = (target - np.round(output)) *  activation_d(b)
        d_w = self.eta_supervised * error * hidden_output
        
        diff = target - np.round(output)
        d_threshold = -self.eta_supervised * diff * activation_d(b)
        self.weights_output += d_w
        self.threshold += d_threshold
 
def read_data(file):
    lines = None
    with open(file, 'r') as f:
        lines = f.readlines()
    split = map(lambda a: a.split(), lines)
    return list(split)

def pretty_data(raw):
    classifications, data = [], []
    for l in raw:
        classifications.append(int(l[0]))
        data.append(np.array(list(map(float, l[1:]))))
    return classifications, data

def assignment_c():
    strings = read_data('task3.txt')
    random.shuffle(strings) # for validation
    targets, inputs = pretty_data(strings)
    train_inputs, train_targets = inputs[:1400], targets[:1400]
    valid_inputs, valid_targets = inputs[1400:], targets[1400:]
    
    eta_unsupervised = 0.02
    eta_supervised = 0.1
    t_unsupervised = 100000
    t_supervised = 3000
    beta = 0.5
 
    avg_errors = []
    for k in range(1, 21):
        # train the network, 20 independent trials
        configurations = []
        for t in range(20):
            net = Hybrid(k, eta_unsupervised, eta_supervised, beta)
            configurations.append(net)
            for i in range(t_unsupervised):
                net.train_unsupervised(random.choice(inputs))
                
            for i in range(t_supervised):
                r = random.randint(0, 1399)
                net.train_supervised(train_inputs[r], train_targets[r])
        
        # calculate classification errors for all confiugrations
        run_errors = []
        for network in configurations:
            errors = 0
            for input, target in zip(valid_inputs, valid_targets):
                res = network.classify(input)
                if res != target:
                    errors += 1
            run_errors.append(errors)

        avg_error = np.average(run_errors) / len(valid_inputs)
        avg_errors.append((k, avg_error))
 
    # plot the avg error as a function of k
    plt.plot(*zip(*avg_errors), color='m', label='Avg. error as fn of k')
    plt.xlabel('$k$')
    plt.ylabel('Classification error')
    plt.legend()
    plt.show()

def assignment_ab():
    strings = read_data('task3.txt')
    random.shuffle(strings) # for validation
    targets, inputs = pretty_data(strings)
    train_inputs, train_targets = inputs[:1400], targets[:1400]
    valid_inputs, valid_targets = inputs[1400:], targets[1400:]
    plt.scatter(*zip(*inputs), label='Data points')
    
    eta_unsupervised = 0.02
    eta_supervised = 0.1
    t_unsupervised = 100000
    t_supervised = 3000
    beta = 0.5
    k = 5
 
    # train the network, 20 independent trials
    configurations = []
    for t in range(20):
        net = Hybrid(k, eta_unsupervised, eta_supervised, beta)
        configurations.append(net)
        for i in range(t_unsupervised):
            net.train_unsupervised(random.choice(inputs))
            
        for i in range(t_supervised):
            r = random.randint(0, 1399)
            net.train_supervised(train_inputs[r], train_targets[r])
    
    # calculate classification errors for all confiugrations
    run_errors = []
    for network in configurations:
        errors = 0
        for input, target in zip(valid_inputs, valid_targets):
            res = network.classify(input)
            if res != target:
                errors += 1
        run_errors.append(errors)

    # find the best configuration and plot the weights from the unsupervised layer
    best_run = np.argmin(run_errors)
    plt.scatter(*zip(*configurations[best_run].weights_input), 
                s=200, color='r', 
                label='Weights after training')
    avg_error = np.average(run_errors) / len(valid_inputs)
    print('Avg. validation error: {}'.format(avg_error))
    print('Lowest validation error: {}'.format(min(run_errors) / len(valid_inputs)))
    
    # find the decision boundary for this configuration
    points = []
    last_y = -30
    best_network = configurations[best_run]
    for x in np.linspace(-20, 30, 250):
        y = last_y
        classification = best_network.classify((np.array([x,y])))
        while best_network.classify(np.array([x, y])) == classification:
            y += 0.01 * -classification
        last_y = y
        points.append(np.array([x, y]))

    # plot the desicion boundary
    plt.plot(*zip(*points), color='m', label='Decision boundary')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([-15, 25, -10, 15])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    assignment_c()
