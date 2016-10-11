import numpy as np

class Hybrid(object):
    def __init__(self, k, eta_unsupervised, eta_supervised, beta):
        self.n_input = 2
        self.k = k
        self.n_output = 1
        self.eta_unsupervised = eta_unsupervised
        self.eta_supervised = eta_supervised
        self.beta = beta

        self.init_weights()
        fn = lambda p w_j: np.exp(-(np.linalg.norm(p - w_j)**2) / 2)

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

    def g(self, j, pattern):
        numerator = self.fn(pattern, self.weights[j])
        denominator = 0
        for i in range(self.k):
            denominator += self.fn(pattern, self.weights[i])
        return numerator / denominator

    def g_list(self, pattern):
        gs = []
        for i in range(self.k):
            gs.append(self.g(i, pattern))
        return gs
    
    def feed(self, pattern):
        pass

    def train_unsupervised(self, pattern):
        gs = self.g_list(pattern)
        winner = np.argmax(gs)
        # references etc?
        w_winner = self.weights_input[winner]
        d_w = self.eta_unsupervised * (pattern - w_winner)
        self.weights_input[winner] += d_w

    def train_hidden(self, pattern):
        input = self.g_list(pattern)
        b = self.weights_output * pattern
        # continue with gradient descent
    
def read_data(file):
    lines = None
    with open(file, 'r') as f:
        lines = f.readlines()
    split = map(lambda a: a.split(), lines)
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
    targets, inputs = pretty_data(read_data('task3.txt'))
    inputs = normalize(inputs)
    
    eta_unsupervised = 0.02
    eta_supervised = 0.1
    t_unsupervised = 100000
    t_supervised = 3000
    beta = 0.5
     
