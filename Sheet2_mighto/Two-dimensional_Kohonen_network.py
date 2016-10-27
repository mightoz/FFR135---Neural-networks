import numpy as np
import matplotlib.pyplot as plt
import time


class kohonen_network():
    sigma_nord = 30
    eta_nord = 0.1
    t_order = 10 ** 3
    sigma_conv = 0.9
    eta_conv = 0.01
    t_conv = 2 * (10 ** 4)
    tao = 300
    num_neurons = 400
    output_sides = 20
    weight_dimensions = 13
    weight_init_range = (-5, 5)

    def __init__(self):
        self.weights = self.initialize_weights()
        print "weights",self.weights
        self.classification_data, self.wine_data = data()

    def initialize_weights(self):
        weight_matrix = []

        for _ in range(0, self.num_neurons):
            weights = np.random.uniform(self.weight_init_range[0], self.weight_init_range[1], 13)
            weight_matrix.append(weights)
        return np.array(weight_matrix)

    def random_target(self):
        nbr_data_points, _ = self.wine_data.shape
        r = int(round(np.random.rand() * nbr_data_points))
        target = self.wine_data[r - 1]
        return target

    def winning_neuron_index(self, target):

        distances = np.linalg.norm(self.weights - target, axis=1)
        winner_index = np.argmin(distances)
        #print "winner_index",winner_index
        return winner_index

    def neighbourhood_fun(self, winner_index, sigma):
        distances = []
        winner_ind = winner_index+1
        x0 = np.mod(winner_ind,self.output_sides)
        y0 = int(np.ceil(winner_ind/self.output_sides))

        for i in range (1,self.num_neurons+1):
            #Determine col
            x1 = np.mod(i,self.output_sides)
            #Determine row
            y1 = int(np.ceil(i/self.output_sides))
            distances.append(np.linalg.norm(np.array([x0,y0])-np.array([x1,y1])))

        distances = np.array(distances)**2

        return np.exp(-(distances / (2 * sigma ** 2)))

    def update_weights(self, eta, sigma, target):
        winner_index = self.winning_neuron_index(target)
        output_array = self.neighbourhood_fun(winner_index, sigma)
        directions = target - self.weights
        delta_weights = eta * np.multiply(output_array, directions.transpose()).transpose()
        self.weights += delta_weights

    def run_ordering_phase(self):
        eta = self.eta_nord
        sigma = self.sigma_nord

        for t in range(0, self.t_order):
            target = self.random_target()
            self.update_weights(eta, sigma, target)
            sigma = self.sigma_nord * np.exp(-t / self.tao)
            eta = self.sigma_nord * np.exp(-t / self.tao)

    def run_convergence_phase(self):
        eta = self.eta_conv
        sigma = self.sigma_conv

        for _ in range(0, self.t_conv):
            target = self.random_target()
            self.update_weights(eta, sigma, target)

    def get_weights(self):

        return np.array(self.weights)

    def classify_wines(self):

        for i in range(0,np.size(self.classification_data),axis=0):
            pattern = self.wine_data[i]
            classification = self.classification_data[i]
            winner = self.winning_neuron_index(pattern)




def data():
    f = file('Wine_Data.txt', 'r')

    classification_data = []
    wine_data = []
    for line in f:
        columns = line.split(",")
        columns = [col.strip() for col in columns]
        data = [float(x) for x in columns]
        classification_data.append(data[0])
        wine_data.append(data[1:])

    classification_data = np.array(classification_data)
    wine_data = np.array(wine_data)
    # Normalize
    wine_data = wine_data - np.mean(wine_data)
    wine_data = wine_data / np.std(wine_data)

    return classification_data, wine_data


if __name__ == "__main__":



    network = kohonen_network()
    weights1 = network.get_weights()
    print "weights1",weights1
    network.run_ordering_phase()
    weights2 = network.get_weights()
    print "weights2",weights2
    network.run_convergence_phase()
    weights3 = network.get_weights()
    print "weights3",weights3
