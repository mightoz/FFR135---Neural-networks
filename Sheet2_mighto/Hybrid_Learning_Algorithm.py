from __future__ import division
import numpy as np
import random as rand
import matplotlib.pyplot as plt


class Hybrid_Net(object):
    eta_unsupervised = 0.02
    unsupervised_iterations = 100000
    supervised_iterations = 3000
    eta_supervised = 0.1
    beta = 0.5

    def __init__(self, k):
        self.num_neurons = k

        self.dataset = np.array(data_set())
        rows, _ = self.dataset.shape

        training_set_end = int(round(rows * 0.7))

        self.training_set = np.array(self.dataset[:training_set_end, :])

        self.validation_set = np.array(self.dataset[training_set_end:, :])

        self.weights = self.initialize_weights()

        self.output_weights = self.initialize_output_weights()

        self.output_threshold = np.random.uniform(-1, 1)

    def perform_unsupervised_learning(self):
        for _ in range(0, self.unsupervised_iterations):
            self.unsupervised_learning_step()

    def unsupervised_learning_step(self):
        pattern_tmp = self.pick_random_pattern()
        pattern = np.array([pattern_tmp[1], pattern_tmp[2]])

        output = self.calculate_output_radialfun(pattern)

        winner_index = np.argmax(output)
        winner = self.weights[winner_index]

        delta_winner = self.eta_unsupervised * (pattern - winner)
        winner += delta_winner

        self.weights[winner_index] = winner

    def pick_random_pattern(self):

        pattern = rand.choice(self.dataset)

        return pattern

    def pick_random_pattern_train(self):

        pattern = rand.choice(self.training_set)

        return pattern

    def pick_random_pattern_valid(self):

        pattern = rand.choice(self.validation_set)

        return pattern

    def calculate_output_radialfun(self, input_pattern):

        scalars = input_pattern - self.weights
        distances = np.linalg.norm(scalars, axis=1)
        distances = -(distances ** 2) / 2

        numerator = np.exp(distances)
        denominator = np.sum(numerator)

        return numerator / denominator

    def initialize_weights(self):

        weights = []

        for _ in range(0, self.num_neurons):
            weight = np.random.uniform(-1, 1, 2)
            weights.append(weight)

        weights = np.array(weights)

        return weights

    def initialize_output_weights(self):

        weights = []

        for _ in range(0, self.num_neurons):
            weight = np.random.uniform(-1, 1)
            weights.append(weight)

        weights = np.array(weights)

        return weights

    def get_weights(self):
        return np.array(self.weights)

    def perform_supervised_learning(self):

        for _ in range(0, self.supervised_iterations):
            self.supervised_learning_step()

    def supervised_learning_step(self):

        # ---------Propagate forward------------

        pattern = self.pick_random_pattern_train()

        input = np.array([pattern[1], pattern[2]])

        target = pattern[0]

        output_hidden = self.calculate_output_radialfun(input)

        b = self.calculate_b_output_layer(output_hidden)

        output = self.activation_function(b)

        delta = self.activation_function_prim(b) * (target - output)

        # -------Update weights and biases-------

        delta_weights = self.eta_supervised * delta * output_hidden

        delta_threshold = -self.eta_supervised * delta

        self.output_weights += delta_weights
        self.output_threshold += delta_threshold

    def calculate_b_output_layer(self, output_hidden):

        w = np.array(self.output_weights)

        v = np.array(output_hidden).transpose()

        b = (np.dot(w, v) - self.output_threshold)

        return b

    def activation_function(self, b):

        output = np.tanh(self.beta * b)

        return output

    def activation_function_prim(self, b):

        output = self.beta * (1 - np.power(np.tanh(self.beta * b), 2))

        return output

    def calculate_classification_error(self):

        errors = []

        for point in self.validation_set:
            input = np.array([point[1], point[2]])
            target = point[0]

            output_hidden = self.calculate_output_radialfun(input)
            b = self.calculate_b_output_layer(output_hidden)

            output = np.sign(self.activation_function(b))

            error = target - output
            errors.append(error)

        classification_error = np.count_nonzero(errors) / len(errors)

        return classification_error

    def decision_boundary_point(self, x, y):

        input = np.array([x, y])

        output_hidden = self.calculate_output_radialfun(input)
        b = self.calculate_b_output_layer(output_hidden)
        output = self.activation_function(b)

        starting_sign = np.sign(output)
        sign_unchanged = True

        while (sign_unchanged):
            input[1] += 0.01 * (-starting_sign)

            output_hidden = self.calculate_output_radialfun(input)
            b = self.calculate_b_output_layer(output_hidden)
            output_sign = np.sign(self.activation_function(b))

            if output_sign != starting_sign:
                sign_unchanged = False

        return input[1]


def data_set():
    f = file('Task3_Dataset.txt', 'r')
    dataset = []
    for line in f:
        columns = line.split('\t')
        columns = [col.strip() for col in columns]
        data = [float(x) for x in columns[0].split()]
        dataset.append(data)
    return dataset


def assignment_ab(k):
    errors = []

    hybrid_nets = []

    for _ in range(0, 20):
        hb = Hybrid_Net(k)

        hb.perform_unsupervised_learning()

        hb.perform_supervised_learning()

        hybrid_nets.append(hb)

        err = hb.calculate_classification_error()
        errors.append(err)

    errors = np.array(errors)

    average_error = np.average(errors)
    lowest_error_index = np.argmin(errors)
    lowest_error = errors[lowest_error_index]

    print"K = ",k
    print"Lowest error = ", lowest_error
    print"Average error = ", average_error

    best_network = hybrid_nets[lowest_error_index]

    print "Best network found"
    ds = np.array(data_set()).transpose()
    xs_input = ds[1]
    ys_input = ds[2]

    weights = best_network.get_weights().transpose()
    xs_weights = weights[0]
    ys_weights = weights[1]

    xs_decision_boundary = np.linspace(-15, 25, num=100)

    ys_decision_boundary = []
    starting_point = -25
    print"Calculating decision boundary"

    for x in xs_decision_boundary:
        y = best_network.decision_boundary_point(x, starting_point)
        starting_point = y
        ys_decision_boundary.append(y)

    ys_decision_boundary = np.array(ys_decision_boundary)

    plt.plot(xs_input, ys_input, 'o', color='#eeefff', label='Input patterns')
    plt.plot(xs_weights, ys_weights, 'o', color='r', label='Weights after unsupervised learning')
    plt.plot(xs_decision_boundary, ys_decision_boundary, color='b', linestyle='-', label='Decision boundary')
    plt.axis([-15, 25, -10, 15])
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$y$', fontsize=20)
    plt.legend()
    plt.show()


def assignment_c():
    average_errors = []

    k_values = [x for x in range(1, 21)]

    for k in k_values:

        errors = []

        hybrid_nets = []

        for _ in range(0, 20):
            hb = Hybrid_Net(k)

            hb.perform_unsupervised_learning()

            hb.perform_supervised_learning()

            hybrid_nets.append(hb)

            err = hb.calculate_classification_error()
            errors.append(err)

        errors = np.array(errors)

        average_error = np.average(errors)

        average_errors.append(average_error)

    average_errors = np.array(average_errors)

    plt.plot(k_values, average_errors, linestyle='-', color='r', label='Average classification error')
    plt.legend()
    plt.xlabel('$k$', fontsize=20)
    plt.ylabel('${P_{error}}$', fontsize=20)
    plt.axis([0, 21, -0.1, 1.1])
    plt.show()


if __name__ == '__main__':

    assignment_ab(5)
    assignment_ab(20)
    assignment_c()
