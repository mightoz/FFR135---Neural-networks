import numpy as np
import matplotlib.pyplot as plt


def training_set():
    f = file('Training_Set.txt', 'r')
    patterns = []
    for line in f:
        columns = line.split('\t')
        columns = [col.strip() for col in columns]
        pattern = [float(x) for x in columns[0].split()]

        patterns.append(pattern)

    trainingSet = np.array(patterns).transpose()

    # Normalize input data
    xs = trainingSet[0]
    ys = trainingSet[1]

    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)

    trainingSet[0] = xs / np.std(xs)
    trainingSet[1] = ys / np.std(ys)

    # xs = trainingSet[0]
    # ys = trainingSet[1]
    # colors = ['r' if x == 1 else 'b' for x in trainingSet[2]]
    # plt.figure()
    # for i in range (0,xs.size):
    #    plt.plot(xs[i],ys[i],'o',color=colors[i])

    # plt.show()


    return trainingSet


def validation_set():
    f = file('Validation_Set.txt', 'r')
    patterns = []
    for line in f:
        columns = line.split('\t')
        columns = [col.strip() for col in columns]
        pattern = [float(x) for x in columns[0].split()]

        patterns.append(pattern)

    validationSet = np.array(patterns).transpose()

    # Normalize input data
    xs = validationSet[0]
    ys = validationSet[1]

    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)

    validationSet[0] = xs / np.std(xs)
    validationSet[1] = ys / np.std(ys)

    return validationSet


# Random weight set -0.2 to 0.2
def initialize_weights(leftLayerSize, rightLayerSize):
    wij = np.zeros((rightLayerSize, leftLayerSize))

    for i in range(0, rightLayerSize):
        wij[i] = [(np.random.uniform(-0.2, 0.2)) for _ in range(0, leftLayerSize)]

    return wij


def initialize_biases(layerSize):
    biases = [(np.random.uniform(-1, 1)) for _ in range(0, layerSize)]
    biases = np.matrix(biases)
    return biases


# Calculate Bi, given Wji, Zetai and thetaj
def calculate_b(weights, previousLayerInput, biases):
    theta = np.matrix(biases).transpose()
    b = (np.dot(weights, previousLayerInput.transpose())) - theta

    return b


def activation_function(b):
    beta = 0.5
    return np.tanh(beta * b)


def activation_function_prim(b):
    beta = 0.5
    return beta*(1 - np.power(np.tanh(beta*b),2))


def calculate_classification_error(set, weights, biases):
    a = np.matrix([set[0], set[1]]).transpose()

    b = calculate_b(weights, a, biases)

    output = np.sign(activation_function(b))

    error = (1.0 / (2.0 * output.size)) * np.sum(np.abs(set[2] - output))

    return error

def calculate_classification_error_hidden_layer(set, wij, thetai, wi, theta):
    a = np.matrix([set[0], set[1]]).transpose()

    #---------Propagate forwards----------
    bi = calculate_b(wij, a, thetai)
    vi = activation_function(bi)

    b = calculate_b(wi,vi.transpose(),theta)
    output = np.sign(activation_function(b))

    error = (1.0 / (2.0 * output.size)) * np.sum(np.abs(set[2] - output))

    return error

def train_network_no_hidden_layers(trainingSet, validationSet):

    experiments = 100

    trainingResults = []
    validationResults = []

    for i in range(0, experiments):

        weights = initialize_weights(2, 1)
        # print "Weights before learning", weights
        biases = initialize_biases(1)
        # print "Biases before learning",biases

        iterations = 0

        learningRate = 0.01

        classification_error_training_set = []

        classification_error_validation_set = []

        while iterations < 200000:

            r = np.random.randint(0, trainingSet[0].size)

            input = np.matrix((trainingSet[0][r], trainingSet[1][r]))  # .transpose()

            target = trainingSet[2][r]

            bi = calculate_b(weights, input, biases)
            output = activation_function(bi)

            outputError = np.dot(activation_function_prim(bi), (target - output))

            deltaWeights = np.dot((learningRate * outputError), input)
            deltaBiases = -learningRate * outputError

            weights += deltaWeights
            biases += deltaBiases

            if iterations % 100 == 0:
                trainErr = calculate_classification_error(trainingSet, weights, biases)
                validationErr = calculate_classification_error(validationSet, weights, biases)
                classification_error_training_set.append(trainErr)
                classification_error_validation_set.append(validationErr)

            iterations += 1

        classification_error_training_set = np.array(classification_error_training_set)
        classification_error_validation_set = np.array(classification_error_validation_set)

        trainingResults.append(np.min(classification_error_training_set))
        validationResults.append(np.min(classification_error_validation_set))


    trainingResults = np.array(trainingResults)
    validationResults = np.array(validationResults)

    measurements = np.array([x * 1 for x in range(1, experiments + 1)])

    plt.figure()
    plt.plot(measurements, trainingResults, linestyle='-', color='r', label='Training set')
    plt.plot(measurements, validationResults, linestyle='-', color='b', label='Validation set')
    plt.legend()
    plt.xlabel("Experiments")
    plt.ylabel("${C_v}$")
    plt.axis([1, experiments, 0, 1])
    plt.show()

    print "Average error training set", np.mean(trainingResults)
    print "Average error validation set", np.mean(validationResults)

def train_network_one_hidden_layers(trainingSet,validationSet):

    numNeuronsSet = np.array([4*2**(x-1) for x in range(0,5)])

    experiments = 100
    trainingMeans = []
    validationMeans = []
    plt.figure()

    for numNeurons in numNeuronsSet:
        trainingResults = []
        validationResults = []

        for i in range(0, experiments):

            wij = np.matrix(initialize_weights(2, int(numNeurons))) #Wij, weights for middle layer
            thetai = initialize_biases(int(numNeurons)) #Thetai, thresholds for middle layer

            wi = np.matrix(initialize_weights(int(numNeurons),1)) #Wi, weights for output layer
            theta = initialize_biases(1) #Theta, threshold for output layer

            iterations = 0

            learningRate = 0.01

            classification_error_training_set = []

            classification_error_validation_set = []

            while iterations < 200000:

                r = np.random.randint(0, trainingSet[0].size) #Randomly chosen pattern

                xi = np.matrix((trainingSet[0][r], trainingSet[1][r]))  # XIj, input for current pattern
                zetaj = trainingSet[2][r] #Zetaj, target output for current pattern

                #---------------Propagate forwards----------------
                bi = calculate_b(wij,xi,thetai)
                vi = activation_function(bi) #Vi, middle layer output

                b = calculate_b(wi,vi.transpose(),theta)
                o = activation_function(b)#.transpose() #O, network output

                #---------------Propagate backwards---------------
                #------Output layer-------
                delta = np.dot(activation_function_prim(b), (zetaj - o)) #Delta/output error
                deltaWi = np.dot((learningRate * delta), vi.transpose()) #Delta weight for output layer
                wi += deltaWi
                deltaTheta = -learningRate * delta
                theta += deltaTheta

                #------Middle layer-------
                deltai = np.dot((activation_function_prim(bi) * delta).transpose(), vi) #Deltai
                deltaWij = np.dot((learningRate * deltai), xi) #Delta weight for hidden layer
                wij += deltaWij
                deltaThetai = -learningRate * deltai
                theta += deltaThetai


                if iterations % 100 == 0:
                    trainErr = calculate_classification_error_hidden_layer(trainingSet, wij, thetai, wi, theta)
                    validationErr = calculate_classification_error_hidden_layer(validationSet, wij, thetai, wi, theta)
                    classification_error_training_set.append(trainErr)
                    classification_error_validation_set.append(validationErr)

                iterations += 1

            classification_error_training_set = np.array(classification_error_training_set)
            classification_error_validation_set = np.array(classification_error_validation_set)

            trainingResults.append(np.min(classification_error_training_set))
            validationResults.append(np.min(classification_error_validation_set))

        trainingResults = np.mean(np.array(trainingResults))
        validationResults = np.mean(np.array(validationResults))

        trainingMeans.append(trainingResults)
        validationMeans.append(validationResults)

    plt.plot(numNeuronsSet,trainingMeans, '.',linestyle='-',color='b',label='Training set')
    plt.plot(numNeuronsSet,validationMeans, '.',linestyle='-',color='r',label='Validation set')
    plt.title("Average classification error over number of hidden neurons")
    plt.xlabel("Number of neurons, hidden layer")
    plt.ylabel("${C_v}$")
    plt.legend()
        #measurements = np.array([x * 1 for x in range(1, experiments + 1)])

        #clr = '.'
        #if (numNeurons == 2):
        #    clr = 'b'
        #elif(numNeurons == 4):
        #    clr = 'g'
        #elif(numNeurons == 8):
        #    clr = 'r'
        #elif(numNeurons == 16):
        #    clr = 'm'
        #else:
        #    clr = 'k'

        #plt.plot(measurements, trainingResults, linestyle='-', color=clr, label='Training set, %s neurons'%numNeurons)
        #plt.plot(measurements, validationResults, linestyle='--', color=clr, label='Validation set, %s neurons'%numNeurons)
        #plt.legend()
        #plt.xlabel("Experiments")
        #plt.ylabel("${C_v}$")
        #plt.axis([1, experiments, 0, 1])
    plt.show()


if __name__ == '__main__':
    trainingSet = training_set()
    validationSet = validation_set()

    #train_network_no_hidden_layers(trainingSet, validationSet)

    train_network_one_hidden_layers(trainingSet,validationSet)
