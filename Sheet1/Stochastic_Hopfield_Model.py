import Functions
import numpy as np
import matplotlib.pyplot as plt


def stochastic_update_step(weightRow, patternCol, beta):
    b = local_field(weightRow, patternCol)
    gb = noise_function(b, beta)

    r = np.random.random_sample()
    if r < gb:
        return 1
    else:
        return -1


def local_field(weightRow, patternCol):
    z = np.dot(weightRow, patternCol)
    return z[0]


def noise_function(b, beta):
    g = 1.0 / (1+np.exp((-2.0) * beta * b))
    return g


def calculate_m(originalPattern, updatedPattern, n):
    uP = np.matrix(updatedPattern)
    uP.shape = (1,updatedPattern.size)
    x = np.dot(uP,originalPattern)
    m = 1.0 / n * x[0]
    return m[0]


def noisy_dynamics(weights, patterns, beta, numNeurons):
    originalPattern = np.array(patterns[0])
    originalPattern.shape = (originalPattern.size,1)
    updatedPattern = np.array(patterns[0])
    updatedPattern.shape = (updatedPattern.size, 1)
    numEpochs = 20

    epochsToMeasure = int(numEpochs/10)
    epochs = [x for x in range(numEpochs)]

    #iterations = [x for x in range (numEpochs*numNeurons)]#TODO
    #mS = []#TODO

    mAverage = 0.0

    #Run dynamics until m values are somewhat 'stable'
    for _ in epochs:
        randomIndexes = np.random.permutation(numNeurons)
        for index in randomIndexes:
            newBit = stochastic_update_step(weights[index], updatedPattern, beta)
            updatedPattern[index] = newBit
            #mS = np.append(mS, calculate_m(originalPattern, updatedPattern, numNeurons))#TODO

    #Caluclate average over last values
    for _ in range (0,epochsToMeasure):
        randomIndexes = np.random.permutation(numNeurons)
        mAverageEpoch = 0.0

        for index in randomIndexes:
            newBit = stochastic_update_step(weights[index], updatedPattern, beta)
            updatedPattern[index] = newBit
            mAverageEpoch+= calculate_m(originalPattern, updatedPattern, numNeurons)

        mAverageEpoch *= 1.0/numNeurons
        mAverage += mAverageEpoch

    mAverage *= 1.0/(numEpochs*0.1) #Divide by number of epochs for average

    #plt.figure()#TODO
    #plt.plot(iterations, mS, color='b', linestyle='-')#TODO
    #plt.axis([0, numEpochs*numNeurons, -0.1, 1])#TODO
    #plt.show()#TODO
    return mAverage

def evaluate_noisy_dynamics(numNeuronsList):


    beta = 2
    alphaSteps = [x*0.01 for x in range (1,101)]

    plt.figure()

    for numNeurons in numNeuronsList:
        print "numNeurons",numNeurons

        mAverages = []

        for alpha in alphaSteps:
            print "alpha",alpha
            mValues = []

            for i in range (0,20):
                p = int(round(alpha*numNeurons))
                patterns = np.array(Functions.store_random_patterns(p, numNeurons))
                weights = np.array(Functions.calculate_weight(patterns))
                m = noisy_dynamics(weights,patterns,beta,numNeurons)

                mValues.append(m)

            average = np.mean(np.array(mValues))
            mAverages.append(average)

            #mAverages.append()# = np.append(mAverages,m)


        clr = '.'
        if (numNeurons == 50):
            clr = 'm'
        elif(numNeurons == 100):
            clr = 'g'
        elif(numNeurons == 250):
            clr = 'r'
        elif(numNeurons == 500):
            clr = 'b'
        else:
            clr = 'k'

        lbl = "N = %s"%numNeurons

        plt.plot(alphaSteps, mAverages, color=clr, linestyle='-',label=lbl)
        plt.axis([0, 1, -0.1, 1])
        plt.xlabel(r'$\alpha$',fontsize=20)
        plt.ylabel(r'${m_1}$',fontsize=20)

    plt.legend()
    plt.show()


if __name__ == '__main__':

    numNeuronsList = np.array([250,500])#50,100,250,500])

    evaluate_noisy_dynamics(numNeuronsList)


    #patterns = np.array(Functions.store_random_patterns(p, n))#TODO
    #weights = np.array(Functions.calculate_weight(patterns))#TODO
    #noisy_dynamics(weights, patterns, beta, n)#TODO
