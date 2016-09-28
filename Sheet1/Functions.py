import numpy as np


def store_random_patterns(p,n):
    """

    :param p: number of patterns
    :param n: number of nodes
    :return: vector of patterns
    """

    patterns = np.zeros((p,n))

    for i in range(0, p):

        for j in range(0, n):

            r = np.random.random(1) # Random number 0-1

            #print r
            ni = patterns[i][j]
            if(r < 0.5): # Change state of pixel with probability 1/2
                ni = 1-ni # Ni
            si = 2*ni-1 #Si, we are using Hopfield model
            patterns[i][j] = si


    #print "patterns: ",patterns
    return patterns


def calculate_weight(patterns):

    #print "calculating weights"
    weights = np.zeros((patterns[0].size,patterns[0].size),dtype=np.float) # Weight matrix Wij, initial weights = 0

    for p in patterns:

        pattern = p; #Pattern #i
        pattern.shape = (pattern.size,1) #Make column vector
        weight = pattern.dot(pattern.transpose()) #Zetai * Zetaj^T
        weight = weight*(1.0/pattern.size)
        weights += weight #Add 1/N * Zetai * ZetaJ^T to sum of weights

    np.fill_diagonal(weights,0) #Set Wii = 0

    return weights #Wi

def calculate_pError(k,p,n):

    pError = 0

    for _ in range (0,k):

        patterns = store_random_patterns(p,n)
        weights = calculate_weight(patterns)
        weights.shape = (n,n)

        errSum = 0.0

        for pattern in patterns:

                si = np.matrix(pattern) #Si
                si.shape = (n,1)

                sx = weights*si
                sx = np.sign(sx) #Si+1

                pattern.shape = (n,1)

                arr = si-sx

                if any(arr):
                    errSum += 1.0

        pError += errSum/p

    pError = pError/k

    return pError

def update_asynchronously(originalPattern, distortedPattern, weights):

    originalPatternCol = originalPattern
    originalPatternCol.shape =(originalPattern.size,1)

    updatedPattern = np.matrix(distortedPattern)
    updatedPattern.shape = (distortedPattern.size,1)


    epoch = 0
    patternFound = 0.0
    while(epoch <= 10 and patternFound == 0):

        randomPermutations = np.random.permutation(originalPattern.size)

        for permutation in randomPermutations:
            r = permutation
            updatedPattern[r] = np.sign(weights[r]*updatedPattern)

        faultyBits = updatedPattern-originalPatternCol

        if(not(faultyBits.any())):
            patternFound = 1.0
        else:
            epoch += 1

    return patternFound

def distort_patterns(patterns,q):

    distortedPatterns = patterns
    for pattern in distortedPatterns:
        pattern = distort_pattern(pattern,q)

    return distortedPatterns

def distort_pattern(pattern,q):

    nbrOfBits = len(pattern)
    bitsToFlip = int(round(nbrOfBits*q))
    distortedPattern = pattern
    bitFlipMap = np.random.permutation(nbrOfBits)

    for i in range(0,bitsToFlip):
        bitToFlip = bitFlipMap[i]
        distortedPattern[bitToFlip] *= -1

    return distortedPattern



#updatedPattern = np.sign(weights*updatedPattern)
    #
    #faultyBits = updatedPattern-originalPatternCol
    #
    #if(not(faultyBits.any())):
    #    patternFound = 1.0