import matplotlib.pyplot as plt
import numpy as np

import Digits
import Functions

if __name__ == '__main__':

    #distortions = [x*0.1 for x in range(10)]

    digits = np.array(Digits.digits())
    weights = Functions.calculate_weight(digits)

    distortions = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    pSuccess0 = np.empty([1,0])
    pSuccess1 = np.empty([1,0])
    pSuccess2 = np.empty([1,0])
    pSuccess3 = np.empty([1,0])
    pSuccess4 = np.empty([1,0])


    plt.figure()

    for distortion in distortions:

        successRates = [0.0,0.0,0.0,0.0,0.0]

        for _ in range (0,100):

            distortedDigits = np.array(Digits.digits())
            distortedDigits = Functions.distort_patterns(distortedDigits,distortion)

            for i in range(0,len(digits)):
                successRates[i] += Functions.update_asynchronously(digits[i], distortedDigits[i], weights)

        successRates = [x*1.0/100.0 for x in successRates]
        pSuccess0 = np.append(pSuccess0,successRates[0])
        pSuccess1 = np.append(pSuccess1,successRates[1])
        pSuccess2 = np.append(pSuccess2,successRates[2])
        pSuccess3 = np.append(pSuccess3,successRates[3])
        pSuccess4 = np.append(pSuccess4,successRates[4])

    plt.plot(distortions,pSuccess0,'.', linestyle='-',color= 'r', label = "Digit 0")
    plt.plot(distortions,pSuccess1,'.', linestyle='-',color= 'g', label = "Digit 1")
    plt.plot(distortions,pSuccess2,'.', linestyle='-',color= 'b', label = "Digit 2")
    plt.plot(distortions,pSuccess3,'.', linestyle='-',color= 'k', label = "Digit 3")
    plt.plot(distortions,pSuccess4,'.', linestyle='-',color= 'm', label = "Digit 4")

    plt.axis([-0.1, 1.1, -0.1, 1.1],aspect=1)
    plt.xlabel('Bit distortion($q$)')
    plt.ylabel('Success rate')
    plt.title('Success rate of pattern recognition over $q$')
    plt.legend()

    plt.show()









        #def test (list):

        #    newList = list
        #    for item in list:
        #        item = -1

        #    return newList
        #g = [2,2,2,2,2]
        #g = test(g)
        #print "g:", g