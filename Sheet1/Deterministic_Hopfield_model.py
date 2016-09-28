import matplotlib.pyplot as plt
import numpy as np
import Functions

if __name__ == '__main__':

    k = 100 #Number of measurements
    ps = np.array([10, 20, 30, 40, 50, 75, 100, 150, 200])
    ns = np.array([100,200])


    vals = np.array([])

    plt.figure()

    for n in ns:

        pErrors = np.array([])
        pns = np.array([])


        for p in ps:
            pError = Functions.calculate_pError(k,p,n)
            pErrors = np.append(pErrors,pError)

            pns = np.append(pns, (p+0.0)/(n+0.0)) #TODO int -> float conversion hack for now.
            print "k = ",k,"p = ",p, "n = ",n,"pError = ", pError

        print "pErrors: ", pErrors
        print "p/n = ", pns

        clr = '.'
        if (n == 100):
            clr = 'r'
        elif(n == 200):
            clr = 'b'

        lbl = "N = %s"%numNeurons
        plt.plot(pErrors,pns, 'o',color=clr, label=n)

    plt.show()


