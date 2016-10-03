import matplotlib.pyplot as plt
import numpy as np
import Functions
import scipy.special as sp


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

            pns = np.append(pns, (p+0.0)/(n+0.0))
            print "k = ",k,"p = ",p, "n = ",n,"pError = ", pError

        clr = '.'
        if (n == 100):
            clr = 'r'
        elif(n == 200):
            clr = 'b'


        lin = np.linspace(0.1,2.5)
        erf = 0.5*(1 - sp.erf(np.sqrt(np.power(2 * lin, -1))))

        lbl = "N = %s"%n
        plt.plot(pns,pErrors, 'o',color=clr, label="N = %s"%n)

    plt.plot(lin,erf,color='k',label="Theoretical curve ${P_{Error}}$")

    plt.xlabel(r"$\alpha = p/N$")
    plt.ylabel("${P_{Error}}$")
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis([0, 2.5, 0, 0.3])

    plt.show()


