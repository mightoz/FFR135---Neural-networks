import numpy as np
import matplotlib.pyplot as plt


def triangular_distribution():
    p1 = np.array([1.0/2.0, np.sqrt(3)/2.0])
    p1 = p1/np.linalg.norm(p1)
    p2 = np.array([1,0])
    points = 1000
    xs = []
    ys = []

    for _ in range (0,points):
        a1 = np.random.rand()
        a2 = np.random.rand()
        if(a1+a2 > 1):
            a1 = 1-a1
            a2 = 1-a2
        x, y = a1*p1 + a2*p2
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)
    xs = xs - np.mean(xs)
    ys = ys - np.mean(ys)

    return xs,ys

if __name__ == '__main__':
    xs,ys = triangular_distribution()
    plt.figure()

    plt.plot(xs,ys,'.',color='r')

    plt.show()