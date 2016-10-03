import numpy as np
import matplotlib.pyplot as plt

def triangle_points(n_points):
    x = np.array([1, 0])
    y = np.array([1, np.sqrt(3)])
    y = y / np.linalg.norm(y)
    
    points = []
    for _ in range(n_points):
        a1 = np.random.uniform()
        a2 = np.random.uniform()
        if (a1 + a2 > 1):
            a1 = 1 - a1
            a2 = 1 - a2
        p = a1*x + a2*y    
        points.append(a1*x + a2*y)
    return points

def normalize(point_list):
    mean = np.average(points, axis=0)
    std = np.std(points, axis=0)
    # Normalize standard deviation?
    norm = lambda p: np.divide(p-mean, 1)
    return list(map(norm, point_list))

if __name__ == '__main__':
    points = triangle_points(1000)
    data = normalize(points)
    print('Mean: {}'.format(np.average(data, axis=0)))
    print('Standard deviation: {}'.format(np.std(data, axis=0)))

    x,y = zip(*data)
    plt.scatter(x, y)
    plt.show()
