import numpy as np
import matplotlib.pyplot as plt
import time


def triangular_distribution():
    p1 = np.array([1.0/2.0, np.sqrt(3)/2.0])
    p1 = p1/np.linalg.norm(p1)
    p2 = np.array([1,0])
    points = 1000#TODO: Remove hardcoded nbr of points
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

    return xs,ys

def swarm_distribution():
    xReference = np.random.rand()
    yReference = np.random.rand()

    points = 100#TODO: Remove hardcoded nbr of points

    swarmX = np.random.uniform(-0.1,0.1,points)
    swarmY = np.random.uniform(-0.1,0.1,points)

    swarmX *= xReference
    swarmY *= yReference

    return swarmX,swarmY

def winning_unit(target, swarm):
    distances = np.linalg.norm(swarm-target,axis=1)
    winnerIndex = np.argmin(distances)

    return winnerIndex

def random_target(data):
    nbrOfDataPoints, _ = data.shape
    r = int(round(np.random.rand()*nbrOfDataPoints))
    target = data[r-1]

    return target

def neighbourhood_function(winner, units, sigma):

    distance = (units-winner)
    return np.exp(-distance**2/(2*sigma**2))

def x():
    # ------------Why the fuck is this not working?--------------
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.relim()
    ax.autoscale_view(True,True,True)

    # some X and Y data
    x = np.arange(10000)
    y = np.random.randn(10000)

    li, = ax.plot(x, y)

    # draw and show it
    fig.canvas.draw()
    plt.show(block=False)

    # loop to update the data
    while True:
        try:
            y[:-10] = y[10:]
            y[-10:] = np.random.randn(10)

            # set the new data
            li.set_ydata(y)

            fig.canvas.draw()

            time.sleep(0.01)
        except KeyboardInterrupt:
            break

if __name__ == '__main__':


    data = np.array(triangular_distribution()).transpose()
    swarm = np.array(swarm_distribution()).transpose()
    swarmIndexes = [x for x in range (0,100)] #TODO: Remove hardcoded nbr of points
    sigmaNord = 100
    sigmaConv = 0.9
    sigma = sigmaNord
    etaNord = 0.1
    etaConv = 0.01
    eta = etaNord
    tao = 200.0
    tOrder = 1000
    tConv = 5*10**4


    #----For super plotting----(not working yet)
    #plotData = data.transpose()
    #plotSwarm = swarm.transpose()

    # draw and show it
    #fig = plt.figure()
    #plt.plot(plotSwarm[0],plotSwarm[1],'.',color='b')

    #ax = fig.add_subplot(111)
    #ax.relim()
    #ax.autoscale_view(True,True,True)

    #li, = ax.plot(plotSwarm[0], plotSwarm[1],'.',color='r')

    #fig.canvas.draw()
    #plt.show(block=False)
    #plt.axis([-0.8,0.8,-0.8,0.8])
    #--------------------------


    for t in range (0,tOrder):

        target = random_target(data)

        winnerIndex = winning_unit(target,swarm)

        winner = swarm[winnerIndex]
        neighbourValues = neighbourhood_function(winnerIndex,swarmIndexes,sigma)
        neighbourValues[winnerIndex] = 1

        directions = (target-swarm)

        delta = eta * np.multiply(neighbourValues, directions.transpose()).transpose()

        swarm += delta

        sigma = sigmaNord*np.exp(-t/tao)
        eta = etaNord*np.exp(-t/tao)

        #---For super plotting----(not working yet)
        #plotSwarm = swarm.transpose()
        #li.set_xdata(plotSwarm[0])

        #li.set_ydata(plotSwarm[1])
        #fig.canvas.draw()

        #time.sleep(0.01)
        #--------------------------

    plotData = data.transpose()
    plotSwarmOrdering = np.array(swarm.transpose())

    sigma = sigmaConv
    eta = etaConv

    for t in range (0,tConv):
        target = random_target(data)

        winnerIndex = winning_unit(target,swarm)

        winner = swarm[winnerIndex]
        neighbourValues = neighbourhood_function(winnerIndex,swarmIndexes,sigma)
        neighbourValues[winnerIndex] = 1

        directions = (target-swarm)

        delta = eta * np.multiply(neighbourValues, directions.transpose()).transpose()

        swarm += delta


    plotSwarmConv = np.array(swarm.transpose())


    plt.figure()
    plt.axis([-0.1,1.1,-0.1,0.95])

    px = np.array([0,0.5,1,0])
    py = np.array([0,np.sqrt(3)/2.0,0,0])

    plt.plot(px,py,linestyle='-',color='b')
    plt.plot(plotSwarmOrdering[0],plotSwarmOrdering[1],linestyle='-',color='r')
    plt.plot(plotSwarmConv[0],plotSwarmConv[1], linestyle = '-', color='g')


    plt.show()