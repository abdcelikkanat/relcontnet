import numpy as np
from lib.nhpp import *


class NHPP3NM:

    def __init__(self, x0, v0, a0, maxTime, timeInterval=0.01, seed=0):

        # Initialize the models parameters
        self.__t0 = 0
        self.__x0 = np.asarray(x0, dtype=np.float)
        self.__v0 = np.asarray(v0, dtype=np.float)
        self.__a0 = np.asarray(a0, dtype=np.float)
        self.__maxTime = maxTime
        self.__timeInterval = timeInterval

        # Get the number of nodes and dimension size
        self.__numOfNodes = self.__x0[0].shape[0]
        self.__dim = self.__x0.shape[1]

        # Set the node specific biases
        self.__gamma = 1.25 * np.ones(shape=(self.__numOfNodes, ), dtype=np.float)
        # Set the seed value
        self.__seed = seed

        # Get the node indices
        self._nodePairs = np.triu_indices(n=self.__numOfNodes, k=1)

        # Set the seed value
        np.random.seed(self.__seed)

    def getPositionOf(self, i, t):

        return self.__x0[i, :] + ( self.__v0[i, :] * t ) + ( self.__a0[i, :] * (t ** 2) / 2.0 )

    def getVelocityOf(self, i, t):

        return self.__v0[i, :] * t

    def getAccelerationOf(self, i, t):

        return self.__a0[i, :]

    def __getDistanceBetween(self, i, j, t, order=2):

        xi = self.getPositionOf(i=i, t=t)
        xj = self.getPositionOf(i=j, t=t)

        return sum([(xi[d] - xj[d]) ** order for d in range(self.__dim)]) ** (1. / order)

        # Find the critical points

    def __findCriticalPoints(self, i, j):
        # Euclidean distance assumption
        deltaX = self.__x0[i, :] - self.__x0[j, :]
        deltaV = self.__v0[i, :] - self.__v0[j, :]
        deltaA = self.__a0[i, :] - self.__a0[j, :]

        c0 = np.dot(deltaX, deltaV)
        c1 = np.dot(deltaV, deltaV) + np.dot(deltaX, deltaA)
        c2 = 1.5 * np.dot(deltaV, deltaA)
        c3 = 0.5 * np.dot(deltaA, deltaA)

        criticalPoints = [0]
        if c3 == 0 and c2 == 0:

            tp = -c0 / c1
            if 0 < tp < self.__maxTime:
                criticalPoints.append(tp)
        else:
            raise ValueError("Not implemented.")

        criticalPoints.append(self.__maxTime)

        return criticalPoints

    def __computeIntensityForPair(self, i, j, t):

        return np.exp( self.__gamma[i] + self.__gamma[j] - self.__getDistanceBetween(i=i, j=j, t=t) )

    def constructNetwork(self):

        network = [[[] for _ in range(i, self.__numOfNodes)] for i in range(self.__numOfNodes)]

        for i, j in zip(self._nodePairs[0], self._nodePairs[1]):
            # Define the intensity function for each node pair (i,j)
            intensityFunc = lambda t: self.__computeIntensityForPair(i=i, j=j, t=t)
            # Get the critical points
            criticalPoints = self.__findCriticalPoints(i=i, j=j)
            # Simulate the models
            nhppij = NHPP(maxTime=self.__maxTime, intensityFunc=intensityFunc, timeBins=criticalPoints, seed=self.__seed)
            eventTimes = nhppij.simulate()
            # Add the event times
            network[i][j].extend(eventTimes)

        return network

