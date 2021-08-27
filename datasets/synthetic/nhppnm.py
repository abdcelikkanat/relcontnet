import numpy as np
from lib.nhpp import *
import math


class NHPPNM:

    def __init__(self, gamma, z0, maxTime, timeInterval=0.01, seed=0):

        # Set the seed value
        self.__seed = seed
        np.random.seed(self.__seed)

        # Initial time
        self.__t0 = 0

        # Set the model parameters
        self.__z0 = np.asarray(z0, dtype=np.float)

        # Get the order, the number of nodes and dimension size
        self.__order = self.__z0.shape[0]
        self.__numOfNodes = self.__z0.shape[1]
        self.__dim = self.__z0.shape[2]

        # Set the node specific biases
        self.__gamma = gamma * np.ones(shape=(self.__numOfNodes,), dtype=np.float)

        # Set the max and time interval
        self.__maxTime = maxTime
        self.__timeInterval = timeInterval

        # Get the node indices
        self.__nodePairs = np.triu_indices(n=self.__numOfNodes, k=1)

    def getAllPositions(self, t):

        z = np.zeros(shape=(self.__numOfNodes, self.__dim))

        for i in range(self.__numOfNodes):
            z[i, :] = self.getLatentVariableAt(i=i, orderIndex=0, t=t)

        return z

    def getPositionOf(self, i, t):

        return self.getLatentVariableAt(i=i, orderIndex=0, t=t)

    def getLatentVariableAt(self, i, orderIndex, t):

        z = np.zeros(shape=(self.__dim,), dtype=np.float)
        for o in range(self.__order-orderIndex):
            z += self.__z0[orderIndex+o, i, :] * (t ** o) / math.factorial(o)

        return z

    def getDistanceBetween(self, i, j, t, norm=2):

        xi = self.getPositionOf(i=i, t=t)
        xj = self.getPositionOf(i=j, t=t)

        return sum([(xi[d] - xj[d]) ** norm for d in range(self.__dim)]) ** (1. / norm)

    # Find the critical points
    def findCriticalPoints(self, i, j):

        diff_sum = [0.0 for _ in range(self.__order)] #np.zeros(shape=(self.__order, self.__dim), dtype=np.float)
        for o in range(self.__order):
            diff_sum[o] = np.sum(self.__z0[o, i, :] - self.__z0[o, j, :])

        l1 = [diff_sum[o] / float(math.factorial(o-1)) for o in range(1, self.__order)] + [0]
        l2 = [diff_sum[o] / float(math.factorial(o)) for o in range(self.__order)]


        c = [0 for _ in range(2*self.__order-1)]
        for k in range(2*self.__order-1):
            for i in range(min(k, self.__order-1), max(k - self.__order + 1, 0) - 1, -1):
                c[k] += l1[i] * l2[k-i]

        criticalPoints = [0]
        roots = np.roots(c[::-1])

        # print(roots)
        # print(np.iscomplex(roots))
        roots = np.real(roots[~np.iscomplex(roots)])
        for r in sorted(roots):
            if 0 < r < self.__maxTime:
                criticalPoints.append(r)
        criticalPoints.append(self.__maxTime)
        # print("===", criticalPoints)

        return criticalPoints

    # Find the critical points
    def __findCriticalPoints2(self, i, j):
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

    def computeIntensityForPair(self, i, j, t):

        return np.exp( self.__gamma[i] + self.__gamma[j] - self.getDistanceBetween(i=i, j=j, t=t) )

    def constructNetwork(self):

        network = [[[] for _ in range(0, self.__numOfNodes)] for i in range(self.__numOfNodes)]

        for i, j in zip(self.__nodePairs[0], self.__nodePairs[1]):
            # Define the intensity function for each node pair (i,j)
            intensityFunc = lambda t: self.computeIntensityForPair(i=i, j=j, t=t)
            # Get the critical points
            criticalPoints = self.findCriticalPoints(i=i, j=j)
            # Simulate the models
            nhppij = NHPP(maxTime=self.__maxTime, intensityFunc=intensityFunc, timeBins=criticalPoints, seed=self.__seed)
            eventTimes = nhppij.simulate()
            # Add the event times
            network[i][j].extend(eventTimes)

        return network

    def __approxIntensityIntegral(self, i, j, tInit, tLast, numOfSamples=10):

        if tInit >= tLast:
            raise ValueError("The initial time {} cannot be greater than the last time step {}!".format(tInit, tLast))

        value = 0.
        timePoints, stepSize = np.linspace(start=tInit, stop=tLast, num=numOfSamples, endpoint=False, retstep=True)
        for t in timePoints:
            value += self.computeIntensityForPair(i=i, j=j, t=t) * stepSize

        return value

    def __computeLogIntensityForPair(self, i, j, t):

        return self.__gamma[i] + self.__gamma[j] - self.getDistanceBetween(i=i, j=j, t=t)

    def getNLL(self, data, numOfSamples=10):

        # neg_logLikelihood = 0.
        # for tInit, tLast, eventTimes in data:
        #     for i, j in zip(self.__nodePairs[0], self.__nodePairs[1]):
        #         ci = self.__approxIntensityIntegral(i=i, j=j, tInit=tInit, tLast=tLast, numOfSamples=numOfSamples)
        #         neg_logLikelihood += -ci
        #         for e in eventTimes:
        #             neg_logLikelihood += self.__computeLogIntensityForPair(i=i, j=j, t=e)
        #
        # return -neg_logLikelihood

        neg_logLikelihood = 0.
        for tInit, tLast, eventsList in data:
            for i, j in zip(self.__nodePairs[0], self.__nodePairs[1]):
                ci = self.__approxIntensityIntegral(i=i, j=j, tInit=tInit, tLast=tLast, numOfSamples=numOfSamples)
                neg_logLikelihood += -ci

            for ij_list in eventsList:
                ij_pair = ij_list[0]
                ij_events = ij_list[1]

                for e in ij_events:
                    neg_logLikelihood += self.__computeLogIntensityForPair(i=ij_pair[0], j=ij_pair[1], t=e)

        return -neg_logLikelihood


if __name__ == "__main__":

    #z0 = np.asarray([ [[-5, 0], [5, 0]], [[2.5, 0], [-2.5, 0]], [[0, 0], [-0, 0]] ])
    z0 = np.asarray([ [[-10, 0], [10, 0]], [[1.0, 0], [-1.0, 0]], [[4, 0], [-4, 0]] ])
    #z0 = np.asarray([ [[-10, 0], [10, 0]], [[1.0, 0], [1.0, 0]], [[4, 0], [4, 0]] ])

    maxTime = 10
    timeInterval = 0.01
    seed = 0

    print("Shape:", z0.shape)

    np.random.seed(seed)

    nhppnm = NHPPNM(z0=z0, maxTime=maxTime, timeInterval=timeInterval, seed=seed)

    #nhppnm.findCriticalPoints(i=0, j=1)
    network = nhppnm.constructNetwork()
    print(network[0][1])

    # Animation
    timePoints = np.arange(0, 5, 0.1)
    z = np.zeros(shape=(50, z0.shape[1], z0.shape[2]))
    for tIdx, t in enumerate(timePoints):
        z[tIdx, :, :] = nhppnm.getAllPositions(t=t)

    # z = np.ones(shape=(100, z0.shape[1], z0.shape[2]))
    from lib.animation import *

    filePath = '/Users/abdulkadir/workspace/relcontnet/temp/animation.gif'
    Animation(x=z, timePoints=timePoints, filePath=filePath,
              r=1, c=1, figSize=(5, 5), bgColor='white', color='r', markerSize=10, delay=1000)
