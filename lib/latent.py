import numpy as np
import math
from datasets.synthetic.nhppnm import *


class Disc2Dataset:

    def __init__(self, discretePos, timePoints, gamma, seed = 0):

        # Set the seed value
        self.__seed = seed
        np.random.seed(self.__seed)

        # Set all positions
        self.__discretePos = discretePos

        # Set time steps
        self.__timePoints = timePoints

        # Get the number of time points
        self.__numOfTimePoints = len(self.__timePoints)

        # Initial and max time
        self.__t0 = 0
        self.__maxTime = self.__timePoints[-1]

        # Get the order, the number of nodes and dimension size
        self.__order = self.__discretePos.shape[0]
        self.__numOfNodes = self.__discretePos.shape[1]
        self.__dim = self.__discretePos.shape[2]

        # Set the gamma value
        self.__gamma = gamma * np.ones(shape=(self.__numOfNodes,), dtype=np.float)

        # Get the node indices
        self._nodePairs = np.triu_indices(n=self.__numOfNodes, k=1)

    def __extractCoefficients(self, initTime, initPos, lastTime, lastPos, model="linear"):

        dim = len(initPos)
        if model == "linear":
            c0 = np.zeros(shape=(dim,), dtype=np.float)
            c1 = np.zeros(shape=(dim, ), dtype=np.float)
            for d in range(dim):
                c0[d] = (initPos[d]*lastTime - initTime*lastPos[d]) / (lastTime - initTime)
                c1[d] = (lastPos[d] - initPos[d]) / (lastTime - initTime)

            return c0, c1

    def __findPositionBetween(self, initTime, initPos, lastTime, lastPos, t, model="linear"):

        if t < initTime or t > lastTime:
            raise ValueError("Time cannot be lower than {} or greater than {}".format(initTime, lastTime))

        if model == "linear":

            c0, c1 = self.__extractCoefficients(initTime, initPos, lastTime, lastPos, model="linear")

            return c0 + c1*t

    def getAllPositions(self, t):

        if t < self.__t0 or t > self.__timePoints:
            raise ValueError("Time cannot be lower than {} or greater than {}".format(self.__t0, self.__timePoints[-1]))

        tIndex = 1
        while tIndex < self.__numOfTimePoints:
            if self.__timePoints[tIndex-1] < t < self.__timePoints[tIndex]:
                break
            tIndex = tIndex + 1

        return self.__findPositionBetween(initTime=self.__timePoints[tIndex-1], initPos=self.__discretePos[tIndex-1, :, :],
                                          lastTime=self.__timePoints[tIndex], lastPos=self.__discretePos[tIndex, :, :],
                                          t=t, model="linear")

    def getPositionOf(self, i, t):

        if t < self.__t0 or t > self.__maxTime:
            raise ValueError("Time cannot be lower than {} or greater than {}".format(self.__t0, self.__timePoints[-1]))

        tIndex = 1
        while tIndex < self.__numOfTimePoints:
            if self.__timePoints[tIndex - 1] <= t < self.__timePoints[tIndex]:
                break
            tIndex = tIndex + 1

        if t == self.__timePoints[-1]:
            tIndex = self.__numOfTimePoints - 1

        # print(t, self.__timePoints[-1], self.__maxTime)
        return self.__findPositionBetween(initTime=self.__timePoints[tIndex - 1], initPos=self.__discretePos[tIndex - 1, i, :],
                                          lastTime=self.__timePoints[tIndex], lastPos=self.__discretePos[tIndex, i, :],
                                          t=t, model="linear")

    def getDistanceBetween(self, i, j, t, norm=2):

        xi = self.getPositionOf(i=i, t=t)
        xj = self.getPositionOf(i=j, t=t)

        return sum([(xi[d] - xj[d]) ** norm for d in range(self.__dim)]) ** (1. / norm)

    def computeIntensityForPair(self, i, j, t):

        return np.exp(self.__gamma[i] + self.__gamma[j] - self.getDistanceBetween(i=i, j=j, t=t))

    def __findCriticalPointsBetween(self, i_initPos, i_lastPos, j_initPos, j_lastPos,
                                  initTime, lastTime, model="linear"):

        if model == "linear":
            order = 2
        else:
            raise NotImplementedError

        i_c0, i_c1 = self.__extractCoefficients(initTime, i_initPos, lastTime, i_lastPos, model="linear")
        j_c0, j_c1 = self.__extractCoefficients(initTime, j_initPos, lastTime, j_lastPos, model="linear")

        diff_sum = [np.sum(i_c0 - j_c0), np.sum(i_c1 - j_c1)]

        l1 = [diff_sum[o] / float(math.factorial(o - 1)) for o in range(1, order)] + [0]
        l2 = [diff_sum[o] / float(math.factorial(o)) for o in range(order)]

        c = [0 for _ in range(2 * order - 1)]
        for k in range(2 * order - 1):
            for i in range(min(k, order - 1), max(k - order + 1, 0) - 1, -1):
                c[k] += l1[i] * l2[k - i]

        criticalPoints = []
        for r in sorted(np.roots(c[::-1])):
            if 0 < r < lastTime:
                criticalPoints.append(r)

        return criticalPoints

    # Find the critical points
    def findCriticalPoints(self, i, j, model="linear"):

        criticalPoints = []
        for tInx in range(1, self.__numOfTimePoints):
            initTime, lastTime = self.__timePoints[tInx-1], self.__timePoints[tInx]
            i_initPos, i_lastPos = self.getPositionOf(i=i, t=initTime), self.getPositionOf(i=i, t=initTime)
            j_initPos, j_lastPos = self.getPositionOf(i=j, t=initTime), self.getPositionOf(i=j, t=initTime)
            criticalPoints.append(initTime)
            criticalPoints.extend( self.__findCriticalPointsBetween(i_initPos=i_initPos, i_lastPos=i_lastPos,
                                                                    j_initPos=j_initPos, j_lastPos=j_lastPos,
                                                                    initTime=initTime, lastTime=lastTime, model=model) )
            # criticalPoints.append(lastTime)
        criticalPoints.append(self.__maxTime)

        return criticalPoints

    def constructNetwork(self):

        network = [[[] for _ in range(0, self.__numOfNodes)] for i in range(self.__numOfNodes)]

        count = 0
        for i, j in zip(self._nodePairs[0], self._nodePairs[1]):
            count += 1
            if count % 100 == 0:
                print("{}/{}".format(count, len(self._nodePairs[0])))

            # Define the intensity function for each node pair (i,j)
            intensityFunc = lambda t: self.computeIntensityForPair(i=i, j=j, t=t)
            # Get the critical points
            criticalPoints = self.findCriticalPoints(i=i, j=j)
            # Simulate the models
            nhppij = NHPP(maxTime=self.__maxTime, intensityFunc=intensityFunc, timeBins=criticalPoints,
                          seed=self.__seed)
            eventTimes = nhppij.simulate()
            # Add the event times
            network[i][j].extend(eventTimes)

        return network