import numpy as np


class NHPP:

    def __init__(self, maxTime, intensityFunc, timeBins=None, seed=0):

        self.__maxTime = maxTime
        self.__intensityFunc = intensityFunc
        self.__timeBins = timeBins
        self.__seed = seed

        # Check if timeBins is set
        if timeBins is None:
            self.__timeBins = self.__getTimeBins()
        else:
            # The first and the last time must be 0 and maxTime
            if self.__timeBins[0] != 0 or self.__timeBins[-1] != self.__maxTime:
                raise ValueError("Invalid time bins!")

        # Set the seed value
        np.random.seed(self.__seed)
    # Find the time bins
    def __getTimeBins(self):

        timeBins = [0, self.__maxTime]

        return timeBins

    # Compute the intensity function
    def simulate(self):
        # Implementation of the algorithm on page 86
        # Given 0 = t_0 < t_1 < ... < t_k < t_{k+1} = T and \lambda_1, \lambda_2, ... , \lambda_{k+1}
        # such that  \lambda(s) <= \lambda_i for all s if t_{i-1} <= s t_i

        # Number of points
        numOfCriticalPoints = len(self.__timeBins)
        # Find the max lambda values for each interval, add [0] to start the indexing from 1
        lambdaValues = [0] + [max(self.__intensityFunc(t=self.__timeBins[inx - 1]), self.__intensityFunc(t=self.__timeBins[inx])) for inx in range(1, numOfCriticalPoints)]
        # Step 1: Initialize the variables
        S = []
        t, J, I = 0, 1, 0
        # Step 2
        U = np.random.uniform(low=0, high=1.0)
        X = (-1.0 / lambdaValues[J]) * np.log(U)
        while True:
            # Step 3
            if t + X <= self.__timeBins[J]:
                # Step 4
                t = t + X
                # Step 5
                U = np.random.uniform(low=0, high=1.0)
                # Step 6
                if U < self.__intensityFunc(t=t) / lambdaValues[J]:
                    # I = I + 1 # no need for the index of the sample list
                    S.append(t)
                # Step 7: go back to Step 2
                U = np.random.uniform(low=0, high=1.0)
                X = (-1.0 / lambdaValues[J]) * np.log(U)
            # Step 8
            else:
                # Step 8
                if J == numOfCriticalPoints - 1:  # k + 1
                    break
                else:
                    # Step 9
                    X = (X - self.__timeBins[J] + t) * (float(lambdaValues[J]) / lambdaValues[J + 1])
                    t = self.__timeBins[J]
                    J = J + 1

        return S
