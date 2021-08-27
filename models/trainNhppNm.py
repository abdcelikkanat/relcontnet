import torch
import math
import time
import numpy as np


class TrainNhppNm(torch.nn.Module):

    def __init__(self, trainSet, testSet, numOfepochs, lr, numOfNodes, dim, order, numOfSamples=10, timeInterval=0.01, seed=0):
        super().__init__()

        # Set the seed value
        self.__seed = seed
        torch.manual_seed(self.__seed)

        self.__numOfNodes = numOfNodes
        self.__dim = dim
        self.__order = order

        self.__numOfepochs = numOfepochs
        self.__lr = lr

        self.__numOfSamples = numOfSamples

        # Set the models parameters
        self.__gamma = torch.rand(size=(self.__numOfNodes, ))
        #self.__gamma = torch.zeros(size=(self.__numOfNodes, 1))
        self.__z0 = []
        self.__z0_nongrad = [torch.randn(self.__numOfNodes, self.__dim) for _ in range(self.__order)]
        #self.__z0_nongrad = [torch.zeros(self.__numOfNodes, self.__dim) for _ in range(self.__order)]


        self.__timeInterval = timeInterval

        # Get the node indices
        self.__nodePairs = torch.triu_indices(row=self.__numOfNodes, col=self.__numOfNodes, offset=1)

        # self.tn_train = tn_train # last time point on time axis in simul
        # self.tn_test = tn_test
        # self.pdist = nn.PairwiseDistance(p=2) # euclidean
        self.__trainSet = trainSet
        self.__testSet = testSet

        self.__factorials = [float(math.factorial(o)) for o in range(self.__order+1)]

        self.__pdist = torch.nn.PairwiseDistance(p=2,  keepdim=False)

    def initializeModelParams(self, gamma=None, order=None):

        if gamma is not None:
            self.__gamma = torch.nn.Parameter(gamma)
        if order is not None:
            # self.__z0[order] = torch.nn.Parameter(self.__z0[order])
            #print([self.__z0[o] for o in range(self.__order)])
            self.__z0 = torch.nn.ParameterList([self.__z0[o] for o in range(len(self.__z0))] + [torch.nn.Parameter(self.__z0_nongrad[order])])

    def get_gamma(self):

        return self.__gamma

    def getLatentVariableAt(self, i, orderIndex, t):

        z = torch.zeros(size=(self.__dim,), dtype=torch.float)
        for o in range(self.__order - orderIndex):
            if orderIndex + o < len(self.__z0):
                z += self.__z0[orderIndex + o][i, :] * (t ** o) / math.factorial(o)
            else:
                z += self.__z0_nongrad[orderIndex + o][i, :] * (t ** o) / math.factorial(o)

        return z

    def getPositionOf(self, i, t):

        return self.getLatentVariableAt(i=i, orderIndex=0, t=t)

    def getAllPositions(self, t):

        z = torch.zeros(size=(self.__numOfNodes, self.__dim,), dtype=torch.float)
        for o in range(self.__order):
            to = t ** o
            if o < len(self.__z0):
                z += self.__z0[o] * to / self.__factorials[o]
            else:
                z += self.__z0_nongrad[o] * to / self.__factorials[o]

        return z

    def getDistanceBetween(self, i, j, t, norm=2):

        if norm != 2:
            raise ValueError("Invalid norm value!")

        xi = self.getPositionOf(i=i, t=t)
        xj = self.getPositionOf(i=j, t=t)

        #return sum([(xi[d] - xj[d]) ** norm for d in range(self.__dim)]) ** (1. / norm)
        diff = xi - xj
        return torch.sqrt(torch.dot(diff, diff))

    def __approxIntensityIntegral(self, i, j, tInit, tLast):

        if tInit >= tLast:
            raise ValueError("The initial time {} cannot be greater than the last time step {}!".format(tInit, tLast))

        value = 0.
        timePoints, stepSize = np.linspace(start=tInit, stop=tLast, num=self.__numOfSamples, endpoint=False, retstep=True)
        for t in timePoints:
            value += self.computeIntensityForPair(i=i, j=j, t=t) * stepSize
        # timePoints = [tInit, tLast] #torch.linspace(start=tInit, end=tLast, steps=self.__numOfSamples+1)
        # stepSize = timePoints[1] - timePoints[0]
        # for t in timePoints[:-1]:
        #     value += self.computeIntensityForPair(i=i, j=j, t=t) * stepSize

        return value

    def computeIntensityForPair(self, i, j, t):

        return torch.exp( self.__gamma[i] + self.__gamma[j] - self.getDistanceBetween(i=i, j=j, t=t) )

    def __computeLogIntensityForPair(self, i, j, t):

        return self.__gamma[i] + self.__gamma[j] - self.getDistanceBetween(i=i, j=j, t=t)

    def meanNormalization(self):

        for o in range(len(self.__z0)):
            m = torch.mean(self.__z0[o], dim=0)
            for i in range(self.__numOfNodes):
                self.__z0[o].data[i, :] = self.__z0[o].data[i, :] - m

    def getAllDistances(self, t, norm=2):

        if norm != 2:
            raise ValueError("Invalid norm value!")

        x = self.getAllPositions(t=t)

        #return sum([(xi[d] - xj[d]) ** norm for d in range(self.__dim)]) ** (1. / norm)
        # diff = xi - xj

        tt = torch.nn.functional.pdist(x)
        return tt

    def computeSumOfIntensityForAllPairs(self, t):

        # temp = self.getAllDistances(t=t)
        # for i, j in zip(self.__nodePairs):
        #     temp[i, j] = torch.exp(self.__gamma[i] + self.__gamma[j] - temp[i, j])
        #
        # return temp
        temp = torch.exp(-self.getAllDistances(t=t))
        expGamma = torch.exp(self.__gamma)
        # temp = expGamma[1:] * temp
        # temp = expGamma[:-1] * temp.T

        tri = torch.triu(torch.outer(expGamma, expGamma), diagonal=1)
        tri_inx = torch.triu_indices(row=self.__numOfNodes, col=self.__numOfNodes, offset=1)
        # temp = temp * torch.reshape(tri[tri_inx[0], tri_inx[1]], (-1,))
        temp = temp * tri[tri_inx[0], tri_inx[1]]

        temp = torch.sum( temp )

        return temp

    def computeAllNonEventsIntegral(self, tInit, tLast):

        totalValue = 0.
        timePoints, stepSize = np.linspace(start=tInit, stop=tLast, num=self.__numOfSamples,
                                           endpoint=False, retstep=True)
        for t in timePoints:
            totalValue += self.computeSumOfIntensityForAllPairs(t=t) * stepSize

        return totalValue

    def __computeLogIntensitySumForANodePairThroughTime(self, i, j, timeList):

        gamma_sum = self.__gamma[i] + self.__gamma[j]

        dist_coeff = torch.zeros(size=(self.__order, self.__dim), dtype=torch.float)
        for o in range(self.__order):
            if o < len(self.__z0):
                dist_coeff[o, :] = ( self.__z0[o][i, :] - self.__z0[o][j, :] ) / self.__factorials[o]
            else:
                dist_coeff[o, :] = ( self.__z0_nongrad[o][i, :] - self.__z0_nongrad[o][j, :] ) / self.__factorials[o]

        dist_sum = 0.0
        for t in timeList:

            dist_vect = torch.zeros(size=(self.__dim, ), dtype=torch.float)
            dist_vect += dist_coeff[0, :]  # Initial coefficient
            power_of_t = 1.0
            for o in range(1, self.__order):
                power_of_t = power_of_t * t

                # dist_vect += dist_coeff[o, :] * power_of_t
                if o < len(self.__z0):
                    dist_vect += power_of_t * (self.__z0[o][i, :] - self.__z0[o][j, :]) / self.__factorials[o]
                else:
                    dist_vect += power_of_t * (self.__z0_nongrad[o][i, :] - self.__z0_nongrad[o][j, :]) / self.__factorials[o]

            dist_sum += torch.sqrt(torch.dot(dist_vect, dist_vect))

        return (len(timeList) * gamma_sum) - dist_sum



    def forward(self, data):

        # neg_logLikelihood = 0.
        # for tInit, tLast, eventTimes in data:
        #     for i, j in zip(self.__nodePairs[0], self.__nodePairs[1]):
        #         ci = self.__approxIntensityIntegral(i=i, j=j, tInit=tInit, tLast=tLast)
        #         neg_logLikelihood += -ci
        #         for e in eventTimes:
        #             neg_logLikelihood += self.__computeLogIntensityForPair(i=i, j=j, t=e)
        #
        # return -neg_logLikelihood

        neg_logLikelihood = 0.
        for tInit, tLast, eventsList in data:

            t0 = time.time()
            # # Non-events
            # if e < 0:
            #     ci = self.__approxIntensityIntegral(i=i, j=j, tInit=tInit, tLast=tLast)
            #     neg_logLikelihood += -ci
            neg_logLikelihood += -self.computeAllNonEventsIntegral(tInit=tInit, tLast=tLast)
            # print("Non-event: {}".format(time.time() - t0))
            # print(neg_logLikelihood)
            t0 = time.time()
            # Events
            for ij_list in eventsList:
                ij_pair = ij_list[0]
                ij_events = ij_list[1]
                neg_logLikelihood += self.__computeLogIntensitySumForANodePairThroughTime(i=ij_pair[0], j=ij_pair[1],timeList=ij_events)
            # for e in ij_events:
            #     s = self.__computeLogIntensityForPair(i=i, j=j, t=e)
            #     neg_logLikelihood += s

            # print("# of events: {}".format(sum([len(el[1]) for el in eventsList])))
            # print("Event: {}".format(time.time() - t0))

        return -neg_logLikelihood

    def learn(self, type="seq"):

        # List for storing the training and testing set losses
        trainLoss, testLoss = [], []

        # Learns the parameters sequentially
        if type == "seq":

            for inx in list(range(1, self.__order + 1)) + [0]:  # range(order+1):

                print("Index: {}".format(inx))

                if inx == 0:
                    self.initializeModelParams(gamma=torch.rand(size=(self.__numOfNodes, )))
                    # nhppNmmModel.initializeModelParams(gamma=torch.zeros(size=(numOfNodes, 1)))
                else:
                    self.initializeModelParams(order=inx - 1)
                # if inx == 1:
                #     nhppNmmModel.initializeModelParams(x0=torch.rand(size=(numOfNodes, dim)))
                # if inx == 2:
                #     nhppNmmModel.initializeModelParams(v0=torch.rand(size=(numOfNodes, dim)))
                # if inx == 3:
                #     nhppNmmModel.initializeModelParams(a0=torch.rand(size=(numOfNodes, dim)))

                # print(len(list(self.parameters())))
                # print(list(self.parameters()))

                # Define the optimizer
                optimizer = torch.optim.Adam(self.parameters(), lr=self.__lr)

                for epoch in range(self.__numOfepochs):
                    t0 = time.time()
                    # running_loss = 0.
                    # print(f"Batch {idx+1} of {len(training_batches)}")
                    self.train()
                    #optimizer.zero_grad()
                    for param in self.parameters():
                        param.grad = None
                    train_loss = self(data=self.__trainSet)
                    t1 = time.time()
                    # print("Forward Time: {}".format(t1 - t0))
                    train_loss.backward()
                    # print("Backward Time: {}".format(time.time() - t1))
                    optimizer.step()
                    # running_loss += loss.item()
                    trainLoss.append(train_loss.item() / len(self.__trainSet))

                    # print("Time: {}".format(time.time() - t0))
                    self.eval()
                    with torch.no_grad():

                        test_loss = self(data=self.__testSet)
                        testLoss.append(test_loss / len(self.__testSet))

                    if epoch % 50 == 0:
                        print(f"Epoch {epoch + 1} train loss: {train_loss.item() / len(self.__trainSet)}")
                        print(f"Epoch {epoch + 1} test loss: {test_loss / len(self.__testSet)}")
                        # if inx == 0:
                        #     print(list(self.parameters()))

        return trainLoss, testLoss