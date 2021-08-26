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
        self.__gamma = torch.rand(size=(self.__numOfNodes, 1))
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

    def initializeModelParams(self, gamma=None, order=None):

        if gamma is not None:
            self.__gamma = torch.nn.Parameter(gamma)
        if order is not None:
            # self.__z0[order] = torch.nn.Parameter(self.__z0[order])
            #print([self.__z0[o] for o in range(self.__order)])
            self.__z0 = torch.nn.ParameterList([self.__z0[o] for o in range(len(self.__z0))] + [torch.nn.Parameter(self.__z0_nongrad[order])])

    def get_gamma(self):

        return self.__gamma

    def getPositionOf(self, i, t):

        return self.getLatentVariableAt(i=i, orderIndex=0, t=t)

    def getAllPositions(self, t):

        z = torch.zeros(size=(self.__numOfNodes, self.__dim))

        for i in range(self.__numOfNodes):
            z[i, :] = self.getLatentVariableAt(i=i, orderIndex=0, t=t)

        return z

    def getLatentVariableAt(self, i, orderIndex, t):

        z = torch.zeros(size=(self.__dim,), dtype=torch.float)
        for o in range(self.__order - orderIndex):
            if orderIndex + o < len(self.__z0):
                z += self.__z0[orderIndex + o][i, :] * (t ** o) / math.factorial(o)
            else:
                z += self.__z0_nongrad[orderIndex + o][i, :] * (t ** o) / math.factorial(o)

        return z

    def getDistanceBetween(self, i, j, t, norm=2):

        xi = self.getPositionOf(i=i, t=t)
        xj = self.getPositionOf(i=j, t=t)

        return sum([(xi[d] - xj[d]) ** norm for d in range(self.__dim)]) ** (1. / norm)

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
        for tInit, tLast, events in data:

            for i, j, e in events:
                # Non-events
                if e < 0:
                    ci = self.__approxIntensityIntegral(i=i, j=j, tInit=tInit, tLast=tLast)
                    neg_logLikelihood += -ci
                # Events
                else:
                    neg_logLikelihood += self.__computeLogIntensityForPair(i=i, j=j, t=e)

        return -neg_logLikelihood

    def learn(self, type="seq"):

        # List for storing the training and testing set losses
        trainLoss, testLoss = [], []

        # Learns the parameters sequentially
        if type == "seq":

            for inx in list(range(1, self.__order + 1)) + [0]:  # range(order+1):

                print("Index: {}".format(inx))

                if inx == 0:
                    self.initializeModelParams(gamma=torch.rand(size=(self.__numOfNodes, 1)))
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
                    # t0 = time.time()
                    # running_loss = 0.
                    # print(f"Batch {idx+1} of {len(training_batches)}")
                    self.train()
                    #optimizer.zero_grad()
                    for param in self.parameters():
                        param.grad = None
                    train_loss = self(data=self.__trainSet)
                    train_loss.backward()
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


        return trainLoss, testLoss