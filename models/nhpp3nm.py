import torch
import time


class TrainNhpp3Nm(torch.nn.Module):

    def __init__(self, numOfNodes, dim, trainSet, tn_test, timeInterval=0.01, seed=0):
        super().__init__()

        # Set the seed value
        self.__seed = seed
        torch.manual_seed(self.__seed)

        self.__numOfNodes = numOfNodes
        self.__dim = dim

        # Set the models parameters
        self.__gamma = torch.rand(size=(self.__numOfNodes, 1))
        self.__x0 = torch.rand(size=(self.__numOfNodes, self.__dim))
        self.__v0 = torch.rand(size=(self.__numOfNodes, self.__dim))
        self.__a0 = torch.rand(size=(self.__numOfNodes, self.__dim))

        # self.__gamma = torch.from_numpy(np.array([1.25, 1.25]))
        # # self.__x0 = torch.nn.Parameter(torch.rand(size=(self.__numOfNodes, self.__dim)))
        # self.__x0 = torch.from_numpy(np.array([[-5, 0], [5, 0]]))
        # self.__v0 = torch.nn.Parameter(torch.rand(size=(self.__numOfNodes, self.__dim)))
        # self.__a0 = torch.nn.Parameter(torch.rand(size=(self.__numOfNodes, self.__dim)))
        # self.__a0 = torch.from_numpy(np.array([[0, 0], [0, 0]]))

        self.__timeInterval = timeInterval

        # Get the node indices
        self.__nodePairs = torch.triu_indices(row=self.__numOfNodes, col=self.__numOfNodes, offset=1)

        # self.tn_train = tn_train # last time point on time axis in simul
        # self.tn_test = tn_test
        # self.pdist = nn.PairwiseDistance(p=2) # euclidean

    def initializeModelParams(self, gamma=None, x0=None, v0=None, a0=None):

        if gamma is not None:
            self.__gamma = torch.nn.Parameter(gamma)
        if x0 is not None:
            self.__x0 = torch.nn.Parameter(x0)
        if v0 is not None:
            self.__v0 = torch.nn.Parameter(v0)
        if a0 is not None:
            self.__a0 = torch.nn.Parameter(a0)

    def get_gamma(self):

        return self.__gamma

    def get_x0(self):

        return self.__x0

    def get_v0(self):

        return self.__v0

    def get_a0(self):

        return self.__a0

    def getPositionOf(self, i, t):

        return self.__x0[i, :] + ( self.__v0[i, :] * t ) + ( self.__a0[i, :] * (t ** 2) / 2.0 )

    def getDistanceBetween(self, i, j, t, order=2):

        xi = self.getPositionOf(i=i, t=t)
        xj = self.getPositionOf(i=j, t=t)

        return torch.sum( torch.pow(xi - xj, order) ) ** (1. / order)

    def __computeIntensityIntegral(self, i, j, tInit, tLast, numOfSamples=10):

        if tInit >= tLast:
            raise ValueError("The initial time {} cannot be greater than the last time step {}!".format(tInit, tLast))

        value = 0.
        # for t in np.arange(tInit, tLast, self.__timeInterval):
        #     value += self.__computeIntensityForPair(i=i, j=j, t=t) * self.__timeInterval
        timePoints, stepSize = np.linspace(start=tInit, stop=tLast, num=numOfSamples, endpoint=False, retstep=True)
        for t in timePoints:
            value += self.__computeIntensityForPair(i=i, j=j, t=t) * stepSize

        return value

    def __computeIntensityForPair(self, i, j, t):

        return torch.exp( self.__gamma[i] + self.__gamma[j] - self.getDistanceBetween(i=i, j=j, t=t) )

    def __computeLogIntensityForPair(self, i, j, t):

        return self.__gamma[i] + self.__gamma[j] - self.getDistanceBetween(i=i, j=j, t=t)

    def forward(self, data):

        neg_logLikelihood = 0.
        for tInit, tLast, k in data:
            ci = self.__computeIntensityIntegral(i=0, j=1, tInit=tInit, tLast=tLast, numOfSamples=10)
            neg_logLikelihood += -ci
            if k > 0:
                #neg_logLikelihood += k * torch.log(self.__computeLogIntensityForPair(i=0, j=1, t=t))
                neg_logLikelihood += torch.log(ci)

                # neg_logLikelihood += -torch.log(torch.exp(torch.lgamma(torch.from_numpy(np.array(k)).type(torch.float))))


        return -neg_logLikelihood


if __name__ == "__main__":
    from datasets.synthetic.nhpp3nm import *
    import matplotlib.pyplot as plt
    x0 = [[-5, 0], [5, 0]]
    v0 = [[2.5, 0], [-2.5, 0]]
    a0 = [[0, 0], [0, 0]]

    numOfNodes = 2
    dim = 2

    seed = 0
    maxTime = 10
    timeInterval = 0.01

    torch.random.manual_seed(seed=seed)

    nm = NHPP3NM(x0=x0, v0=v0, a0=a0, maxTime=maxTime, timeInterval=timeInterval, seed=seed)
    network = nm.constructNetwork()
    print((network[0][1]))

    eventTimes = sorted(network[0][1])

    # Set up the training set
    trainSet = [[t-0.15, t, 0] for t in np.arange(0.25, maxTime, 0.5)]
    k = 0
    for i in range(len(trainSet)):
        count = 0
        for k in range(len(eventTimes)):
            if trainSet[i][0] <= eventTimes[k] <= trainSet[i][1]:
                count += 1
        trainSet[i][2] = count
    # trainSet = [ [t, 0] for t in np.arange(timeInterval, maxTime, timeInterval) ]
    # k = 0
    # for i in range(len(trainSet)):
    #     if k < len(eventTimes):
    #         if trainSet[i][0] > eventTimes[k]:
    #             k = k + 1
    #     trainSet[i][1] = k

    np.random.shuffle(trainSet)

    # Set the number of epochs
    num_epochs = 300
    # Set the learning rate
    lr = 0.1
    # Training loss
    train_loss = []
    # Testing loss
    test_loss = []

    # Define the models
    nhppNmmModel = TrainNhpp3Nm(numOfNodes=2, dim=2, trainSet=trainSet, tn_test=None)

    for inx in range(4):

        if inx == 0:
            nhppNmmModel.initializeModelParams(gamma=torch.rand(size=(numOfNodes, 1)))
        if inx == 1:
            nhppNmmModel.initializeModelParams(x0=torch.rand(size=(numOfNodes, dim)))
        if inx == 2:
            nhppNmmModel.initializeModelParams(v0=torch.rand(size=(numOfNodes, dim)))
        if inx == 3:
            nhppNmmModel.initializeModelParams(a0=torch.rand(size=(numOfNodes, dim)))

        print(len(list(nhppNmmModel.parameters())))

        # Define the optimizer
        optimizer = torch.optim.Adam(nhppNmmModel.parameters(), lr=lr)

        for epoch in range(num_epochs):
            t0 = time.time()
            running_loss = 0.
            # print(f"Batch {idx+1} of {len(training_batches)}")
            nhppNmmModel.train()
            optimizer.zero_grad()
            loss = nhppNmmModel(data=trainSet)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if epoch % 50 == 0:
                print(f"Epoch {epoch + 1} train loss: {running_loss / len(trainSet)}")

            train_loss.append(running_loss / len(trainSet))

    # nhppNmmModel
    print("gamma")
    print(nhppNmmModel.get_gamma().detach().numpy())
    print("x")
    print(nhppNmmModel.get_x0().detach().numpy())
    print("v")
    print(nhppNmmModel.get_v0().detach().numpy())
    print("a")
    print(nhppNmmModel.get_a0().detach().numpy())

    print(nhppNmmModel.getPositionOf(i=0, t=2).detach().numpy(), "--",
          nhppNmmModel.getPositionOf(i=1, t=2).detach().numpy())
    print(nhppNmmModel.getDistanceBetween(i=0, j=1, t=2).detach().numpy())

    plt.figure(1)
    plt.plot(range(len(train_loss)), train_loss, '.')
    plt.title("Loss")
    plt.show()
    plt.figure(2)
    #plt.plot(range(len(train_loss)), train_loss, '.')
    dist = []
    time_list = np.arange(0, maxTime, timeInterval)
    for t in time_list:
        dist.append(nhppNmmModel.getDistanceBetween(i=0, j=1, t=t).detach().numpy())
    plt.plot(time_list, dist, 'r.')
    plt.title("Distance")
    plt.show()