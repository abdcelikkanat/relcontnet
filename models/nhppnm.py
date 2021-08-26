import torch
import time
import math


class TrainNhppNm(torch.nn.Module):

    def __init__(self, numOfNodes, dim, order, numOfSamples=10, timeInterval=0.01, seed=0):
        super().__init__()

        # Set the seed value
        self.__seed = seed
        torch.manual_seed(self.__seed)

        self.__numOfNodes = numOfNodes
        self.__dim = dim
        self.__order = order
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

        neg_logLikelihood = 0.
        for tInit, tLast, eventTimes in data:
            for i, j in zip(self.__nodePairs[0], self.__nodePairs[1]):
                ci = self.__approxIntensityIntegral(i=i, j=j, tInit=tInit, tLast=tLast)
                neg_logLikelihood += -ci
                for e in eventTimes:
                    neg_logLikelihood += self.__computeLogIntensityForPair(i=i, j=j, t=e)

        return -neg_logLikelihood


if __name__ == "__main__":

    import os, sys

    print(os.path.dirname(__file__))
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from datasets.synthetic.nhppnm import *
    import matplotlib.pyplot as plt

    example1 = np.asarray([[[-5, 0], [5, 0]], [[2.5, 0], [-2.5, 0]], [[0, 0], [0, 0]]])
    example2 = np.asarray([ [[-10, 0], [10, 0]], [[1.0, 0], [-1.0, 0]], [[4, 0], [-4, 0]] ])
    example3 = np.asarray([[[-5, 0], [5, 0]], [[2.5, 0], [-2.5, 0]] ])
    example4 = np.asarray([[[+6, 0], [-6, 0]], [[-5, 0], [5, 0]], [[2, 0], [-2, 0]]])
    example5 = np.asarray([[[+12, 0], [-12, 0]], [[-8, 0], [8, 0]], [[2, 0], [-2, 0]]])
    example6 = np.asarray( [ [[-15, 0], [15, 0], [0, 8], [0, -8]], [[23, 0], [-23, 0], [0, -6], [0, 6]], [[-18, 0], [18, 0], [0, 2], [0, -2]], [[6, 0], [-6, 0], [0, 0], [0, 0]] ] )
    example7 = np.asarray([[[-15, 0], [15, 0]], [[23, 0], [-23, 0]],
                     [[-18, 0], [18, 0]], [[6, 0], [-6, 0]]])


    z0 = example7

    order = z0.shape[0]
    numOfNodes = z0.shape[1]
    dim = z0.shape[2]

    seed = 0
    maxTime = 8 #5.5
    timeInterval = 0.01

    torch.random.manual_seed(seed=seed)

    nm = NHPPNM(gamma=1.25, z0=z0, maxTime=maxTime, timeInterval=timeInterval, seed=seed)
    network = nm.constructNetwork()
    print(len(network[0][1]), network[0][1])

    eventTimes = sorted(network[0][1])

    # # Animation
    # from lib.animation import *
    # timePoints = np.arange(0, maxTime, 0.1)
    # x = np.zeros(shape=(len(timePoints), numOfNodes, dim), dtype=np.float)
    # for inx in range(len(timePoints)):
    #     x[inx, :, :] = nm.getAllPositions(t=timePoints[inx])
    # anim = Animation(timePoints=timePoints,
    #                  r=1, c=2, figSize=(20, 10), bgColor='white',
    #                  color='k', marker='.', markerSize=20, delay=250, margin=0.1)
    # anim.addData(x=x, index=0, title="1")
    # anim.addData(x=x, index=1, title="2")
    # anim.plot(filePath="/Users/abdulkadir/workspace/relcontnet/figures/example5_w=0.5.gif")


    # Set up the training set
    # trainSet = [[t-0.01, t, 0] for t in np.arange(0.1, maxTime, 0.03)]
    trainSet = [[t, t+1.0, []] for t in np.arange(0.0, maxTime, 1.0)]
    k = 0
    for i in range(len(trainSet)):
        # count = 0
        for k in range(len(eventTimes)):
            if trainSet[i][0] <= eventTimes[k] <= trainSet[i][1]:
                # count += 1
                trainSet[i][2].append( eventTimes[k] )
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
    nhppNmmModel = TrainNhppNm(numOfNodes=numOfNodes, dim=dim, order=order, numOfSamples=10, timeInterval=0.01, seed=0)

    for inx in list(range(1, order+1)) + [0]: #range(order+1):

        if inx == 0:
            nhppNmmModel.initializeModelParams(gamma=torch.rand(size=(numOfNodes, 1)))
            #nhppNmmModel.initializeModelParams(gamma=torch.zeros(size=(numOfNodes, 1)))
        else:
            nhppNmmModel.initializeModelParams(order=inx-1)
        # if inx == 1:
        #     nhppNmmModel.initializeModelParams(x0=torch.rand(size=(numOfNodes, dim)))
        # if inx == 2:
        #     nhppNmmModel.initializeModelParams(v0=torch.rand(size=(numOfNodes, dim)))
        # if inx == 3:
        #     nhppNmmModel.initializeModelParams(a0=torch.rand(size=(numOfNodes, dim)))

        print(len(list(nhppNmmModel.parameters())))
        print( list(nhppNmmModel.parameters()) )

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

            #nhppNmmModel.meanNormalization()

    # nhppNmmModel
    # print("gamma")
    # print(nhppNmmModel.get_gamma().detach().numpy())
    # print("x")
    # print(nhppNmmModel.get_x0().detach().numpy())
    # print("v")
    # print(nhppNmmModel.get_v0().detach().numpy())
    # print("a")
    # print(nhppNmmModel.get_a0().detach().numpy())

    print(nhppNmmModel.getPositionOf(i=0, t=2).detach().numpy(), "--",
          nhppNmmModel.getPositionOf(i=1, t=2).detach().numpy())
    print(nhppNmmModel.getDistanceBetween(i=0, j=1, t=2).detach().numpy())

    # Animation
    BASEDIR = "../" #"/Users/abdulkadir/workspace/"

    from lib.animation import *
    timePoints = np.arange(0, maxTime, 0.1)
    x = np.zeros(shape=(len(timePoints), numOfNodes, dim), dtype=np.float)
    x_est = np.zeros(shape=(len(timePoints), numOfNodes, dim), dtype=np.float)
    for inx in range(len(timePoints)):
        x[inx, :, :] = nm.getAllPositions(t=timePoints[inx])
        x_est[inx, :, :] = nhppNmmModel.getAllPositions(t=timePoints[inx]).detach().numpy()
    anim = Animation(timePoints=timePoints,
                     r=1, c=2, figSize=(20, 10), bgColor='white',
                     color='k', marker='.', markerSize=20, delay=250, margin=0.1)
    anim.addData(x=x, index=0, title="Ground-truth")
    anim.addData(x=x_est, index=1, title="Estimation")
    anim.plot(filePath=BASEDIR + "relcontnet/figures/example5_datasetfixed_maxt={}.gif".format(maxTime))


    plt.figure(2)
    plt.plot(range(len(train_loss)), train_loss, '.')
    plt.title("Loss")
    # plt.show()
    plt.savefig(BASEDIR + "relcontnet/figures/train_loss_datasetfixed_maxt={}.png".format(maxTime))

    plt.figure(3)
    #plt.plot(range(len(train_loss)), train_loss, '.')
    dist_corr = []
    dist_est = []
    for t in timePoints:
        dist_corr.append(nm.getDistanceBetween(i=0, j=1, t=t))
        dist_est.append(nhppNmmModel.getDistanceBetween(i=0, j=1, t=t).detach().numpy())
    plt.plot(timePoints, dist_corr, 'k.')
    plt.savefig(BASEDIR + "relcontnet/figures/distance_corr_datasetfixed_maxt={}.png".format(maxTime))

    plt.figure(4)
    plt.plot(timePoints, dist_est, 'r.')
    plt.title("Estimated Distance")
    # plt.show()
    plt.savefig(BASEDIR + "relcontnet/figures/distance_est_datasetfixed_maxt={}.png".format(maxTime))


    print("---| Estimated |---")
    print("Beta: {}".format(nhppNmmModel.get_gamma().detach().numpy()))
    for o in range(order):
        print("z{}: {}".format(o, nhppNmmModel.getLatentVariableAt(i=0, orderIndex=o, t=0).detach().numpy()),
              "z{}: {}".format(o, nhppNmmModel.getLatentVariableAt(i=1, orderIndex=o, t=0).detach().numpy()))