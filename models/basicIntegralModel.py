import torch
import time
import math
import numpy as np
import matplotlib.pyplot as plt

class BasicIntegralModel(torch.nn.Module):

    def __init__(self, order, seed=0):
        super().__init__()

        # Set the seed value
        self.__seed = seed
        torch.manual_seed(self.__seed)

        self.__order = order

        self.__c0 = torch.nn.Parameter( torch.rand(size=(1, )) )
        self.__c1 = torch.nn.Parameter( torch.rand(size=(1, )) )
        self.__c2 = torch.nn.Parameter( torch.rand(size=(1, )) )
        self.__c3 = torch.nn.Parameter( torch.rand(size=(1, )) )

    def computeFuncAt(self, t):

        return 0

    def analyticIntegral(self, tInit, tLast):

        integral = lambda t: ( (t ** 4) * (self.__c3 / 4) ) + ( (t ** 3) * (self.__c2 / 3) ) + ( (t ** 2) * (self.__c1 / 2) ) + ( (t ** 1) * (self.__c0 / 1) )

        return integral(tLast) - integral(tInit)

    def forward(self, data):

        loss = 0.
        for tInit, tLast, value in data:

            integralval = self.analyticIntegral(tInit=tInit, tLast=tLast)
            loss += 0.5 * (value - integralval) ** 2

        return loss

    def get_c3(self):

        return self.__c3

    def get_c2(self):

        return self.__c2

    def get_c1(self):

        return self.__c1

    def get_c0(self):

        return self.__c0


if __name__ == "__main__":

    c3 = 1
    c2 = -9
    c1 = 23
    c0 = -15

    seed = 0
    maxTime = 7 #5.5

    def poly(t, c3, c2, c1, c0):
        return ((t ** 3) * c3) + ((t ** 2) * c2) + ((t ** 1) * c1) + (c0)

    def integ(t, c3, c2, c1, c0):
        return ( (t ** 4) * (c3 / 4) ) + ( (t ** 3) * (c2 / 3) ) + ( (t ** 2) * (c1 / 2) ) + ( (t ** 1) * (c0 / 1) )

    torch.random.manual_seed(seed=seed)

    trainSet = [[t, t + 0.1, 0] for t in np.arange(0.0, maxTime, 2.0)]
    for i in range(len(trainSet)):
        trainSet[i][2] = integ(t=trainSet[i][1], c3=c3, c2=c2, c1=c1, c0=c0) - integ(t=trainSet[i][0], c3=c3, c2=c2, c1=c1, c0=c0)
    np.random.shuffle(trainSet)

    # Set the number of epochs
    num_epochs = 3000
    # Set the learning rate
    lr = 0.1
    # Training loss
    train_loss = []
    # Testing loss
    test_loss = []

    # Define the models
    bim = BasicIntegralModel(order=3, seed=0)

    # Define the optimizer
    optimizer = torch.optim.Adam(bim.parameters(), lr=lr)

    for epoch in range(num_epochs):
        t0 = time.time()
        running_loss = 0.
        # print(f"Batch {idx+1} of {len(training_batches)}")
        bim.train()
        optimizer.zero_grad()
        loss = bim(data=trainSet)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if epoch % 50 == 0:
            print(f"Epoch {epoch + 1} train loss: {running_loss / len(trainSet)}")

        train_loss.append(running_loss / len(trainSet))

    c0_est = bim.get_c0().detach().numpy()[0]
    c1_est = bim.get_c1().detach().numpy()[0]
    c2_est = bim.get_c2().detach().numpy()[0]
    c3_est = bim.get_c3().detach().numpy()[0]
    print("Estimated: c0: {} c1: {} c2: {} c3: {}".format(c0_est, c1_est, c2_est, c3_est) )
    print("Correct: c0: {} c1: {} c2: {} c3: {}".format(c0, c1, c2, c3))
    # print("Estimated: c0: {} c1: {} c2: {} c3: {}".format(c0_est, c1_est, c2_est, c3_est))

    plt.figure()
    timePoints = np.arange(0, maxTime, 0.1)
    plt.plot(timePoints, [poly(t=t, c3=c3, c2=c2, c1=c1, c0=c0) for t in timePoints], 'k-')
    plt.plot(timePoints, [poly(t=t, c3=c3_est, c2=c2_est, c1=c1_est, c0=c0_est) for t in timePoints], 'r-')
    plt.show()


    print(integ(t=1, c3=c3, c2=c2, c1=c1, c0=c0) - integ(t=0, c3=c3, c2=c2, c1=c1, c0=c0),
          integ(t=3, c3=c3, c2=c2, c1=c1, c0=c0) - integ(t=2, c3=c3, c2=c2, c1=c1, c0=c0))

    print(integ(t=1, c3=c3_est, c2=c2_est, c1=c1_est, c0=c0_est) - integ(t=0, c3=c3_est, c2=c2_est, c1=c1_est, c0=c0_est),
          integ(t=3, c3=c3_est, c2=c2_est, c1=c1_est, c0=c0_est) - integ(t=2, c3=c3_est, c2=c2_est, c1=c1_est, c0=c0_est))