import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
basefolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import numpy as np
import matplotlib.pyplot as plt
from lib.animation import *
from lib.latent import *
from models.trainNhppNm import *

# ===== Model Parameters ===== #
seed = 0
gamma = 1.25
numOfStudents = 120
initMean = [0, 0]
initCov = 256 * np.eye(2)
groupAMean = [-30, 30]
groupACov = 36*np.eye(2)
groupBMean = [30, 30]
groupBCov = 26*np.eye(2)
groupCMean = [0, -50]
groupCCov = 16*np.eye(2)

sizeOfGroupA = 20
sizeOfGroupB = 40
sizeOfGroupC = 60

turningTime = 2
groupConstructionTime = 4
lastTime = 6
timeInterval = 0.01

# Set the seed
np.random.seed(seed)
# Initial position
x0 = np.random.multivariate_normal(mean=initMean, cov=initCov, size=numOfStudents)

# Perform random movements until the turning point
x = [x0]
timePoints = [0]
for tIdx, t in enumerate(np.arange(timeInterval, turningTime, timeInterval)):
    timePoints.append(t)
    x.append(x[tIdx] + np.random.multivariate_normal(mean=np.zeros(shape=(2, )), cov=np.eye(N=2), size=numOfStudents) )

# Assign the points into 3 groups
allInx = np.arange(0, numOfStudents)
np.random.shuffle(allInx)
groupAInx = allInx[0:sizeOfGroupA]
groupBInx = allInx[sizeOfGroupA:sizeOfGroupA+sizeOfGroupB]
groupCInx = allInx[sizeOfGroupA+sizeOfGroupB:sizeOfGroupA+sizeOfGroupB+sizeOfGroupC]
# Sample positions within the groups
xg = np.zeros_like(x0)
xg[groupAInx, :] = np.random.multivariate_normal(mean=groupAMean, cov=groupACov, size=sizeOfGroupA)
xg[groupBInx, :] = np.random.multivariate_normal(mean=groupBMean, cov=groupBCov, size=sizeOfGroupB)
xg[groupCInx, :] = np.random.multivariate_normal(mean=groupCMean, cov=groupCCov, size=sizeOfGroupC)
# Move the points towards their group positions
lastPos = x[-1]
for tIdx, t in enumerate(np.arange(turningTime, groupConstructionTime, timeInterval)):
    timePoints.append(t)
    x.append( lastPos + ((xg - lastPos) / (groupConstructionTime-turningTime)) * ( (tIdx+1) * timeInterval ) )
    x[-1] += np.random.multivariate_normal(mean=np.zeros(shape=(2, )), cov=0.1*np.eye(N=2), size=numOfStudents)

# Perform random movements until the last time point within the groups
lastPos = x[-1]
for tIdx, t in enumerate(np.arange(groupConstructionTime, lastTime, timeInterval)):
    timePoints.append(t)
    x.append(lastPos + np.random.multivariate_normal(mean=np.zeros(shape=(2, )), cov=0.1*np.eye(N=2), size=numOfStudents) )


#
# anim = Animation(timePoints=timePoints,
#                  r=1, c=1, figSize=(6, 6), bgColor='white',
#                  color='k', marker='.', markerSize=10, delay=250, margin=0.1)
# anim.addData(x=np.asarray(x), index=0, title="Ground-truth")
# # anim.addData(x=z_est, index=1, title="Estimation")
# anim.plot(filePath=os.path.join(basefolder, "figures", "students_and_courses.gif"))


network = np.load("net.npy", allow_pickle=True)
# d2d = Disc2Dataset(discretePos=np.asarray(x), timePoints=timePoints, gamma=gamma, seed=seed)
# network = d2d.constructNetwork()
# np.save("net.npy", network)

#print(network[0][1])

windowSize = 1.0
stepSize = 1.0
maxTime = lastTime
eventTimes = network[0][98]
numOfNodes = numOfStudents
dim = 2
order = 3
numOfEpochs = 10
lr = 0.1

########################################
###### Construct the training set ######
########################################
trainSet = [[t, t+windowSize, []] for t in np.arange(0.0, maxTime, stepSize)]
for i in range(len(trainSet)):
    for k in range(len(eventTimes)):
        if trainSet[i][0] <= eventTimes[k] <= trainSet[i][1]:
            trainSet[i][2].append( eventTimes[k] )
# Sort the training set
np.random.shuffle(trainSet)

########################################
########### Define the model ###########
########################################
nhppNmmModel = TrainNhppNm(trainSet=trainSet, testSet=None, numOfepochs=numOfEpochs, lr=lr,
                           numOfNodes=numOfNodes, dim=dim, order=order, numOfSamples=10, timeInterval=0.01, seed=0)

train_loss, test_loss = nhppNmmModel.learn()

# Plot the training loss
plt.figure()
plt.title("Training Loss")
plt.plot(train_loss)
plt.xlabel("Epochs")
plt.ylabel("Negative Log-likelihood")
plt.savefig(os.path.join("./", "trainingLoss_{}.png".format("bigmodel")))