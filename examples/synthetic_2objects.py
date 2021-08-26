import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.trainNhppNm import *
from datasets.synthetic.nhppnm import *
import matplotlib.pyplot as plt
import torch
from lib.animation import *

# Definition of the models
examples = dict()
# t=2
examples[0] = np.asarray([[[-5, 0], [5, 0]], [[2.5, 0], [-2.5, 0]], [[0, 0], [0, 0]]])  # gamma=1.25
# t=? gamma = 1.25
examples[1] = np.asarray([[[+12, 0], [-12, 0]],
                          [[-8, 0], [8, 0]],
                          [[2, 0], [-2, 0]]])
# t=? gamma = 1.75
examples[2] = np.asarray([[[-15, 0], [15, 0]],
                          [[23, 0], [-23, 0]],
                          [[-18, 0], [18, 0]],
                          [[6, 0], [-6, 0]]])

examples[3] = np.asarray( [ [[-15, 0], [15, 0], [0, 8], [0, -8]],
                            [[23, 0], [-23, 0], [0, -6], [0, 6]],
                            [[-18, 0], [18, 0], [0, 2], [0, -2]],
                            [[6, 0], [-6, 0], [0, 0], [0, 0]] ] )

# not colliding example
## examples[4] = # c = np.asarray([ [[-15, 0], [15, 0]], [[23, 0], [-23, 0]], [[-18, 0], [18, 0]], [[6, 0], [-6, 0]] ] )
# c[0, 0, 0] += -5
# c[0, 1, 0] += +5


########################################
########## Parameter Settings ##########
########################################
# Choose the model

# Set the time and other parameters
modelNumber = 2
seed = 0
gamma = 1.75
maxTime = 8  #5.5
timeInterval = 0.01
windowSize = 1.0
stepSize = 1.0
# Set the number of epochs
numOfEpochs = 300
# Set the learning rate
lr = 0.1

# Extract the model parameters
z0 = examples[modelNumber]
order = z0.shape[0]
numOfNodes = z0.shape[1]
dim = z0.shape[2]
# Set the seed value for torch
torch.random.manual_seed(seed=seed)

# Animation
basedir = "../"
filename = "model{}_seed={}_gamma={}_maxT={}_timeInter={}_windowSize={}_stepSize={}_numOfEpochs={}_lr={}"\
            .format(modelNumber, seed, gamma, maxTime, timeInterval, windowSize, stepSize, numOfEpochs, lr)

########################################
######### Sampling event times #########
########################################
nm = NHPPNM(gamma=gamma, z0=z0, maxTime=maxTime, timeInterval=timeInterval, seed=seed)
network = nm.constructNetwork()
eventTimes = network[0][1]
print(len(eventTimes), eventTimes)
# Sort the events
#eventTimes = sorted(network[0][1])

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

# Define the node pairs
i, j = ( 0, 1 )
# Define the time points for the figures
timePoints = np.arange(0, maxTime, 0.1)

# Create a folder
basefolder = os.path.join(basedir, "relcontnet/figures/", "example"+str(modelNumber))
if not os.path.isdir(basefolder):
    os.makedirs(name=basefolder)

# Animate the latent positions
plt.figure()
z_true = np.zeros(shape=(len(timePoints), numOfNodes, dim), dtype=np.float)
z_est = np.zeros(shape=(len(timePoints), numOfNodes, dim), dtype=np.float)
for inx in range(len(timePoints)):
    z_true[inx, :, :] = nm.getAllPositions(t=timePoints[inx])
    z_est[inx, :, :] = nhppNmmModel.getAllPositions(t=timePoints[inx]).detach().numpy()
anim = Animation(timePoints=timePoints,
                 r=1, c=2, figSize=(12, 6), bgColor='white',
                 color='k', marker='.', markerSize=20, delay=250, margin=0.1)
anim.addData(x=z_true, index=0, title="Ground-truth")
anim.addData(x=z_est, index=1, title="Estimation")
anim.plot(filePath=os.path.join(basefolder, "latentPositions_{}.gif".format(filename)))

# Plot the training loss
plt.figure()
plt.title("Training Loss")
plt.plot(train_loss)
plt.xlabel("Epochs")
plt.ylabel("Negative Log-likelihood")
plt.savefig(os.path.join(basefolder, "trainingLoss_{}.png".format(filename)))

# Plot the node pair distances
plt.figure()
true_dist, est_dist = [], []
for t in timePoints:
    true_dist.append(nm.getDistanceBetween(i=i, j=j, t=t))
    est_dist.append(nhppNmmModel.getDistanceBetween(i=i, j=j, t=t).detach().numpy())

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].title.set_text("True Distances".format(i, j))
ax[0].plot(timePoints, true_dist, 'k-')
ax[1].title.set_text("Estimated Distances".format(i, j))
ax[1].plot(timePoints, true_dist, 'r-')
ax[0].set_xlabel("Time")
ax[1].set_xlabel("Time")
plt.savefig(os.path.join(basefolder, "nodePairDistances_{}.png".format(filename)))


plt.figure()
plt.title("Squared Error for Intensity Function")
intensityError = []
for t in timePoints:
    intensityError.append( (nm.computeIntensityForPair(i=i, j=j, t=t) - nhppNmmModel.computeIntensityForPair(i=i, j=j, t=t))**2 )
plt.plot(timePoints, intensityError, 'r-')
plt.xlabel("Time")
plt.savefig(os.path.join(basefolder, "intensityError_{}.png".format(filename)))

# Save the model parameters
torch.save(nhppNmmModel.state_dict(), os.path.join(basefolder, "modelParams_{}.pkl".format(filename)))