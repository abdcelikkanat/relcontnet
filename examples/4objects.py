import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.trainNhppNm import *
from datasets.synthetic.nhppnm import *
import matplotlib.pyplot as plt
import torch
from lib.animation import *
from examples.utilities import *

# Definition of the models
examples = dict()
# t=? gamma = 1.25
examples[3] = np.asarray( [ [[-15, 0], [15, 0], [0, 8], [0, -8]],
                            [[23, 0], [-23, 0], [0, -6], [0, 6]],
                            [[-18, 0], [18, 0], [0, 2], [0, -2]],
                            [[6, 0], [-6, 0], [0, 0], [0, 0]] ] )


########################################
########## Parameter Settings ##########
########################################
# Choose the model

# Set the time and other parameters
modelNumber = 3
seed = 0
gamma = 1.75
maxTime = 7  #5.5
trainingsetTime = 5
timeInterval = 0.01
windowSize = 1.0  #1.0
stepSize = 1.0 #1.0
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
basedir = os.path.dirname(os.path.abspath(__file__))
filename = "model{}_seed={}_gamma={}_maxT={}_timeInter={}_windowSize={}_stepSize={}_numOfEpochs={}_lr={}"\
            .format(modelNumber, seed, gamma, maxTime, timeInterval, windowSize, stepSize, numOfEpochs, lr)

# Sample the event times
nm = NHPPNM(gamma=gamma, z0=z0, maxTime=maxTime, timeInterval=timeInterval, seed=seed)
networkEvents = nm.constructNetwork()

# Construct the training and testing sets
dataset = contructDataset(networkEvents, numOfNodes, windowSize, stepSize, maxTime)
trainSet, testSet = splitDataset(dataset=dataset, trainingsetTime=trainingsetTime)

# Learning the model
init_time = time.time()
nhppNmmModel = TrainNhppNm(trainSet=trainSet, testSet=testSet, numOfepochs=numOfEpochs, lr=lr,
                           numOfNodes=numOfNodes, dim=dim, order=order, numOfSamples=10, timeInterval=0.01, seed=seed)
train_loss, test_loss = nhppNmmModel.learn()
print("-> Total Running time: {}".format(time.time() - init_time))

# Define the node pairs
i, j = ( 0, 1 )
# Define the time points for the figures
timePoints = np.arange(0, maxTime, 0.1)

# Create a folder
basefolder = os.path.join(basedir, "../figures/", "example"+str(modelNumber))
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
                 color=['k', 'r', 'b', 'g'], marker='.', markerSize=10, delay=250, margin=[0.1, 0.1])
anim.addData(x=z_true, index=0, title="Ground-truth")
anim.addData(x=z_est, index=1, title="Estimation")
anim.plot(filePath=os.path.join(basefolder, "latentPositions_{}.gif".format(filename)))

# Plot the training and testing loss
plt.figure()
plt.title("Training Losses")
plt.plot(train_loss, 'r', label="Training")
plt.xlabel("Epochs")
plt.ylabel("Negative Log-likelihood")
plt.legend()
plt.savefig(os.path.join(basefolder, "trainLoss_{}.png".format(filename)))

# Plot the testing loss
plt.figure()
plt.title("Testing Losses")
plt.plot(test_loss, 'b', label="Testing")
plt.xlabel("Epochs")
plt.ylabel("Negative Log-likelihood")
plt.legend()
plt.savefig(os.path.join(basefolder, "testLoss_{}.png".format(filename)))

# Plot the node pair distances
plt.figure()
true_dist, est_dist = [], []
for t in timePoints:
    true_dist.append(nm.getDistanceBetween(i=i, j=j, t=t))
    est_dist.append(nhppNmmModel.getDistanceBetween(i=i, j=j, t=t).detach().numpy())

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].title.set_text("True Distances for ({} {}) pair".format(i, j))
ax[0].plot(timePoints, true_dist, 'k-')
ax[1].title.set_text("Estimated Distances".format(i, j))
ax[1].plot(timePoints, true_dist, 'r-')
ax[0].set_xlabel("Time")
ax[1].set_xlabel("Time")
plt.savefig(os.path.join(basefolder, "nodePairDistances_{}.png".format(filename)))


plt.figure()
plt.title("Squared Error for Intensity Function for ({} {}) pair")
intensityError = []
for t in timePoints:
    intensityError.append( (nm.computeIntensityForPair(i=i, j=j, t=t) - nhppNmmModel.computeIntensityForPair(i=i, j=j, t=t))**2 )
plt.plot(timePoints, intensityError, 'r-')
plt.xlabel("Time")
plt.savefig(os.path.join(basefolder, "intensityError_{}.png".format(filename)))

# Save the model parameters
torch.save(nhppNmmModel.state_dict(), os.path.join(basefolder, "modelParams_{}.pkl".format(filename)))

print("-> Train Set Ground-truth negative log-likelihood: {}".format(nm.getNLL(data=trainSet, numOfSamples=10)))
print("-> Train Set Estimated negative log-likelihood: {}".format(nhppNmmModel.forward(data=trainSet).detach().numpy()))
print("-> Test Set Ground-truth negative log-likelihood: {}".format(nm.getNLL(data=testSet, numOfSamples=10)))
print("-> Test Set Estimated negative log-likelihood: {}".format(nhppNmmModel.forward(data=testSet)))
