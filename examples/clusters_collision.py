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
torch.set_num_threads(16)

print("Number of threads: {}".format(torch.get_num_threads()))

# Definition of the models
examples = dict()
# t=1 t=3 and t=5

# Parameter Settings for network
seed = 0
dim = 2
clusterASize = 8 # 64
clusterACov = 0.1 * np.eye(dim)
clusterBSize = 6 # 36
clusterBCov = 0.5 * np.eye(dim)
numOfNodes = clusterASize + clusterBSize
# Parameter Settings for learning
numOfNegativeSamples = 5
gamma = 1.75
maxTime = 7.0  #5.5
trainingsetTime = 5.0
timeInterval = 0.01
windowSize = 1.0
stepSize = 1.0
numOfEpochs = 300
lr = 0.1

# Set the seed value for torch
torch.random.manual_seed(seed=seed)
np.random.seed(seed=seed)

# Set the base directory
basedir = os.path.dirname(os.path.abspath(__file__))
filename = "clustercollision_seed={}_gamma={}_maxT={}_timeInter={}_windowSize={}_stepSize={}_numOfEpochs={}_lr={}"\
            .format(seed, gamma, maxTime, timeInterval, windowSize, stepSize, numOfEpochs, lr)
basefolder = os.path.join(basedir, "../figures/", "cluster_collision")

# Construc the clusters
c = np.asarray([ [[-15, 0], [15, 0]], [[23, 0], [-23, 0]], [[-18, 0], [18, 0]], [[6, 0], [-6, 0]] ] )
z0 = np.zeros(shape=(c.shape[0], clusterASize+clusterBSize, c.shape[2]), dtype=np.float)
r = 2.0
for o in range(c.shape[0]):
    z0[o, 0:clusterASize, :] = c[o, 0, :]
    z0[o, clusterASize:clusterASize+clusterBSize, :] = c[o, 1, :]
for i in range(numOfNodes):
    z0[0, i, 0] += r * np.random.normal(loc=0, scale=1.0, size=(1,))
    z0[0, i, 1] += r * np.random.normal(loc=0, scale=1.0, size=(1,))
# Extract the model parameters
order = z0.shape[0]
numOfNodes = z0.shape[1]
dim = z0.shape[2]

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
print("Total Running time: {}".format(time.time() - init_time))

# Define the node pairs
i, j = ( 0, 1 )
# Define the time points for the figures
timePoints = np.arange(0, maxTime, 0.1)

# Animate only the ground-truth movements
plt.figure()
z_true = np.zeros(shape=(len(timePoints), numOfNodes, dim), dtype=np.float)
for inx in range(len(timePoints)):
    z_true[inx, :, :] = nm.getAllPositions(t=timePoints[inx])
for inx in range(len(timePoints)):
    z_true[inx, :, :] = nm.getAllPositions(t=timePoints[inx])
anim = Animation(timePoints=timePoints,
                 r=1, c=1, figSize=(8, 6), bgColor='white',
                 color=['r' for _ in range(clusterASize)] + ['b' for _ in range(clusterBSize)],
                 marker='.', markerSize=5, delay=250, margin=[0.1, 3.0])
anim.addData(x=z_true, index=0, title="Ground-truth")
anim.plot(filePath=os.path.join(basefolder, "singlelatentPosition_{}.gif".format(filename)))

# Animate the latent positions
plt.figure()
z_true = np.zeros(shape=(len(timePoints), numOfNodes, dim), dtype=np.float)
z_est = np.zeros(shape=(len(timePoints), numOfNodes, dim), dtype=np.float)
for inx in range(len(timePoints)):
    z_true[inx, :, :] = nm.getAllPositions(t=timePoints[inx])
    z_est[inx, :, :] = nhppNmmModel.getAllPositions(t=timePoints[inx]).detach().numpy()
anim = Animation(timePoints=timePoints,
                 r=1, c=2, figSize=(12, 6), bgColor='white',
                 color='k', marker='.', markerSize=5, delay=250, margin=[0.1, 3.0])
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


print("Train Ground-truth negative log-likelihood: {}".format(nm.getNLL(data=trainSet, numOfSamples=10)))
print("Train Estimated negative log-likelihood: {}".format(nhppNmmModel.forward(data=trainSet).detach().numpy()))
print("Test Ground-truth negative log-likelihood: {}".format(nm.getNLL(data=testSet, numOfSamples=10)))
print("Test Estimated negative log-likelihood: {}".format(nhppNmmModel.forward(data=testSet)))



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