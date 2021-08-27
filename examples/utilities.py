import numpy as np


def contructDataset(networkEvents, numOfNodes, windowSize, stepSize, maxTime):

    dataset = [[t, t + windowSize, []] for t in np.arange(0.0, maxTime, stepSize)]
    for tInx in range(len(dataset)):
        inx1, inx2 = np.triu_indices(numOfNodes, k=1)
        for i, j in zip(inx1, inx2):
            events_ij = []
            for e in networkEvents[i][j]:
                if dataset[tInx][0] <= e <= dataset[tInx][1]:
                    events_ij.append(e)
            dataset[tInx][2].append([[i, j], events_ij])

    return dataset


def splitDataset(dataset, trainingsetTime):

    trainSet, testSet = [], []
    for tInx in range(len(dataset)):
        if dataset[tInx][1] <= trainingsetTime:
            trainSet.append(dataset[tInx])
        else:
            testSet.append(dataset[tInx])

    return trainSet, testSet