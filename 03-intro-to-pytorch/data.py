import numpy as np
from mnist import MNIST

import torch as th
from torch.utils.data import TensorDataset, DataLoader


class DataUtils:
    def __init__(self, batch=10):
        """
        Constructor

        Args:
            batch (int, optional): batch size of the train data.
            Defaults to 10.
        """
        self.batch_size = batch

    def trainTestLoad(self):
        """
        DataLoader function for the MNIST dataset

        Returns:
            numpy.ndarray: train and test data and labels
        """
        data = MNIST("data")
        trainData, trainLabels = data.load_training()
        testData, testLabels = data.load_testing()
        return (
            np.asarray(trainData),
            np.asarray(trainLabels),
            np.asarray(testData),
            np.asarray(testLabels),
        )

    def loadTorchDataset(self):
        """
        Dataloader function to use torch's dataloader

        Returns:
            tuple: Tuple of the train and test loaders
        """
        tr, trL, te, teL = self.trainTestLoad()
        trainDataset = TensorDataset(th.Tensor(tr), th.Tensor(trL))
        testData = th.Tensor(te)
        testLabels = th.Tensor(teL)
        trainLoader = DataLoader(trainDataset, batch_size=self.batch_size, shuffle=True)
        return trainLoader, testData, testLabels
