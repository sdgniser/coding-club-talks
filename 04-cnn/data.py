"""
Fork of DataUtils used by Spandan for FeedForwardNetwork with some modifications

Dataset: Small subset of Kaggle Dogs v Cats - https://www.kaggle.com/c/dogs-vs-cats

"""

import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader


class CNNDataUtils:
    def __init__(self, batch=50, val_share=0.1):
        """
        Constructor
        Args:
            batch (int, optional): batch size of the train data.
            Defaults to 10.
            val_share (float, optional): Portion of data to reserve for validation
            Defaults to 0.1 (10%).
        """
        self.batch_size = batch
        self.val_share = val_share

    def trainTestLoad(self):
        """
        DataLoader function for the MNIST dataset
        Returns:
            numpy.ndarray: train and test data and labels
        """
        training_data = np.load("data.npy", allow_pickle=True)

        # All images
        data = torch.Tensor([i[0] for i in training_data]).to('cuda').view(-1, 50, 50)
        data = data / 255.0
        # All labels
        labels = torch.Tensor([i[1] for i in training_data]).to('cuda')
        
        # Reserving a part of the data for post-training validation
        val_size = int(len(data) * self.val_share)
        trainData = data[:-val_size]
        trainLabels = labels[:-val_size]

        testData = data[-val_size:]
        testLabels = labels[-val_size:]

        return (
            trainData,
            trainLabels,
            testData,
            testLabels,
        )

    def loadTorchDataset(self):
        """
        Dataloader function to use torch's dataloader
        Returns:
            tuple: Tuple of the train and test loaders
        """
        tr, trL, te, teL = self.trainTestLoad()
        trainDataset = TensorDataset(tr.unsqueeze(dim=1), trL)
        testData = te.unsqueeze(dim=1)
        testLabels = teL
        trainLoader = DataLoader(trainDataset, batch_size=self.batch_size, shuffle=True)

        return trainLoader, testData, testLabels
