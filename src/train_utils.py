from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import scipy.io
import torchvision.transforms as tvtf
import numpy as np

class Samson(data.Dataset):

    img_folder = 'Data_Matlab'
    gt_folder = 'GroundTruth'
    training_file = 'samson_1.mat'
    labels_file = 'end3.mat'

    def __init__(self, root, transform=None, target_transform=None):
        """Init Samson dataset."""
        super(Samson, self).__init__()

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use 'https://rslab.ut.ac.ir/data' to download it")

        PATH = os.path.join(self.root, self.img_folder, self.training_file)
        PATH_L = os.path.join(self.root, self.gt_folder, self.labels_file)

        training_data = scipy.io.loadmat(PATH)
        labels = scipy.io.loadmat(PATH_L)

        self.train_data = training_data['V'].T
        self.labels = labels['A'].T
        temp_data = self.train_data.reshape(95, 95, -1)
        temp_label = self.labels.reshape(95, 95, -1)
        data, label = self.createImageCubes(temp_data, temp_label)
        self.data = data
        self.label = label

    def padWithZeros(self, X, margin=2):
        newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    def createImageCubes(self, X, y):
        windowSize = 11
        margin = int((windowSize - 1) / 2)
        zeroPaddedX = self.padWithZeros(X, margin=margin)
        # split patches
        patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
        patchesLabels = np.zeros((X.shape[0] * X.shape[1], y.shape[2]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r-margin, c-margin]
                patchIndex = patchIndex + 1
        return patchesData, patchesLabels

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is abundance fractions for each pixel.
        """
        
        img, target = self.data[index], self.label[index]
        

        if self.transform is not None:
            img = torch.tensor(img)

        if self.target_transform is not None:
            target = torch.tensor(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""

        return len(self.data)
        

    def _check_exists(self):
        """Check if the path specified exists."""
        return os.path.exists(os.path.join(self.root, self.img_folder, self.training_file)) and os.path.exists(
            os.path.join(self.root, self.gt_folder, self.labels_file)
        )
        
        
def get_dataloader(BATCH_SIZE: int, DIR):
    """Create a DataLoader for input data stream."""
    trans = tvtf.Compose([tvtf.ToTensor()])

    # Load train data
    source_domain = Samson(root=DIR, transform=trans, target_transform=trans)
    source_dataloader = torch.utils.data.DataLoader(source_domain, BATCH_SIZE)
    
    return source_dataloader, source_domain