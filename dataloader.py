from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class FaceDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        csv_filename = os.path.join(self.root_dir, csv_file)
        self.labels = pd.read_csv(csv_filename)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = io.imread(img_name)
        image = transform.resize(image, (256, 256), preserve_range=True)
        labels = self.labels.iloc[idx, 1]

        if self.transform:
            print('before', image)
            image = self.transform(image)
            print('after', image)
        sample = (image, labels)
        return sample

class FaceLoader(object):
    def __init__(self, args):
        super(FaceLoader, self).__init__()
        transform = transforms.Compose(
            [
             # TODO: Add data augmentations here
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(20),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = FaceDataset('dev_labels.csv', './data', transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize,
                                                  shuffle=True)

        devset = FaceDataset('dev_labels.csv', './data', transform_test)
        self.devloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize,
                                                  shuffle=True, num_workers=1)

        testset = FaceDataset('test_labels.csv', './data', transform_test)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
                                                 shuffle=False, num_workers=1)
