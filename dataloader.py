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

class CifarLoader(object):
	"""docstring for CifarLoader"""
	def __init__(self, args):
		super(CifarLoader, self).__init__()
		transform = transforms.Compose(
		    [
		     # TODO: Add data augmentations here
                     transforms.RandomHorizontalFlip(),
                     transforms.RandomRotation(20),
		     transforms.ToTensor(),
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		     ])

		transform_test = transforms.Compose([
		    transforms.ToTensor(),
		    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])

		trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		                                        download=True, transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root='./data', train=False,
		                                       download=True, transform=transform_test)
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
		                                         shuffle=False, num_workers=2)

		self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
