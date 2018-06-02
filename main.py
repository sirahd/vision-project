from utils import argParser
from dataloader import FaceLoader
import matplotlib.pyplot as plt
import numpy as np
import models
import torch
import pdb
import warnings
import os

warnings.filterwarnings("ignore")

def train(net, dataloader, optimizer, criterion, epoch):

    running_loss = 0.0
    total_loss = 0.0

    for i, data in enumerate(dataloader.trainloader, 0):
        # get the inputs
        inputs, labels = data
        #inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = outputs.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i + 1) % 50 == 0:    # print every 2000 mini-batches
            net.log('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            total_loss += running_loss
            running_loss = 0.0

    net.log('Final Summary:   loss: %.3f' %
          (total_loss / i))

def test(net, dataloader, tag=''):
    correct = 0
    total = 0
    if tag == 'Train':
        dataTestLoader = dataloader.trainloader
    else:
        dataTestLoader = dataloader.devloader
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data
            outputs = net(images)
            outputs = outputs.view(-1)

            itr = 0
            for i in outputs:
                #print (i)
                predicted = -1.0
                if i >= 0.0:
                    predicted = 1.0
                else:
                    predicted = 0.0
                correct += (predicted == labels[itr].item())
                itr += 1

            #print (outputs.data)
            #print (labels)
            total += len(labels)
            #print (correct)
            #print (total)

    net.log('%s Accuracy of the network: %d %%' % (tag,
        100 * correct / total))

    """
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for data in dataTestLoader:
            images, labels = data
            #images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(2):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        net.log('%s Accuracy of %5s : %2d %%' % (
            tag, dataloader.classes[i], 100 * class_correct[i] / class_total[i]))
    """

def main():

    args = argParser()

    loader = FaceLoader(args)
    net = args.model()
    print('The log is recorded in ')
    print(net.logFile.name)

    criterion = net.criterion()
    optimizer = net.optimizer()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        net.adjust_learning_rate(optimizer, epoch, args)
        train(net, loader, optimizer, criterion, epoch)
        if epoch % 1 == 0: # Comment out this part if you want a faster training
            test(net, loader, 'Train')
            test(net, loader, 'Test')

    model_path = os.path.join('model', net.__class__.__name__)
    torch.save(net.state_dict(), model_path)

    print('The log is recorded in ')
    print(net.logFile.name)
    print('The model is saved in {}'.format(model_path))
    print(model_path)

if __name__ == '__main__':
    main()

