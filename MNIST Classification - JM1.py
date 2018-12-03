#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 06:46:27 2018

@author: Jake
"""

'''
Using identical model from Fashion-MNIST to classify MNIST
Compare results

'''

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pylab as plt
import numpy as np
from collections import OrderedDict

############################
# UTILITY/HELPER FUNCTIONS #
############################
def show_image(image):
    '''
    Displays 28x28 pixel image with correct classification
    '''
    clothing_class = index_to_class(int(image[1]))
    plt.imshow(image[0].numpy().reshape(28,28))
    plt.title('Correct class: ' + clothing_class +
              ' (' + str(image[1].item()) + ')')

def index_to_class(index):
    '''
    Returns class of item based on index
    '''
    class_dict = {0 : '0',
                  1 : '1',
                  2 : '2',
                  3 : '3',
                  4	: '4',
                  5	: '5',
                  6	: '6',
                  7	: '7',
                  8	: '8',
                  9 : '9'}
    return(class_dict[index])

def get_activations(model, x):
    '''
    Returns activations on "x"
    '''
    x = x.view(1, 1, 28, 28)
    activations, activation_names = [], []
    
    for i, layer in enumerate(model.modules()):
        if i > 0:
            x = layer(x)
            if not any(word in str(layer) for word in ('MaxPool', 'Dropout',
                                  'Flatten')):
                activations.append(x)
                activation_names.append(str(layer).split('(', 1)[0])
    return activations, activation_names

def plot_parameters(weights, title):
    '''
    Plots out kernel parameters for visualization
    '''
    n_filters = weights.shape[0]
    n_rows = int(np.ceil(np.sqrt(n_filters)))

    min_value = weights.min().item()
    max_value = weights.max().item()
    
    fig, ax = plt.subplots(n_rows, ncols = n_filters//n_rows)
    fig.subplots_adjust(wspace = 1.0, hspace = 1.0)
    
    for i, ax in enumerate(ax.flat):
        ax.set_xlabel('Kernel: ' + str(i + 1))
        ax.imshow(weights[i][0], vmin = min_value, vmax = max_value,
                  cmap = 'seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(title)
    plt.show()

def plot_activations(model, x):
    '''
    Plots out activations for visualization
    Activations is a list - will need to iterate through
    '''
    activations, activation_names = get_activations(model, x)
    activations = [activation.detach().numpy() for activation in activations]
    n_activations = activations[1][0].shape[0]
    
    min_values = [value.min().item() for value in activations]
    max_values = [value.max().item() for value in activations]
    
    for i, layer in enumerate(activations):
        fig, ax = plt.subplots(4, n_activations//4, figsize = (20,20))        
        for j, ax in enumerate(ax.flat):
            ax.set_xlabel('Activation: ' + str(j + 1))
            ax.imshow(activations[0][0][j], vmin = min_values[i],
                      vmax = max_values[i], cmap = 'seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    
        plt.suptitle('Activation Layer [{}]'.format(activation_names[i]))
        plt.show()
    
class Flatten(nn.Module):
    '''
    Used in Sequential container to flatten tensor
    '''
    def forward(self, input):
        return input.view(input.size(0), -1)

#####################
# Functions for CNN #
#####################
def create_datasets():
    '''
    Creates FashionMNIST dataset
    Creates "train_dataset" and "validation_dataset"
    Currently saves to "./JM1"
    '''
    train_dataset = dsets.MNIST(root = './JM1',
                                train = True,
                                download = True,
                                transform = transforms.ToTensor())
    validation_dataset = dsets.MNIST(root = './JM1',
                                     train = False,
                                     download = True,
                                     transform = transforms.ToTensor())
    return train_dataset, validation_dataset

def create_loaders(train_dataset, validation_dataset):
    '''
    Creates "train_loader" and "validation_loader" from respective datasets
    '''
    train_loader = DataLoader(train_dataset, batch_size = 50)
    validation_loader = DataLoader(validation_dataset, batch_size = 50)
    return train_loader, validation_loader

def train_model(model, epochs, train_loader, validation_loader,
                optimizer, criterion, loss_list, train_accuracy_list,
                val_accuracy_list, n_train_dataset, n_val_dataset):
    '''
    Trains "model" for "n_epochs" using "train_loader, optimizer, criterion"
    '''
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        print('\nEvaluating epoch: ' + str(epoch + 1) + '/' + str(epochs))
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            _, y_hat = torch.max(output.data, 1)
            train_correct += (y_hat == y).sum().item()
        loss_list.append(loss.data)
        train_accuracy = train_correct / n_train_dataset
        train_accuracy_list.append(train_accuracy)
        val_accuracy = check_accuracy(model, validation_loader, n_val_dataset)
        val_accuracy_list.append(val_accuracy)
        print('-- Accuracy after {} epochs --\
              \n----- Training:  {}% -----\
              \n---- Validation:  {}% ----'.format(
              epoch + 1,
              round(train_accuracy * 100, 2),
              round(val_accuracy * 100, 2)))

def check_accuracy(model, validation_loader, n_val_dataset):
    '''
    Checks accuracy of "model" using "validation_loader, validation_dataset"
    '''
    correct = 0
    model.eval()
    print('Validating accuracy...')
    for x, y in validation_loader:
        output = model(x)
        _, y_hat = torch.max(output.data, 1)
        correct += (y_hat == y).sum().item()
    accuracy = correct / n_val_dataset
    return accuracy

def check_misclassified(model, validation_dataset, n_misclassified):
    '''
    Checks first n_misclassified
    '''
    count = 0
    print('\n-- First {} misclassified --'.format(n_misclassified))
    for x, y in DataLoader(dataset = validation_dataset, batch_size=1):
        output = model(x)
        _, y_hat=torch.max(output, 1)
        if y_hat != y:
            show_image((x,y))
            plt.show()
            print("Classified as: " + index_to_class(int(y_hat)) + 
                  ' (' + str(y_hat.item()) + ')')
            count += 1
        if count >= n_misclassified:
            break

def show_plots(epochs, loss_list, train_accuracy_list, val_accuracy_list):
    '''
    Plots loss and accuracy over epochs
    '''
    fig, ax1 = plt.subplots()
    
    ax1.plot(range(epochs), train_accuracy_list, label = 'Training Accuracy')
    ax1.plot(range(epochs), val_accuracy_list, label = 'Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    plt.legend(loc = 'best')
    
    ax2 = ax1.twinx()
    ax2.plot(range(epochs), loss_list, 'r-', label = 'Loss')
    ax2.set_ylabel('Loss')
    
    plt.legend(loc = 'best')
    plt.show()

#############
# CNN MODEL #
#############
model = nn.Sequential(OrderedDict([
        ('conv1',       nn.Conv2d(in_channels = 1, out_channels = 64,
                                  kernel_size = 5, stride = 1, padding = 2)),
        ('relu1',       nn.ReLU()),
        ('maxpool1',    nn.MaxPool2d(kernel_size = 2)),
        ('dropout1',    nn.Dropout(0.5)),
        ('conv2',       nn.Conv2d(in_channels = 64, out_channels = 32,
                                  kernel_size = 5, stride = 1, padding = 2)),
        ('relu2',       nn.ReLU()),
        ('maxpool2',    nn.MaxPool2d(kernel_size = 2)),
        ('dropout2',    nn.Dropout(0.5)),
        ('flatten',     Flatten()),
        ('dense1',      nn.Linear(32 * 7 * 7, 10))]))

###########
# TESTING #
###########
rand_num = np.random.randint(0,10)

# Training/model constants
#learning_rate = 0.1
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
epochs = 10

# Create lists for accuracy dictionary
loss_list = []
train_accuracy_list = []
val_accuracy_list = []

#############
# RUN MODEL #
#############
train_dataset, validation_dataset = create_datasets()
train_loader, validation_loader = create_loaders(train_dataset,
                                                 validation_dataset)
train_model(model, epochs, train_loader, validation_loader, optimizer,
            criterion, loss_list, train_accuracy_list, val_accuracy_list,
            n_train_dataset = len(train_dataset),
            n_val_dataset = len(validation_dataset))
final_accuracy = check_accuracy(model, validation_loader,
                                n_val_dataset = len(validation_dataset))
print('Accuracy on {} images after {} epochs: {}%'.
      format(len(validation_dataset), epochs, final_accuracy * 100))
check_misclassified(model, validation_dataset, n_misclassified = 5)
show_plots(epochs, loss_list, train_accuracy_list, val_accuracy_list)

##########################
# A COUPLE OF FUN CHECKS #
##########################
plot_parameters(model.state_dict()['conv1.weight'],
                'First Convolutional Weights')
plot_parameters(model.state_dict()['conv2.weight'],
                'Second Convolutional Weights')
plot_activations(model, train_dataset[2][0])

##### Accuracy on 10000 images after 10 epochs: 99.24% #####