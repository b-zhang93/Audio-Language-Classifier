import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
import torchvision
import torchvision.transforms as tf
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#### Helper functions evaluate / score / fit our model + GPU loaders ####

def evaluate(model, dataloader, calc_loss=False, opt=None, grad_clip=None):
    '''
    Helper function to get accuracy and loss(optional) even do a step with the optimizer if present
    We can use this function in multiple ways depending on the parameters: just calculate accuracy and/or loss and/or gradients + update weights
    This way, we can utilize this function for both validation and train dataloaders for each epoch

    Inputs
    model = the model
    dataloader = validation or train or test dataloader
    calc_loss = set to True if you wish to calculate loss, else by default just returns empty list
    opt = set an optimizer function to calculate gradients and update weights, else None by default
    grad_clip = if not None, use gradient clipping, else default is None and it won't use gradient clipping. Controls for exploding / vanishing gradients.
    '''

    correct = 0
    total = 0
    losses = []

    if opt is not None:

        # calculate gradients and use optimizer
        for batch in dataloader:

            # forward pass through the model
            inputs, labels = batch
            outputs = model(inputs)

            # calculate loss if parameter set to True
            if (calc_loss==True):
                loss = model.loss(outputs, labels)   # Calculate loss
                losses.append(loss.item())

            # compute gradients
            loss.backward()
            # update params
            opt.step()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            # reset gradients
            opt.zero_grad()

            # Get the prediction of the net on the images
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)

            # Count those we got correct
            correct += torch.sum(predicted==labels).item()

        # calculate total correct cases
        accuracy = 100 * correct / total
        avg_loss = np.mean(losses)

    else:

        # no gradients calculated
        with torch.no_grad():
            for batch in dataloader:

                # forward pass through the model
                inputs, labels = batch
                outputs = model(inputs)

                # calculate loss if parameter set to True
                if (calc_loss==True):
                    loss = model.loss(outputs, labels)   # Calculate loss
                    losses.append(loss.item())

                # Get the prediction of the net on the images
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)

                # Count those we got correct
                correct += torch.sum(predicted==labels).item()

            # calculate total correct cases
            accuracy = 100 * correct / total
            avg_loss = np.mean(losses)

    return avg_loss, accuracy

def fit(epochs, model, train_loader, val_loader, grad_clip=False):
    ''' Function where we fit the model and record the metrics for each epoch with the inputted hyperparameters'''

    # Instantiate our model and empty lists to record metrics
    model.cuda() # move model to GPU'
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # train and validate for each epoch
    for epoch in range(epochs):

        ## Training Phase ##
        model.train()
        # run the evaluate function with optimizer to get train loss and accuracy
        train_loss, train_accuracy = evaluate(model, train_loader, calc_loss=True, opt=model.optimizer, grad_clip=grad_clip)

        # record training metrics
        train_losses.append(np.mean(train_loss))
        train_accuracies.append(np.mean(train_accuracy))

        ## Validation phase ##
        model.eval()
        # run the evaluate function on validation dataloader to get average loss and accuracy
        val_loss, val_accuracy = evaluate(model, val_loader, calc_loss=True)

        # record validation metrics
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # print the metrics
        print(f"Epoch: {epoch+1} / {epochs} | "\
            f"Avg Train Loss: {np.round(train_loss, 4)} | "\
            f"Train accuracy: {np.round(train_accuracy, 2)} | "\
            f"Avg Validation Loss: {np.round(val_loss, 4)} | "\
            f"Validation Accuracy: {np.round(val_accuracy, 2)}")

    return train_losses, train_accuracies, val_losses, val_accuracies


# create confusion matrix
def generate_cm(model, dataloader, classes):
    label_list = []
    prediction_list = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            if (torch.cuda.is_available()):
                torch.device('cuda')
                label_list.extend(list(labels.cpu().numpy()))
                prediction_list.extend(list(predicted.cpu().numpy()))
            else:
                label_list.extend(list(labels.numpy()))
                prediction_list.extend(list(predicted.numpy()))

    # calculate the confusion matrix
    cm = confusion_matrix(label_list, prediction_list, normalize="true")

    # plot the matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = disp.plot(xticks_rotation='vertical', ax=ax, cmap='BuGn')
    plt.show()

# plot accuracies
def plot_accuracy(train_accuracy, val_accuracy):
    plt.figure(figsize=(7,5))
    plt.plot(train_accuracy, '-x')
    plt.plot(val_accuracy, '-o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracies')
    plt.title('Accuracy vs Epochs')
    plt.show()

# plot losses
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(7,5))
    plt.plot(train_losses, '-x')
    plt.plot(val_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'])
    plt.title('Loss vs Epochs')
    plt.show()

# helper functions to use GPU
# these functions are from FreeCodeCamp's pytorch deeplearning tutorial guides
def get_device():
    '''Pick GPU, otherwise CPU if not avail'''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    '''Move tensors to the chosen device'''
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# using oop to set up a wrapper class for our dataloaders to use gpu
class DeviceDataLoader():
    '''Wraps a dataloader to move data into a device'''
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        '''Yields a batch of data after moving to device'''
        for i in self.dl:
            yield to_device(i, self.device)

    def __len__(self):
        '''Number of batches'''
        return len(self.dl)
