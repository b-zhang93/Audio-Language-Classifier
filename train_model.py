## Run this script to train the model with the train/validation/test data we generated with the previous scripts ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import torchvision.transforms as tf
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import CNN_model_3
fom helpers import *

##########################################################
# Step 1: Instantiate some variables and our model       #
##########################################################
# set our seed for repreoducibility
random_seed = 10
# our batch size for training
batch_size=64
# Instantiate our model with the optimal parameters (refer to notebooks)
model = CNN_model_3(opt_fun=torch.optim.Adam, lr=0.001)


##########################################################
# Step 2: Load in our data and prepare it for training   #
##########################################################
# our transformations when importing the images
transformations = tf.Compose([tf.Resize([64,64]), tf.ToTensor()])
# load in train / test data
trainset = ImageFolder('data/train', transform=transformations)
testset = ImageFolder('data/test', transform=transformations)
# record the classes
classes = trainset.classes

# Perform a Train / Validation Split 80/20
torch.manual_seed(random_seed) # setting the seed
val_size = int(len(trainset)*0.20) # val length
train_size = len(trainset) - val_size # train length
# using pytorch's built in random_split method to split our training data
train_ds, val_ds = random_split(trainset, [train_size, val_size])

# intialize our data loader to feed batches into the model
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4)


##########################################################
# Step 3: Move our models and data to the GPU if avail   #
##########################################################
device = get_device() # use GPU if it is available, otherwise use CPU

# if we are using GPU, then move data loaders and model to GPU as well
if torch.cuda.is_available():
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    model.cuda()


##########################################################
# Step 4: Training the model                             #
##########################################################
# fit the model
history = fit(epochs=20, model=model, train_loader=train_dl, val_loader=val_dl)
# record metrics
train_losses, train_accuracies, val_losses, val_accuracies = history


##########################################################
# Step 5: Score the model and output performance metrics #
##########################################################
# load our test data loader onto GPU
test_dl = DeviceDataLoader(DataLoader(testset, batch_size*2), device)
# calculate the loss and accuracy
test_loss, test_score =  evaluate(model, dataloader=test_dl, calc_loss=True)
# output the performance
print(f"Test Score: {test_score:.2f}% | Test Loss: {test_loss:.4f}") # print test metrics
print("")
print("Training Metrics:")
plot_accuracy(train_accuracies, val_accuracies)
plot_losses(train_losses, val_losses)
print("")
print('Confusion Matrix:')
generate_cm(model, test_dl, classes)


##########################################################
# Step 6: Save our model                                 #
##########################################################
torch.save(model.state_dict(), "cnn_model_trained.pt")
print("Model has been saved as 'cnn_model_trained.pt'. Use torch.load to load the saved model")
