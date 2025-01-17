"""
Helper functions for loading and using datasets.

Allows the main script to be launched directly by command line, or via an IDE
"""

import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import re       # Regex for path scrubbing
import pickle   # For load/save of random state
import random

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset): # Creates a map-style dataset
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)



class ImageLoader(Dataset): # Creates a map-style dataset
    def __init__(self, root):

        self.identityTransform = transforms.Compose(
            [ 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])

        img = self.identityTransform(img)

        return {"img": img}


    def __len__(self):
        return len(self.files)


def GetDataPath(dataset_name):
    """
    Helper function: Allow launching of the script from the project root directory (ie. within an IDE)

    Returns ../../data/DATASET_NAME, or ./data/DATASET_NAME, as appropriate
    """
    # Try from the implementation directory:
    dataPath = "../../data/%s" % dataset_name
    if not os.path.isdir(dataPath):
        # Try from the project root:
        dataPath = "./data/%s" % dataset_name
        if not os.path.isdir(dataPath):
            print("Error: Valid data path not found!")

    return dataPath


def GetModelPath():
    """
    Helper function: Get the relative /saved_models/ path

    Returns ../../saved_models/, or ./saved_models/, as appropriate
    """
    # Try from the implementation directory:
    modelPath = "../../saved_models/"
    if not os.path.isdir(modelPath):
        # Try from the project root:
        modelPath = "./saved_models/"
        if not os.path.isdir(modelPath):
            print("Error: Valid model path not found!")

    return modelPath


def GetImagesPath():
    """
    Helper function: Get the relative /images/ path

    Returns ../../images/, or ./images/, as appropriate
    """
    # Try from the implementation directory:
    imagesPath = "../../images/"
    if not os.path.isdir(imagesPath):
        # Try from the project root:
        imagesPath = "./images/"
        if not os.path.isdir(imagesPath):
            print("Error: Valid images path not found!")

    return imagesPath


def GetArbitraryPath(thePath):
    """
    Helper function: Get a relative arbitrary path

    Returns ../../thePath/, or ./thePath/, as appropriate
    """
    # Try from the implementation directory:
    arbitraryPath = "../../" + thePath + "/"
    if not os.path.isdir(arbitraryPath):
        # Try from the project root:
        arbitraryPath = "./" + thePath + "/"
        if not os.path.isdir(arbitraryPath):
            print("Error: Valid path not found!")

    return arbitraryPath


def GetHighestWeightIndex():
    """
    Helper function: Get the index of the highest weights file in the /saved_models directory
    """
    dataPath = GetModelPath()

    return GetHighestWeightIndexUsingPath(dataPath)


def GetHighestWeightIndexUsingPath(dataPath):
    """
    Helper function: Get the index of the highest weights file in the /saved_models directory.

    For efficiency, use this function if you've already got a copy of the current /saved_models directory
    """

    highestIndex = max([int(re.sub('[^0-9]','', f)) for f in os.listdir(dataPath)])

    return highestIndex


def GetModelDataPath(modelType, epoch = -1):
    """
    Helper function: Get the correct relative path of the saved model weights/biases

    modelType   = "generator" or "discriminator" (defaults to generator if errors occur)
    epoch       = Specific epoch to load. Loads the max if no valid epoch is supplied
    """

    dataPath = GetModelPath()
    
    # If no valid epoch is supplied, get the max:
    if epoch < 0:
        epoch = GetHighestWeightIndexUsingPath(dataPath)

    dataName = "generator_" + str(epoch) + ".pth"
    if modelType == "discriminator":
        dataName = "discriminator_" + str(epoch) + ".pth"

    finalPath = dataPath + dataName

    print("Using saved model path: \"" + finalPath + "\"")

    return finalPath


def LoadRandomState(stateNum):
    """
    Loads a previous random state from the saved_models directory
    """
    print("Loading random state")

    filename = 'rngState_' + str(stateNum) + '.pth'

    try:
        randomStates = pickle.load( open(GetModelPath() + filename, "rb") )

        random.setstate(randomStates["pythonRandom"])
        torch.set_rng_state(randomStates["torchRandom"])
        torch.cuda.set_rng_state(randomStates["torchCudaRandom"])
        np.random.set_state(randomStates["numpyRandom"])

    except:
        print("ERROR: Failed to load random state!")


def SaveRandomState(stateNum):
    """
    Save a random state checkpoint
    """
    randomStates = {
        "pythonRandom"      : random.getstate(), 
        "torchRandom"       : torch.get_rng_state(),
        "torchCudaRandom"   : torch.cuda.get_rng_state(),
        "numpyRandom"       : np.random.get_state()
    }

    filename = 'rngState_' + str(stateNum) + '.pth'

    savePath = GetModelPath() + filename

    pickle.dump(randomStates, open(savePath, "wb"))


def LoadTrainingTime(stateNum):
    """
    Load the number of seconds spent training
    """

    filename = 'time_' + str(stateNum) + '.pth'

    try:
        timeVals = pickle.load( open(GetModelPath() + filename, "rb"))
        return timeVals["trainingTime"]

    except:
        print("ERROR: Failed to load training times! Returning 0")
        return 0


def SaveTrainingTime(stateNum, seconds):
    """
    Save the number of seconds spent training
    """
    times = {
        "trainingTime" : seconds
    }

    filename = 'time_' + str(stateNum) + '.pth'
    savePath = GetModelPath() + filename

    pickle.dump(times, open(savePath, "wb"))