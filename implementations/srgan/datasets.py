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


class ImageDataset(Dataset):
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



def GetDataPath(dataset_name):
    """
    Helper function: Allow launching of the script from the project root directory (ie. within an IDE)
    """
    # Try from the implementations directory:
    dataPath = "../../data/%s" % dataset_name
    if not os.path.isdir(dataPath):
        print("Couldn't find path \"" + dataPath + "\", trying alternative")

        # Try from the project root:
        dataPath = "./data/%s" % dataset_name
        if os.path.isdir(dataPath):
            print("Alternative path \""+ dataPath + "\" found")
        else:
            print("Valid data path not found!")
    
    print("Using data path: \"" + dataPath + "\"")

    return dataPath



def GetModelDataPath(modelType, epoch = -1):
    """
    Helper function: Get the correct relative path of the saved model weights/biases

    modelType = "generator" or "discriminator" (defaults to generator if errors occur)
    epoch = specific epoch to load. Loads the max if no valid epoch is supplied
    """

    # Try from the implementations directory:
    dataPath = "../../saved_models/"
    if not os.path.isdir(dataPath):
        print("Couldn't find path \"" + dataPath + "\", trying alternative")

        # Try from the project root:
        dataPath = "./saved_models/"
        if os.path.isdir(dataPath):
            print("Alternative path \""+ dataPath + "\" found")
        else:
            print("Valid data path not found!")
    
    # If no valid epoch is supplied, get the max:
    if epoch < 0:
        epoch = max([int(re.sub('[^0-9]','', f)) for f in os.listdir(dataPath)])

    dataName = "generator_" + str(epoch) + ".pth"
    if modelType == "discriminator":
        dataName = "discriminator_" + str(epoch) + ".pth"

    finalPath = dataPath + dataName

    print("Using saved model path: \"" + finalPath + "\"")

    return finalPath


def GetRandomSavePath():
    """
    Helper function: Get the relative path for saving RNG states
    """
    # Try from the implementations directory:
    dataPath = "../../saved_models/"
    if not os.path.isdir(dataPath):
        print("Couldn't find RNG path \"" + dataPath + "\", trying alternative")

        # Try from the project root:
        dataPath = "./saved_models/"
        if os.path.isdir(dataPath):
            print("Alternative RNG path \""+ dataPath + "\" found")
        else:
            print("Valid RNG path not found!")
    
    print("Using RNG state directory: \"" + dataPath + "\"")

    return dataPath


def LoadRandomState(stateNum):
    print("Loading random state")

    filename = 'rngState_' + str(stateNum)

    try:
        randomStates = pickle.load( open(GetRandomSavePath() + filename, "rb") )

        random.setstate(randomStates["pythonRandom"])
        torch.set_rng_state(randomStates["torchRandom"])
        torch.cuda.set_rng_state(randomStates["torchCudaRandom"])
        np.random.set_rng_state(randomStates["numpyRandom"])

    except:
        print("Failed to load random state!")


def SaveRandomState(stateNum):
    randomStates = {
        "pythonRandom"      : random.getstate(), 
        "torchRandom"       : torch.get_rng_state(),
        "torchCudaRandom"   : torch.cuda.get_rng_state(),
        "numpyRandom"       : np.random.get_state()
    }

    filename = 'rngState_' + str(stateNum)
    savePath = GetRandomSavePath() + filename

    pickle.dump(randomStates, open(savePath, "wb"))