"""
Convert an image using a trained SRGAN model
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# My stuff:
import time
import re


# Main script functionality
def main(opt):
    """
    opt is the result of ArgumentParser's parse_args()
    """

    outputDir = "processedOutput"
    os.makedirs(outputDir, exist_ok=True)

    print("----------------")
    print("Testing results:")
    print("----------------")
    
    cuda        = torch.cuda.is_available()

    hr_shape    = (opt.hr_height, opt.hr_width)


    # Count the number of unique residual layers mentioned in the generator state dict:
    generatorStateDict = torch.load(GetModelDataPath("generator")) # Load the max trained weights from the /saved_models directory
    resBlocks = {}
    for key in generatorStateDict:
        processedKey = re.split(r'^(res_blocks\.[0-9].)', key)
        if len(processedKey) > 1:
            resBlocks[processedKey[1]] = processedKey[1] # Insert an arbitrary entry: We just care about counting the unique keys

    num_residual_blocks = len(resBlocks)
    print("Counted " + str(num_residual_blocks) + " residual blocks in loaded generator state dict")


    # Initialize generator and discriminator
    generator           = GeneratorResNet(n_residual_blocks=num_residual_blocks)

    
    if cuda:
        print("Cuda is supported!!!")
        torch.cuda.empty_cache()

        generator           = generator.cuda()


    # Load pretrained models
    generator.load_state_dict(generatorStateDict)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


    #----------------
    # Process images:
    #----------------
    print("Processing images using the trained model:")

    torch.cuda.empty_cache()

    testStartTime   = time.time()
    totalTestTime   = 0
    numTests        = 0

    with torch.no_grad():   # Prevent OOM errors

        # Set models to eval mode, so batchnorm is disabled
        generator.eval()

        dataPath = GetDataPath(opt.valid_dataset_name)

        dataloader = DataLoader(
            ImageLoader(dataPath),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

        # Validate:
        for i, imgs in enumerate(dataloader):
            testStartTime = time.time()

            # Configure model input
            imgs_lr = Variable(imgs["img"].type(Tensor))

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # --------------
            #  Log Progress
            # --------------
            testTime = time.time() - testStartTime
            sys.stdout.write(
                "[Processed image %d/%d] [Test time: %fs]\n"
                % (i, len(dataloader), testTime)
            )
            

            gen_hr  = make_grid(gen_hr, nrow=1, normalize=True)

            save_image(gen_hr, GetArbitraryPath(outputDir) + ("0" if i < 10 else "") + "%d.png" % (i + 1), normalize=False)


            # Record the iteration time:
            totalTestTime = totalTestTime + testTime
            numTests = numTests + 1


        # ------------
        # Print stats:
        # ------------
        testTime = time.time() - testStartTime
        averageTestTime = totalTestTime / numTests

        print("\nTest results:\n-------------")
        print("Total test time = " + str(testTime) + " (secs) for " + str(len(dataloader.dataset)) + " test images")
        print("Average test time = " + str(averageTestTime) + " (secs)")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--epoch", type=int, default=-1, help="epoch to load testing weights of. Loads the highest if argument is < 0")

    parser.add_argument("--valid_dataset_name", type=str, default="testImages", help="name of the folder containing images to process")
    # parser.add_argument("--valid_dataset_name", type=str, default="Linnaeus 5 256X256_quick", help="name of the testing dataset")
    # parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to process at once")

    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    # parser.add_argument("--hr_height", type=int, default=64, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    # parser.add_argument("--hr_width", type=int, default=64, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")

    opt = parser.parse_args()
    print(opt)

    main(opt)
    