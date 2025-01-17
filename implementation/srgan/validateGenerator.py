"""
Validation testing for trained SRGAN models

Tests the Generator model ONLY. The discriminator is not loaded.

NOTE: This means that only content loss is reported. Loss does NOT include the generator loss component
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

import time
import re


# Main script functionality
def main(opt):
    """
    opt is the result of ArgumentParser's parse_args()
    """

    os.makedirs("images", exist_ok=True)

    print("----------------")
    print("Testing results:")
    print("----------------")
    
    cuda        = torch.cuda.is_available()

    hr_shape    = (opt.hr_height, opt.hr_width)

    # Count the number of unique residual layers mentioned in the generator state dict:
    generatorStateDict = torch.load(GetModelDataPath("generator", opt.epoch))
    resBlocks = {}
    for key in generatorStateDict:
        processedKey = re.split(r'^(res_blocks\.[0-9].)', key)
        if len(processedKey) > 1:
            resBlocks[processedKey[1]] = processedKey[1] # Insert an arbitrary entry: We just care about counting the unique keys

    num_residual_blocks = len(resBlocks)
    print("Counted " + str(num_residual_blocks) + " residual blocks in loaded generator state dict")


    # Initialize generator and discriminator
    generator           = GeneratorResNet(n_residual_blocks=num_residual_blocks)

    feature_extractor   = FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN       = torch.nn.MSELoss()
    criterion_content   = torch.nn.L1Loss()

    if cuda:
        print("Cuda is supported!!!")
        torch.cuda.empty_cache()

        generator           = generator.cuda()
        # discriminator       = discriminator.cuda()
        feature_extractor   = feature_extractor.cuda()
        criterion_GAN       = criterion_GAN.cuda()
        criterion_content   = criterion_content.cuda()

    # Load pretrained models
    generator.load_state_dict(generatorStateDict)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Seed after network construction, and before DataLoader init for deterministic "random" data shuffling/consistent comparisons
    torch.backends.cudnn.deterministic  = True
    torch.backends.cudnn.benchmark      = False
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    # Restore the RNG states
    if opt.epoch < 0:
        LoadRandomState(GetHighestWeightIndex())
    else:
        LoadRandomState(opt.epoch)

    #----------------------------
    # Validate the trained model:
    #----------------------------
    print("Testing the trained model:")

    torch.cuda.empty_cache()

    testStartTime   = time.time()
    totalTestTime   = 0
    numTests        = 0

    with torch.no_grad():   # Prevent OOM errors

        # Set models to eval mode, so batchnorm is disabled
        generator.eval()

        dataPath = GetDataPath(opt.valid_dataset_name)

        dataloader = DataLoader(
            ImageDataset(dataPath, hr_shape=hr_shape),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_cpu,
        )

        # Initialize min/max result caches:
        max_D_loss          = -1
        max_D_loss_index    = -1
        min_D_loss          = sys.maxsize
        min_D_loss_index    = -1

        max_G_loss          = -1
        max_G_loss_index    = -1
        min_G_loss          = sys.maxsize
        min_G_loss_index    = -1

        # Validate:
        for i, imgs in enumerate(dataloader):
                testStartTime = time.time()

                # Configure model input
                imgs_lr = Variable(imgs["lr"].type(Tensor))
                imgs_hr = Variable(imgs["hr"].type(Tensor))

                # ---------------
                # Test Generator
                # ---------------

                # Generate a high resolution image from low resolution input
                gen_hr = generator(imgs_lr)

                # Content loss
                gen_features    = feature_extractor(gen_hr)
                real_features   = feature_extractor(imgs_hr)
                loss_content    = criterion_content(gen_features, real_features.detach())

                loss_G = loss_content

                # Update log records
                if loss_G.item() > max_G_loss:
                    max_G_loss          = loss_G.item()
                    max_G_loss_index    = i
                if loss_G.item() < min_G_loss:
                    min_G_loss          = loss_G.item()
                    min_G_loss_index    = i

                # --------------
                #  Log Progress
                # --------------
                testTime = time.time() - testStartTime
                sys.stdout.write(
                    "[Test image %d/%d] [Raw G loss: %f] [Test time: %fs]\n"
                    % (i, len(dataloader), loss_G.item(), testTime)
                )

                # Save image grid with upsampled inputs and SRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                gen_hr  = make_grid(gen_hr, nrow=1, normalize=True)
                imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)

                imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)

                img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)
                save_image(img_grid, GetImagesPath() + "test_%d.png" % i, normalize=False)

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

        print("Min generator test loss = " + str(min_G_loss) + ", at index " + str(min_G_loss_index))
        print("Max generator test loss = " + str(max_G_loss) + ", at index " + str(max_G_loss_index))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=-1, help="epoch to load testing weights of. Loads the highest if argument is < 0")
    parser.add_argument("--valid_dataset_name", type=str, default="Linnaeus 5 256X256_test", help="name of the testing dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the testing batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    opt = parser.parse_args()
    print(opt)

    main(opt)
    