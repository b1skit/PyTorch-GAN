"""
Validation testing for trained SRGAN models
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
    discriminator       = Discriminator(input_shape=(opt.channels, *hr_shape))
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
        discriminator       = discriminator.cuda()
        feature_extractor   = feature_extractor.cuda()
        criterion_GAN       = criterion_GAN.cuda()
        criterion_content   = criterion_content.cuda()

    # Load pretrained models
    generator.load_state_dict(generatorStateDict)
    discriminator.load_state_dict(torch.load(GetModelDataPath("discriminator", opt.epoch)))

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

    currentTestStartTime   = time.time()
    totalTestTime   = 0
    numTests        = 0

    with torch.no_grad():   # Prevent OOM errors

        # Set models to eval mode, so batchnorm is disabled
        generator.eval()
        discriminator.eval()

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
        avg_D_loss          = 0

        max_D_real_loss          = -1
        max_D_real_loss_index    = -1
        min_D_real_loss          = sys.maxsize
        min_D_real_loss_index    = -1
        avg_D_real_loss          = 0

        max_D_fake_loss          = -1
        max_D_fake_loss_index    = -1
        min_D_fake_loss          = sys.maxsize
        min_D_fake_loss_index    = -1
        avg_D_fake_loss          = 0

        max_G_loss          = -1
        max_G_loss_index    = -1
        min_G_loss          = sys.maxsize
        min_G_loss_index    = -1
        avg_G_loss          = 0

        max_G_adv_loss          = -1
        max_G_adv_loss_index    = -1
        min_G_adv_loss          = sys.maxsize
        min_G_adv_loss_index    = -1
        avg_G_adv_loss          = 0

        max_G_cont_loss         = -1
        max_G_cont_loss_index   = -1
        min_G_cont_loss         = sys.maxsize
        min_G_cont_loss_index   = -1
        avg_G_cont_loss         = 0

        # Validate:
        for i, imgs in enumerate(dataloader):
            currentTestStartTime = time.time()

            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # Adversarial ground truths
            valid   = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake    = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ---------------
            # Test Generator
            # ---------------

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            # Content loss
            gen_features    = feature_extractor(gen_hr)
            real_features   = feature_extractor(imgs_hr)
            loss_content    = criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            # -------------------
            # Test Discriminator
            # -------------------

            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            # Update log records
            if loss_G.item() > max_G_loss:
                max_G_loss          = loss_G.item()
                max_G_loss_index    = i
            if loss_G.item() < min_G_loss:
                min_G_loss          = loss_G.item()
                min_G_loss_index    = i
            if loss_D.item() > max_D_loss:
                max_D_loss          = loss_D.item()
                max_D_loss_index    = i
            if loss_D.item() < min_D_loss:
                min_D_loss          = loss_D.item()
                min_D_loss_index    = i

            if loss_content > max_G_cont_loss:
                max_G_cont_loss         = loss_content.item()
                max_G_cont_loss_index   = i
            if loss_content < min_G_cont_loss:
                min_G_cont_loss         = loss_content.item()
                min_G_cont_loss_index   = i

            if loss_GAN > max_G_adv_loss:
                max_G_adv_loss          = loss_GAN.item()
                max_G_adv_loss_index    = i
            if loss_GAN < min_G_adv_loss:
                min_G_adv_loss          = loss_GAN.item()
                min_G_adv_loss_index    = i

            if loss_real > max_D_real_loss:
                max_D_real_loss         = loss_real.item()
                max_D_real_loss_index   = i
            if loss_real < min_D_real_loss:
                min_D_real_loss         = loss_real.item()
                min_D_real_loss_index   = i

            if loss_fake > max_D_fake_loss:
                max_D_fake_loss         = loss_fake.item()
                max_D_fake_loss_index   = i
            if loss_fake < min_D_fake_loss:
                min_D_fake_loss         = loss_fake.item()
                min_D_fake_loss_index   = i

            avg_D_loss = avg_D_loss + loss_D.item()
            avg_G_loss = avg_G_loss + loss_G.item()

            avg_G_adv_loss  = avg_G_adv_loss + loss_GAN.item()
            avg_G_cont_loss = avg_G_cont_loss + loss_content.item()

            avg_D_fake_loss = avg_D_fake_loss + loss_fake.item()
            avg_D_real_loss = avg_D_real_loss + loss_real.item()

            # --------------
            #  Log Progress
            # --------------
            testTime = time.time() - currentTestStartTime
            sys.stdout.write(
                "[Test image %d/%d] [D loss: %f] [G loss: %f] [Test time: %fs]\n"
                % (i, len(dataloader), loss_D.item(), loss_G.item(), testTime)
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
        averageTestTime = totalTestTime / numTests

        avg_G_adv_loss  = avg_G_adv_loss / numTests
        avg_G_cont_loss = avg_G_cont_loss / numTests
        avg_G_loss      = avg_G_loss / numTests

        avg_D_real_loss = avg_D_real_loss / numTests
        avg_D_fake_loss = avg_D_fake_loss / numTests
        avg_D_loss      = avg_D_loss / numTests


        print("\nTest results:\n-------------")
        print("Total test time = " + str(totalTestTime) + " (secs) for " + str(len(dataloader.dataset)) + " test images")
        print("Average test time = " + str(averageTestTime) + " (secs)")

        totalTrainable = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print("\nNumber of trainable parameters in discriminator = " + str(totalTrainable))

        print("\nGenerator contains " + str(num_residual_blocks) + " residual blocks")

        totalTrainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        print("Number of trainable parameters in generator = " + str(totalTrainable))

        print("\nNote: Generator loss = contentLoss + 1e-3 * adversarialLoss")


        print("\nMin generator adverserial test loss = " + str(min_G_adv_loss) + ", at index " + str(min_G_adv_loss_index))
        print("Max generator adverserial test loss = " + str(max_G_adv_loss) + ", at index " + str(max_G_adv_loss_index))
        print("Avg generator adverserial test loss = " + str(avg_G_adv_loss))

        print("\nMin generator content test loss = " + str(min_G_cont_loss) + ", at index " + str(min_G_cont_loss_index))
        print("Max generator content test loss = " + str(max_G_cont_loss) + ", at index " + str(max_G_cont_loss_index))
        print("Avg generator content test loss = " + str(avg_G_cont_loss))

        print("\nMin generator total test loss = " + str(min_G_loss) + ", at index " + str(min_G_loss_index))
        print("Max generator total test loss = " + str(max_G_loss) + ", at index " + str(max_G_loss_index))
        print("Avg generator total test loss = " + str(avg_G_loss))


        print("\nMin discriminator real test loss = " + str(min_D_real_loss) + ", at index " + str(min_D_real_loss_index))
        print("Max discriminator real test loss = " + str(max_D_real_loss) + ", at index " + str(max_D_real_loss_index))
        print("Avg discriminator real test loss = " + str(avg_D_real_loss))

        print("\nMin discriminator fake test loss = " + str(min_D_fake_loss) + ", at index " + str(min_D_fake_loss_index))
        print("Max discriminator fake test loss = " + str(max_D_fake_loss) + ", at index " + str(max_D_fake_loss_index))
        print("Avg discriminator fake test loss = " + str(avg_D_fake_loss))

        print("\nMin discriminator total test loss = " + str(min_D_loss) + ", at index " + str(min_D_loss_index))
        print("Max discriminator total test loss = " + str(max_D_loss) + ", at index " + str(max_D_loss_index))
        print("Avg discriminator total test loss = " + str(avg_D_loss))        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=-1, help="epoch to load testing weights of. Loads the highest if argument is < 0")
    parser.add_argument("--valid_dataset_name", type=str, default="Linnaeus 5 256X256_test", help="name of the testing dataset")  
    parser.add_argument("--batch_size", type=int, default=1, help="size of the testing batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")

    opt = parser.parse_args()
    print(opt)

    main(opt)
    