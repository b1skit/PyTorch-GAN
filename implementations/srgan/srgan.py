"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
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


#TODO:
# Branch for different network configs
# How do noise filtering models work? Might be worth adding some layers similar to that?
# Speed: 
    # Tune batch size to max GPU mem usage (nvidia-smi)
# Break testing out into a seperate file   




os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
# parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--epoch", type=int, default=2, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
# parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
# parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
# parser.add_argument("--train_dataset_name", type=str, default="Linnaeus 5 256X256_train", help="name of the training dataset")
parser.add_argument("--train_dataset_name", type=str, default="Linnaeus 5 256X256_quick", help="name of the training dataset")
parser.add_argument("--valid_dataset_name", type=str, default="Linnaeus 5 256X256_test", help="name of the testing dataset")
# parser.add_argument("--valid_dataset_name", type=str, default="Linnaeus 5 256X256_quick", help="name of the testing dataset")
# parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--batch_size", type=int, default=8, help="size of the training batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
# parser.add_argument("--lr", type=float, default=0.000283, help="adam: learning rate") # Multiply batch size by k: Multiply learning rate by sqrt(k)
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
# parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving image samples")
# parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator           = GeneratorResNet()
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

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(GetModelDataPath("generator", opt.epoch - 1)))
    discriminator.load_state_dict(torch.load(GetModelDataPath("discriminator", opt.epoch - 1)))


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# Schedule learning rate:
G_scheduler     = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size = opt.decay_epoch, gamma = 0.1)
D_scheduler     = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size = opt.decay_epoch, gamma = 0.1)

# Start the optimizer from the correct point:
if opt.epoch != 0:
    for _ in range(0, opt.epoch - 1):
        G_scheduler.step()
        D_scheduler.step()   
    
print("Scheduled learning rate for decay at epoch " + str(opt.decay_epoch))


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


# Seed after network construction, and before DataLoader init for deterministic "random" data shuffling/consistent comparisons
torch.backends.cudnn.deterministic  = True
torch.backends.cudnn.benchmark      = False
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

# Restore the RNG states if we're resuming training
if opt.epoch != 0:
    LoadRandomState(opt.epoch - 1)
   



dataPath = GetDataPath(opt.train_dataset_name)

dataloader = DataLoader(
    ImageDataset(dataPath, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)


# ----------
#  Training
# ----------
trainingStartTime = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        epochStartTime = time.time()

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid   = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake    = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad() # Clear the last step

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

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad() # Clear the last step

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        epochTime = time.time() - epochStartTime
        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Epoch time: %fs] [D learn: %f] [G learn: %f]\n"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(), epochTime, optimizer_D.param_groups[0]['lr'], optimizer_G.param_groups[0]['lr'])
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)

            imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)

            img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    
    # Step the scheduler once per epoch
    G_scheduler.step()
    D_scheduler.step()

    if opt.checkpoint_interval != -1 and (epoch % opt.checkpoint_interval == 0 or epoch == opt.n_epochs - 1):
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)

        # Save RNG state:
        SaveRandomState(epoch)



# Validate the trained model:
#----------------------------
print("Testing the trained model:")

torch.cuda.empty_cache()

with torch.no_grad():   # Prevent OOM errors

    generator.load_state_dict(torch.load(GetModelDataPath("generator")))
    discriminator.load_state_dict(torch.load(GetModelDataPath("discriminator")))

    # Set models to eval mode, so batchnorm is disabled
    generator.eval()
    discriminator.eval()

    generator           = generator.cuda()
    discriminator       = discriminator.cuda()
    feature_extractor   = feature_extractor.cuda()
    criterion_GAN       = criterion_GAN.cuda()
    criterion_content   = criterion_content.cuda()

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

            # Adversarial ground truths
            valid   = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake    = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ---------------
            # Test Generator
            # ---------------

            # optimizer_G.zero_grad() # Clear the last step

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

            # loss_G.backward()
            # optimizer_G.step()

            # -------------------
            # Test Discriminator
            # -------------------

            # optimizer_D.zero_grad() # Clear the last step

            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            # loss_D.backward()
            # optimizer_D.step()

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

            # --------------
            #  Log Progress
            # --------------
            testTime = time.time() - testStartTime
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
            save_image(img_grid, "images/test_%d.png" % i, normalize=False)


    # Print stats:
    print("\nTraining results:\n-----------------")
    trainingEndTime = time.time()
    print("Total training time (secs) = " + str(trainingEndTime - trainingStartTime))

    print("\nTest results:\n-------------")
    
    print("Min generator test loss = " + str(min_G_loss) + ", at index " + str(min_G_loss_index))
    print("Max generator test loss = " + str(max_G_loss) + ", at index " + str(max_G_loss_index))

    print("Min discriminator test loss = " + str(min_D_loss) + ", at index " + str(min_D_loss_index))
    print("Max discriminator test loss = " + str(max_D_loss) + ", at index " + str(max_D_loss_index))
    

    print("\nNetwork stats:\n--------------")

    totalTrainable = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print("Number of trainable parameters in generator = " + str(totalTrainable))

    totalTrainable = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print("Number of trainable parameters in discriminator = " + str(totalTrainable))

    print("Generator:\n----------")
    print(generator)    

    print("Discriminator:\n--------------")
    print(discriminator)

    