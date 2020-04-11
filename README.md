# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

This work is a modified fork of Erik Lindernoren's SRGAN implementation in the excellent PyTorch-GAN collection.
(https://github.com/eriklindernoren/PyTorch-GAN)

This repository was created as part of a research experiment exploring the influence that the depth of the discriminator network has on the performance of the SRGAN model. Weights are included for models trained on the Linnaeus 5 dataset of 256x256 images, with discriminator networks containing B=8, 12, 16, 20 residual blocks.

Download the Linnaeus 5 dataset (Chaladze, G. Kalatozishvili L. 2017.) here: http://chaladze.com/l5/
- You will need to rename the image files and flatted the directory structure so that all images are located in
the ./data/Linnaeus 5 256X256_train  and  ./data/Linnaeus 5 256X256_test/  directories, respectively.

The code has only been tested in a Linux environment. Use other operating systems at your own risk.


Instructions for training:
--------------------------
1. Ensure your dataset folder is located in the ./data/ directory
2. Launch the srgan.py script within your IDE of choice (eg. VSCode)
    OR
   Launch the srgan.py script via the command line with the following steps:
    2.1 Navigate to the ./implementation/srgan/ directory
    2.2 Execute the following command:
        python srgan.py

    Recommended: Use this command to view a list of optional command line parameters to control program execution:\
        python srgan.py --help


Instructions for using included pre-trained weights:
----------------------------------------------------
1. Ensure your images are located in the ./data/testImages/ subdirectory.
2. Copy the desired pre-trained weights and training state data files to the ./saved_models/ directory
3. Launch the processImage.py script within your IDE of choice (eg. VSCode)\
    OR\
   Launch the processImage.py script via the command line with the following steps:\
    3.1 Navigate to the ./implementation/srgan/ directory\
    3.2 Execute the following command:\
        python processImage.py\
	-> You may need to supply the location of your test images via command line arguments\

    Note: Use the following command to view a list of optional command line parameters to control program execution:\
        python processImage.py --help


File breakdown:
---------------
|File                                          	| Description |
|-----------------------------------------------|:---------------------------------------------------------------|
|/data/                                        	| Image data directory |
|/images/                                      	| Automatically created: Images are output here during training  |
|/implementation/srgan/dataset.py              	| Helper functions for loading and using datasets|
|/implementation/srgan/models.py		| SRGAN generator and discriminator network implementation|
|/implementation/srgan/processImage.py	       	| Test script: Super-resolve an image using pre-trained weights|
|/implementation/srgan/srgan.py			| Network training. This is the main program funcitonality|
|/implementation/srgan/validateGenerator.py	| Test script: Evaluate generator loss (ie. content loss) only|
|/implementation/srgan/validateModel.py		| Validation testing script. Called by srgan.py|
|/learned weights/8 residual blocks/		| Pre-trained network, for a generator with b=8 residual blocks|
|/learned weights/12 residual blocks/		| Pre-trained network, for a generator with b=12 residual blocks|
|/learned weights/16 residual blocks/		| Pre-trained network, for a generator with b=16 residual blocks|
|/learned weights/20 residual blocks/		| Pre-trained network, for a generator with b=20 residual blocks|
|/LICENCE					| Licence file, as provided by Erik Linernoren|
|/README.md					| This file|
|/saved_models/					| Automatically created: Saved weights/training states output here|
