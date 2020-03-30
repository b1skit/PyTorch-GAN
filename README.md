Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

This work is a modified fork of Erik Lindernoren's SRGAN implementation in the excellent PyTorch-GAN collection.
(https://github.com/eriklindernoren/PyTorch-GAN)

Download the Linnaeus 5 dataset (Chaladze, G. Kalatozishvili L. 2017.) here: http://chaladze.com/l5/
- You will need to rename the image files and flatted the directory structure so that all images are located in
the ./data/Linnaeus 5 256X256_train  and  ./data/Linnaeus 5 256X256_test/  directories, respectively.

Instructions for use:
1. Ensure your dataset folder is located in the ./data/ directory
2. Launch this script within your IDE of choice (eg. VSCode)
    OR
   Launch via the command line with the following steps:
    2.1 Navigate to the ./implementation/srgan/ directory
    2.2 Execute the following command:
        python srgan.py

    Note: Use the following command to view a list of option command line parameters to control program execution:
        python srgan.py --help
