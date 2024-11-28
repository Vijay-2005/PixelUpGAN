# Conditional GAN for Super-Resolution with Perceptual Loss

This repository implements a Conditional GAN (CGAN) for super-resolution of images. The model generates high-resolution (HR) images from low-resolution (LR) inputs by utilizing perceptual loss with VGG19 features to enhance the visual quality of the outputs.

---

## Table of Contents

1. [Features](#features)
2. [Setup Instructions](#setup-instructions)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Model Details](#model-details)
6. [Callbacks](#callbacks)
7. [References](#references)

---

## Features

- **Data Preprocessing**: 
  - Convert high-resolution images to low-resolution versions by downscaling and upscaling.
  - Normalize image data to [0, 1] for improved training stability.
  
- **Generator**: 
  - A deep residual network with skip connections for better HR image reconstruction.

- **Discriminator**: 
  - A convolutional network that distinguishes between real and generated HR images.

- **Perceptual Loss**: 
  - Uses features extracted from the `block5_conv4` layer of a pre-trained VGG19 network to calculate content loss.

- **Callbacks**: 
  - Save intermediate generated images and model weights during training at specified intervals.

---

## Setup Instructions

### Prerequisites
Ensure the following are installed:
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

### Installation Steps
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Vijay-2005/PixelUpGAN
   cd PixelUpGAN
2. Install the required Python packages:
  ```bash
  pip install -r requirements.txt
3. Prepare the dataset:
- Download high-resolution images(from DIV2K Dataset).
- Place the images in the directory specified by HR_PATH in the script (default: /content/drive/MyDrive/DIV2K_train_HR).