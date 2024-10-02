# Image Segmentator
This project implement a image editor using AI, through the combination of image segmentation and image diffusion capabilities, by using two pre-trained models: 
**Segment Anything (Meta)** for image segmentation and **Stable Diffusion** for image diffusion and text-to-image generation.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Libraries](#libraries)
5. [Acknowledgments](#acknowledgments)

## Overview
This project utilizes two state-of-the-art models:
- **Segment Anything (Meta):** A highly versatile model for image segmentation, which was trained to find contours within images, generating then the called *maskes*. You can find more about it [here](https://github.com/facebookresearch/segment-anything).
- **Stable Diffusion:** A diffusion-based text-to-image generation model from Hugging Face, a diffusor model works by taking a real image and gradually add noise to it, then the training is based on the model trying to learn how to reconstruct the original image, we also can add a *condition*, being text-prompts or even images, which will guide the model denoising. You can find more about it [here](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting).

The goal is to apply image segmentation to selected areas and generate new content based on user inputs. The model was built using *Google Colab* to leverage it GPU's availabilities.  

## Installation
Instructions on how to install the project locally.

```bash
# Clone this repository
$ git clone https://github.com/yourusername/image-segmentationr.git

# Go into the repository
$ cd image-segmentation

# Install dependencies
$ pip install -r requirements.txt
```

## Usage
This model is pretty straightforward, you just need to:
1. Load a image;
2. SAM will then segment it and display the generated masks through the visualization function;
3. With the masks, you just need to select which one you want to edit;
4. And provide a list of text prompts which Stable Diffusion will use to generate the updateds;
5. You can experiment with the parameters:
```bash
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.99, # iou (intersection of union) evaluate how well the segmentation was done, so a lower value will result in more masks, but with lower accuracy
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)
image = pipe(
            prompt = inpainting_prompts[i],
            guidance_scale=7.5, # This control how closely the result image is to the prompt, so a lower value will allow more deviation from prompt (default = 7.5)
            num_inference_steps=60, # Control the number of steps between noise to result image, so more steps = high quality, but at cost of a slower model (default = 50)
            generator=generator,
            image=source_image,
            mask_image=stable_diffusion_mask
            ).images[0]
```

## Libraries
This project relies on the following Python libraries:
- **tqdm**: Progress bar for loops and iterations.
- **diffusers**: Pre-trained generative models for vision and audio.
- **transformers**: Pre-trained transformer models from Hugging Face.
- **accelerate**: Optimizes the efficiency of model training and inference.
- **xformers**: Memory and efficiency optimization for transformer-based models.
- **opencv**: Computer vision tools and image manipulation functions.
- **segment-anything**: Meta's image segmentation tool.
- **pycocotools, matplotlib, onnxruntime, onnx**: Additional utilities for handling image data and model operations.

## Acknowledgments
- This project was inspired by the course [*"Generative AI, from GANs to CLIP, with Python and Pytorch"*](https://www.udemy.com/course/generative-creative-ai-from-gans-to-clip-with-python-and-pytorch) by Javier Ideami, credits are due the author;
- All images and models credits go to their respective authors:
