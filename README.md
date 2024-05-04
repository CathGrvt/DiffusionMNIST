# Custom Degradation DDPM

This repository contains a custom degradation strategy for training diffusion models (DDPM) using PyTorch. The custom degradation strategy replaces the standard Gaussian noise degradation with a pseudo QR code transformation. The goal is to explore the effectiveness of this alternative degradation strategy in generating high-fidelity images, particularly focusing on datasets like MNIST.

## Table of contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Credits](#credits)

## Introduction

Diffusion models are a class of generative models that learn a data distribution by modelling the diffusion process of a data sample. This repository explores a custom degradation strategy using pseudo QR codes instead of the traditional Gaussian noise. The aim is to explore whether this alternative degradation method can improve the fidelity of generated images.

## Requirements

- Python 3.9 or later
- PyTorch
- torchvision
- NumPy

## Setup

### Using Conda

Create and activate the Conda environment:

```bash
conda env create --name <env_name> -f environment.yml
conda activate <env_name>
```


Install dependencies if not using the Conda environment:

```bash
pip install -r requirements.txt
```

## Usage

To train the DDPM model with the pure custom degradation strategy, run:

```bash
python run_pure.py
```

To train the DDPM model with the MIX custom degradation strategy, run:

```bash
python run_mix.py
```

To train the DDPM model with the gaussian standard degradation strategy with all different set of hyperparameters, run:

```bash
python run_gaussian.py
```

Ensure that you have the required datasets and adjust hyperparameters as needed in the scripts.


### Documentation

- Auto-generated documentation using Doxygen. Run `doxygen` in the `/docs` folder to generate.
- This README serves as the front page for the documentation.

### Structure
The src directory contains the implementation of the DDPM and utility functions, while the notebooks directory provides Jupyter notebooks for experimentation and analysis. The data directory stores the MNIST dataset, and the papers directory includes relevant research papers. You can find the weights produced by the training on notebooks in the directory called "notebooks\notebooks_weights"


## Credits

- The implementation of the diffusion model and training pipeline is based on the original DDPM paper (Ho et al. 2020).

## Appendix: Use of Auto-generation Tools

Throughout the development of this project, ChatGPT by OpenAI was utilized for various purposes including code generation, prototyping, debugging assistance, and conceptual explanations. Below, I outline the instances of ChatGPT's use, detailing the nature of the prompts provided, the context in which the output was employed, and any modifications made to the generated content.

### Code Generation and Prototyping
- **Prompts Submitted**: Queries were made to generate Python code snippets for loading and processing DICOM images, understanding a 3D CNN model in PyTorch, if it was needed to modify the 2D CNN model, and writing custom dataset classes for PyTorch DataLoader.
- **Usage**: Generated code was used as a foundation for the project's data preprocessing pipeline and model implementation. This included loading DICOM images, converting them to tensors, and defining the neural network architecture.
- **Modifications**: The code provided by ChatGPT was adapted to fit the specific requirements of the dataset and project objectives, on its own it didn't function with the provided dataset and context. This involved adjusting data loading mechanisms from my part to handle the dataset's unique structure, and optimizing performance.

### Debugging Assistance
- **Prompts Submitted**: Assistance was requested for debugging issues related to Docker file, Custom Dataset, DataLoader, including errors with variable image sizes and tensor stacking.
- **Usage**: Suggestions from ChatGPT were employed to resolve runtime errors and improve the data loading process.
- **Modifications**: Debugging advice was integrated with existing code, with adjustments made to accommodate the specific data formats and processing goals.

### Drafting and Proofreading
- **Prompts Submitted**: Requests were made for drafting sections of the README file and technical documentation, as well as for proofreading and suggesting alternative wordings.
- **Usage**: The output was utilised to enhance the clarity and completeness of project documentation.
- **Modifications**: Generated text was revised to better align with the coursework's scope, terminology, and presentation style.

### Co-Pilot
- **Usage**: GitHub Copilot was used for code suggestion, completion, and documentation.
