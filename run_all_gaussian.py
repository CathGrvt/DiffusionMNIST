from src.models import CNN, DDPM
from src.utilities import (
    train,
    create_directory,
    mnist_preprocessing,
    plot_fid_scores,
    plot_losses,
    prepare_data_calculation,
    calculate_fid_torchmetrics,
    calculate_fid_custom,
)
import torch
from torch import nn
from accelerate import Accelerator
import os
from src.bansal_fucntions import calculate_fid_given_samples


def main():
    """
    The main function trains and analyzes Deep Generative Models (DDPM) on MNIST dataset with different
    hyperparameters and model configurations, evaluating FID scores using various methods.
    """
    mnist_dataloader = mnist_preprocessing()

    # dataloader for comparing during training (FID scores)
    test_dataloader = mnist_preprocessing(batch_size=64)
    test, _ = next(iter(test_dataloader))

    # Initialise starter model with default parameters
    gt = CNN(
        in_channels=1, expected_shape=(28, 28), n_hidden=(16, 32, 32, 16), act=nn.GELU
    )
    ddpm = DDPM(gt=gt, betas=(1e-4, 0.02), n_T=1000)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    accelerator = Accelerator()
    ddpm, optim, mnist_dataloader = accelerator.prepare(ddpm, optim, mnist_dataloader)
    directory = create_directory("starter_model")

    # Train the model
    train(ddpm, mnist_dataloader, test, accelerator, optim, directory)

    # Plots of training
    plot_losses(os.path.join(directory, "./ddpm_losses.pt"), accelerator.device)
    plot_fid_scores(os.path.join(directory, "./fid_scores.pt"), accelerator.device)

    # Analysis
    weights_path = os.path.join(directory, "./ddpm_mnist.pth")
    real_images, gen_images = prepare_data_calculation(
        ddpm, mnist_dataloader, accelerator.device, weights_path
    )
    # FID score using Bansal's functions
    final_fid = calculate_fid_given_samples(
        [real_images, gen_images], device=accelerator.device
    )
    print("Final FID score: ", final_fid)
    # FID score using torchmetrics
    fid = calculate_fid_torchmetrics(real_images, gen_images)
    print("Final FID score via torchmetrics: ", fid)
    # FID score using my own custom function
    fid_custom = calculate_fid_custom(real_images, gen_images)
    print("Final FID score via custom function: ", fid_custom)

    # Initialise starter model with hyperparmeters set 2
    gt = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(32, 64, 128, 64, 32),
        act=nn.LeakyReLU,
    )
    ddpm = DDPM(gt=gt, betas=(0.001, 0.015), n_T=1000)
    optim = torch.optim.Adam(ddpm.parameters(), lr=0.0005)

    directory = create_directory("set2_model")

    # Train the model
    train(ddpm, mnist_dataloader, test, accelerator, optim, directory)

    # Plots of training
    plot_losses(os.path.join(directory, "./ddpm_losses.pt"), accelerator.device)
    plot_fid_scores(os.path.join(directory, "./fid_scores.pt"), accelerator.device)

    # Analysis
    weights_path = os.path.join(directory, "./ddpm_mnist.pth")
    real_images, gen_images = prepare_data_calculation(
        ddpm, mnist_dataloader, accelerator.device, weights_path
    )
    # FID score using Bansal's functions
    final_fid = calculate_fid_given_samples(
        [real_images, gen_images], device=accelerator.device
    )
    print("Final FID score: ", final_fid)
    # FID score using torchmetrics
    fid = calculate_fid_torchmetrics(real_images, gen_images)
    print("Final FID score via torchmetrics: ", fid)
    # FID score using my own custom function
    fid_custom = calculate_fid_custom(real_images, gen_images)
    print("Final FID score via custom function: ", fid_custom)

    # Initialise starter model with hyperparmeters set 3
    # Adjusting the CNN model to find a balance between capacity and efficiency
    gt = CNN(
        in_channels=1, expected_shape=(28, 28), n_hidden=(64, 96, 96, 64), act=nn.ELU
    )
    # Adjusting the DDPM model parameters to explore a balanced noise schedule
    ddpm = DDPM(gt=gt, betas=(0.0005, 0.02), n_T=1000)
    # A starting beta of 0.0005 with an end of 0.02 proposes a gradual but not too aggressive noise introduction.
    # Setting the optimizer with a moderate learning rate, aiming for a balance between convergence speed and stability
    optim = torch.optim.Adam(ddpm.parameters(), lr=0.0003)
    directory = create_directory("set3_model")

    # Train the model
    train(ddpm, mnist_dataloader, test, accelerator, optim, directory)

    # Plots of training
    plot_losses(os.path.join(directory, "ddpm_losses.pt"), accelerator.device)
    plot_fid_scores(os.path.join(directory, "fid_scores.pt"), accelerator.device)

    # Analysis
    weights_path = os.path.join(directory, "ddpm_mnist.pth")
    real_images, gen_images = prepare_data_calculation(
        ddpm, mnist_dataloader, accelerator.device, weights_path
    )
    # FID score using Bansal's functions
    final_fid = calculate_fid_given_samples(
        [real_images, gen_images], device=accelerator.device
    )
    print("Final FID score: ", final_fid)
    # FID score using torchmetrics
    fid = calculate_fid_torchmetrics(real_images, gen_images)
    print("Final FID score via torchmetrics: ", fid)
    # FID score using my own custom function
    fid_custom = calculate_fid_custom(real_images, gen_images)
    print("Final FID score via custom function: ", fid_custom)


if __name__ == "__main__":
    main()
