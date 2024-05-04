from src.models import CNN, QR_DDPM
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
    The `main` function trains and evaluates one custom degradation deep learning model (QR_DDPM)
    on MNIST dataset, including visualization, loss plotting, FID score calculation using
    different methods, and custom degradation model training.
    This is based on a degradation strategy using pseudo-qr codes.
    """
    mnist_dataloader = mnist_preprocessing()

    # dataloader for comparing during training (FID scores)
    test_dataloader = mnist_preprocessing(batch_size=64)
    test, _ = next(iter(test_dataloader))

    # Initialise pure custom degradation model with parameters that worked best
    gt = CNN(
        in_channels=1,
        expected_shape=(28, 28),
        n_hidden=(16, 32, 64, 32, 16),
        act=nn.GELU,
    )
    ddpm = QR_DDPM(gt=gt, n_T=1000)
    optim = torch.optim.Adam(
        ddpm.parameters(), lr=1e-4, weight_decay=1e-5
    )  # Should be lower than 1e-4
    accelerator = Accelerator()
    ddpm, optim, mnist_dataloader = accelerator.prepare(ddpm, optim, mnist_dataloader)

    directory = create_directory("custom_pure_model")

    # # Train the model
    train(ddpm, mnist_dataloader, test, accelerator, optim, directory)

    # # Plots of training
    plot_losses(os.path.join(directory, "./ddpm_losses.pt"), accelerator.device)
    plot_fid_scores(os.path.join(directory, "./fid_scores.pt"), accelerator.device)

    # Analysis
    weights_path = os.path.join(directory, "ddpm_mnist.pth")
    real_images, gen_images = prepare_data_calculation(
        ddpm, mnist_dataloader, accelerator.device, weights_path
    )
    # FID score using Bansal's functions
    try:
        final_fid = calculate_fid_given_samples(
            [real_images, gen_images], device=accelerator.device
        )
        print("Final FID score: ", final_fid)
    except Exception as e:
        print(e)
        pass

    # FID score using torchmetrics
    fid = calculate_fid_torchmetrics(real_images, gen_images)
    print("Final FID score via torchmetrics: ", fid)
    # FID score using my own custom function
    fid_custom = calculate_fid_custom(real_images, gen_images)
    print("Final FID score via custom function: ", fid_custom)


if __name__ == "__main__":
    main()
