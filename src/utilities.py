from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.transforms import functional as TF
import os
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm


def mnist_preprocessing(batch_size=128):
    """
    The function `mnist_preprocessing` preprocesses the MNIST dataset for training by applying
    transformations and creating a DataLoader with specified batch size and settings.

    @param batch_size The `batch_size` parameter in the `mnist_preprocessing` function is used to
    specify the number of samples in each batch of data during training or inference. In this case, the
    default value for `batch_size` is set to 128 if no value is provided when calling the function.

    @return A DataLoader object for the MNIST dataset with the specified batch size and preprocessing
    steps.
    """
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))]
    )
    dataset = MNIST("./data", train=True, download=False, transform=tf)
    dataloader = DataLoader(
        dataset, batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    return dataloader


def first_visualisation_custom(ddpm, dataloader, accelerator, directory):
    """
    The function `first_visualisation_custom` generates visualizations using a given model `ddpm` and
    data loader `dataloader`.

    @param ddpm
    @param dataloader
    @param accelerator (CPU or GPU)
    @param directory path to the directory where you want the images to be stored
    """
    x, _ = next(iter(dataloader))
    with torch.no_grad():
        ddpm(x)
        B, C, H, W = x.shape

        filter = ddpm.generate_degradation_noise((C, H, W), B).to(accelerator.device)
        save_image(filter, os.path.join(directory, "degradation_filter.png"))

        t = torch.randint(1, ddpm.n_T, (x.shape[0],), device=x.device)
        alpha_t = ddpm.alpha_t[t, None, None, None]  # Get right shape for broadcasting
        transformed_x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * filter
        save_image(transformed_x, os.path.join(directory, "transformed_mnist.png"))

        model_estimation = ddpm.gt(transformed_x, t / ddpm.n_T)
        save_image(model_estimation, os.path.join(directory, "model_estimation.png"))

        xh = ddpm.sample(1, (1, 28, 28), accelerator.device)
        # Save samples to `./contents` directory
        save_image(xh, os.path.join(directory, "model_sample.png"))


def reformat(dataset):
    """
    The function `reformat` takes a dataset of images, resizes them to 299x299 pixels, converts
    grayscale images to RGB, and stacks them into a single tensor.

    @param dataset

    @return The function `reformat` returns a single tensor containing all processed images stacked
    along the 0th dimension.
    """
    processed_imgs = []
    for img in dataset:
        resized_img = TF.resize(img, size=(299, 299))
        # Convert grayscale to RGB by repeating the grayscale channel 3 times
        rgb_img = resized_img.repeat(1, 3, 1, 1)
        processed_imgs.append(rgb_img)
    # Stack all processed images into a single tensor
    return torch.cat(processed_imgs, dim=0).cpu()


def create_directory(directory):
    """
    The function `create_directory` creates a directory if it does not already exist.

    @param directory the path of the directory to be created.

    @return  the directory path that was created or already existed.
    """
    os.makedirs(directory, exist_ok=True)
    return directory


def train(ddpm, dataloader, real_images, accelerator, optim, directory):
    """
    This function trains a model using a given data loader and optimizer, saves model checkpoints and
    loss/fidelity scores, and generates samples during training.

    @param ddpm
    @param dataloader
    @param real_images
    @param accelerator T
    @param optim  optimizer object used to update the parameters of the model during training.

    @param directory the directory path where the trained model, loss history, and FID scores will be saved during the
    training process.
    """
    n_epoch = 100
    losses = []
    fid_scores = []

    for i in range(n_epoch):
        ddpm.train()

        pbar = tqdm(dataloader)  # Wrap our loop with a visual progress bar
        for x, _ in pbar:
            optim.zero_grad()

            loss = ddpm(x)

            # loss.backward()
            # ^Technically should be `accelerator.backward(loss)` but not necessary for local training
            accelerator.backward(loss)

            losses.append(loss.item())
            avg_loss = np.average(losses[min(len(losses) - 100, 0) :])
            pbar.set_description(
                f"loss: {avg_loss:.3g}"
            )  # Show running average of loss in progress bar

            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(
                64, (1, 28, 28), accelerator.device
            )  # Can get device explicitly with `accelerator.device`
            save_image(xh, f"./contents/ddpm_sample_{i:04d}.png")

            try:
                fid = calculate_fid_custom(reformat(real_images), reformat(xh))
            except Exception:
                fid = None
            fid_scores.append(fid)
            # Save samples to `./contents` directory
            # save model
            torch.save(ddpm.state_dict(), os.path.join(directory, "./ddpm_mnist.pth"))
            torch.save(losses, os.path.join(directory, "./ddpm_losses.pt"))
            torch.save(fid_scores, os.path.join(directory, "./fid_scores.pt"))


def plot_fid_scores(fid_scores_path, device):
    """
    The function `plot_fid_scores` loads FID scores from a file, plots them against epochs, and displays
    the FID curve.

    @param fid_scores_path file path to the saved FID scores that
    you want to plot.
    @param device
    """
    fid_scores = torch.load(fid_scores_path, map_location=torch.device(device))
    plt.plot(fid_scores)
    plt.xlabel("Epoch")
    plt.ylabel("FID score")
    plt.title("FID Curve")
    plt.show()


def plot_losses(losses_path, device):
    """
    The function `plot_losses` loads and plots losses from a file, displaying both iteration-wise and
    epoch-wise loss curves.

    @param losses_path file path to the saved losses data that you want to plot.
    @param device
    """
    losses = torch.load(losses_path, map_location=torch.device(device))
    plt.plot(losses)
    plt.axis([0, 5000, 0, 1.08])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()

    epoch_losses = []
    for i in range(0, len(losses), 467):
        avg_loss = np.average(losses[i : i + 467])
        epoch_losses.append(avg_loss)

    plt.plot(epoch_losses)
    plt.axis([0, 100, 0, 0.05])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()


def prepare_data_calculation(ddpm, mnist_dataloader, device, weights_path):
    """
    The function prepares data for calculation by loading weights, generating images, and processing
    them for a comparison task.

    @param ddpm
    @param mnist_dataloader data loader object that provides batches of
    MNIST dataset images for processing.
    @param device T
    @param weights_path file path to the saved weights of the model `ddpm`
    @return `real_images` and `generated_images`.
    """
    ddpm.to(device)
    ddpm.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    ddpm.eval()

    with torch.no_grad():
        generated_dataset = ddpm.sample(128, (1, 28, 28), device)

    real_mnist, _ = next(iter(mnist_dataloader))
    real_mnist = real_mnist.to(device)

    processed_mnist = []
    for mnist in real_mnist:
        resized_mnist = TF.resize(mnist, size=(299, 299))
        rgb_mnist = resized_mnist.repeat(1, 3, 1, 1)
        processed_mnist.append(rgb_mnist)
    real_images = torch.cat(processed_mnist, dim=0)

    processed_imgs = []
    for img in generated_dataset:
        resized_img = TF.resize(img, size=(299, 299))
        rgb_img = resized_img.repeat(1, 3, 1, 1)
        processed_imgs.append(rgb_img)
    imgs = torch.cat(processed_imgs, dim=0)

    real_images, generated_images = real_images.cpu(), imgs.cpu()
    return real_images, generated_images


def calculate_fid_torchmetrics(real_images, generated_images):
    fid = FrechetInceptionDistance(feature=64)
    fid.reset()
    fid.update(real_images.cpu().to(dtype=torch.uint8), real=True)
    fid.update(generated_images.to(dtype=torch.uint8), real=False)
    fid.compute()


def calculate_fid_custom(real_images, gen_images):
    """
    The function `calculate_fid_custom` calculates the Fréchet Inception Distance (FID) between real and
    generated images using features extracted from a pretrained Inception v3 model.

    @param real_images MNIST
    @param gen_images

    @return FID score

    """
    # Load the pretrained Inception v3 model
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    inception_model.eval()
    # Modify the model to return features from an intermediate layer
    inception_model.fc = torch.nn.Identity()

    # Extract features
    with torch.no_grad():
        gen_features = inception_model(gen_images)
        gen_features = gen_features.cpu().numpy()
        real_features = inception_model(real_images)
        real_features = real_features.cpu().numpy()

    # Calculate mean and covariance of features
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(
        real_features, rowvar=False
    )
    mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(
        gen_features, rowvar=False
    )

    # Calculate the FID score
    ssdiff = np.sum((mu_real - mu_gen) ** 2.0)
    covmean = sqrtm(sigma_real.dot(sigma_gen))

    # Check for imaginary numbers in covmean and eliminate them
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)

    return fid


# def calculate_inception_custom(generated_images):
#     # Initialize the Inception model
#     inception_model = inception_v3(weights=True)
#     inception_model.eval()

#     # Pass the preprocessed images through the Inception model
#     with torch.no_grad():
#         logits = inception_model(generated_images)

#     print(logits.shape)
#     # Calculate softmax probabilities
#     probabilities = F.softmax(logits, dim=1)
#     # Calculate the marginal distribution by averaging probabilities across all images
#     marginal = torch.mean(probabilities, dim=0)
#     # Calculate KL divergence for each image and then average
#     kl_div = probabilities * (
#         torch.log(probabilities) - torch.log(marginal.unsqueeze(0))
#     )
#     kl_div = kl_div.sum(dim=1)
#     mean_kl_div = torch.mean(kl_div)

#     # Compute the Inception Score
#     inception_score = torch.exp(mean_kl_div).item()

#     return inception_score


# def calculate_nll(losses_path, device):
#     # Load the losses
#     losses = torch.load(losses_path, map_location=torch.device(device))
#     losses = torch.Tensor(losses)
#     nll = torch.mean(
#         torch.log(torch.sqrt(2 * torch.tensor(np.pi))) + 0.5 * torch.log(losses)
#     )

#     print("Negative Log Likelihood:", nll.item())
#     # Calculate the negative log likelihood bits per dim

#     bits_per_dim = nll.item() / (torch.log(torch.tensor(2)).item())

#     print("Negative Log Likelihood (bits/dim):", bits_per_dim)
