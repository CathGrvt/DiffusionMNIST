import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


def ddpm_schedules(beta1: float, beta2: float, T: int) -> Dict[str, torch.Tensor]:
    """Returns pre-computed schedules for DDPM sampling with a linear noise schedule."""
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = torch.exp(
        torch.cumsum(torch.log(1 - beta_t), dim=0)
    )  # Cumprod in log-space (better precision)

    return {"beta_t": beta_t, "alpha_t": alpha_t}


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        expected_shape,
        act=nn.GELU,
        kernel_size=7,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm((out_channels, *expected_shape)),
            act(),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        expected_shape=(28, 28),
        n_hidden=(64, 128, 64),
        kernel_size=7,
        last_kernel_size=3,
        time_embeddings=16,
        act=nn.GELU,
    ) -> None:
        super().__init__()
        last = in_channels

        self.blocks = nn.ModuleList()
        for hidden in n_hidden:
            self.blocks.append(
                CNNBlock(
                    last,
                    hidden,
                    expected_shape=expected_shape,
                    kernel_size=kernel_size,
                    act=act,
                )
            )
            last = hidden

        # The final layer, we use a regular Conv2d to get the
        # correct scale and shape (and avoid applying the activation)
        self.blocks.append(
            nn.Conv2d(
                last,
                in_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            )
        )

        # This part is literally just to put the single scalar "t" into the CNN
        # in a nice, high-dimensional way:
        self.time_embed = nn.Sequential(
            nn.Linear(time_embeddings * 2, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, 128),
            act(),
            nn.Linear(128, n_hidden[0]),
        )
        frequencies = torch.tensor(
            [0] + [2 * np.pi * 1.5**i for i in range(time_embeddings - 1)]
        )
        self.register_buffer("frequencies", frequencies)

    def time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        phases = torch.concat(
            (
                torch.sin(t[:, None] * self.frequencies[None, :]),
                torch.cos(t[:, None] * self.frequencies[None, :]) - 1,
            ),
            dim=1,
        )

        return self.time_embed(phases)[:, :, None, None]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Shapes of input:
        #    x: (batch, chan, height, width)
        #    t: (batch,)

        embed = self.blocks[0](x)
        # ^ (batch, n_hidden[0], height, width)

        # Add information about time along the diffusion process
        #  (Providing this information by superimposing in latent space)
        embed += self.time_encoding(t)
        #         ^ (batch, n_hidden[0], 1, 1) - thus, broadcasting
        #           to the entire spatial domain

        for block in self.blocks[1:]:
            embed = block(embed)

        return embed


class DDPM(nn.Module):
    def __init__(
        self,
        gt,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt

        noise_schedule = ddpm_schedules(betas[0], betas[1], n_T)

        # `register_buffer` will track these tensors for device placement, but
        # not store them as model parameters. This is useful for constants.
        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.beta_t  # Exists! Set by register_buffer
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])
        self.alpha_t

        self.n_T = n_T
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Algorithm 18.1 in Prince"""

        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting

        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        # This is the z_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this z_t. Loss is what we return.

        return self.criterion(eps, self.gt(z_t, t / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        """Algorithm 18.2 in Prince"""

        _one = torch.ones(n_sample, device=device)
        z_t = torch.randn(n_sample, *size, device=device)
        for i in range(self.n_T, 0, -1):
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(
                z_t, (i / self.n_T) * _one
            )
            z_t /= torch.sqrt(1 - beta_t)

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of the algorithm)

        return z_t


# The `MIX_DDPM` class implements a diffusion probabilistic model with custom degradation techniques
# for image processing tasks.
class MIX_DDPM(nn.Module):
    def __init__(
        self,
        gt,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt

        noise_schedule = ddpm_schedules(betas[0], betas[1], n_T)

        self.register_buffer("beta_t", noise_schedule["beta_t"])
        self.register_buffer("alpha_t", noise_schedule["alpha_t"])

        self.n_T = n_T
        self.criterion = criterion

    def adjust_contrast(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """
        Adjust the contrast of the images.
        The range of contrast adjustment increases with the diffusion step intensity.
        """
        intensity = intensity.mean().cpu()
        # Define the minimum and maximum contrast levels based on intensity
        min_contrast = (
            0.5 - intensity * 0.5
        )  # Minimum contrast decreases with intensity
        max_contrast = (
            1.5 + intensity * 0.5
        )  # Maximum contrast increases with intensity

        # Randomly choose a contrast level within the range for each image
        contrast_levels = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(
            min_contrast, max_contrast
        )

        # Adjust contrast
        means = x.mean(dim=[2, 3], keepdim=True)
        x = (x - means) * contrast_levels + means

        return x

    def apply_cutout_masks(self, x: torch.Tensor, intensity: float) -> torch.Tensor:
        """
        Apply random cutout masks to simulate occlusions.
        The number and size of masks increase with the diffusion step intensity.
        """
        B, C, H, W = x.shape
        intensity = intensity.mean()

        num_masks = int(
            1 + intensity * 5
        )  # Starts with 1 mask, increases with intensity
        mask_size = int(
            3 + intensity * H / 4
        )  # Mask size starts from 3x3 and increases

        for _ in range(num_masks):
            # Randomly choose the center of the mask
            mask_center_x = torch.randint(0, W, (1,)).item()
            mask_center_y = torch.randint(0, H, (1,)).item()

            # Calculate the indices of the mask
            left = max(mask_center_x - mask_size // 2, 0)
            right = min(mask_center_x + mask_size // 2 + 1, W)
            top = max(mask_center_y - mask_size // 2, 0)
            bottom = min(mask_center_y + mask_size // 2 + 1, H)

            # Apply the mask
            x[
                :, :, top:bottom, left:right
            ] = 0  # Assuming the background is black. Adjust if necessary.

        return x

    def apply_custom_degradation(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """
        Apply custom degradation based on Structural Perturbation and Contrast Variation.
        The degradation intensity scales with the diffusion step.
        """
        intensity = step / self.n_T
        x = self.apply_cutout_masks(x, intensity)
        x = self.adjust_contrast(x, intensity)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function performs a forward pass through a neural network with custom degradation and the
        option to add original diffusion noise.

        @param x It seems like the code snippet you provided is a part of a neural network model,
        specifically the forward pass of the model. The `forward` method takes a tensor `x` as input and
        performs operations on it to generate an output tensor.

        @return The `forward` method is returning the result of applying the criterion function to `eps`
        and the ground truth obtained by passing `z_t` and `t / self.n_T` to the `gt` method.
        """
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)
        alpha_t = self.alpha_t[t, None, None, None]  # Get right shape for broadcasting

        # Apply custom degradation
        z_t = self.apply_custom_degradation(
            x, alpha_t
        )  # Pass `t` instead of `alpha_t` ?

        # Option to add original diffusion noise
        include_original_diffusion_noise = (
            True  # Set to False if custom degradation replaces the need for this
        )
        if include_original_diffusion_noise:
            z_t = torch.sqrt(alpha_t) * z_t + torch.sqrt(1 - alpha_t) * eps

        return self.criterion(eps, self.gt(z_t, t / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:
        """
        The function `sample` implements Algorithm 18.2 from Prince, generating a tensor `z_t` based on
        given parameters and calculations.

        @param n_sample The `n_sample` parameter in the `sample` function represents the number of
        samples you want to generate. It is used to create a tensor of size `(n_sample, *size)` filled
        with random values from a normal distribution.
        @param size The `size` parameter in the `sample` function represents the size of the output
        tensor that will be generated. It is used to specify the shape of the tensor that will be
        created by the function. The `size` parameter is expected to be a tuple that defines the
        dimensions of the output tensor
        @param device The `device` parameter in the `sample` function is used to specify the device on
        which the computations will be performed. This could be a CPU or a specific GPU device. It is
        important to ensure that the tensors and operations are placed on the correct device for
        efficient computation.

        @return The function `sample` returns a torch.Tensor `z_t` after performing Algorithm 18.2 from
        the Prince library. The algorithm involves operations on the input tensor `z_t` based on the
        values of `alpha_t` and `beta_t` at each step in the loop. The final result is the modified
        tensor `z_t` after the loop iterations.
        """
        """Algorithm 18.2 in Prince"""

        _one = torch.ones(n_sample, device=device)
        z_t = torch.randn(n_sample, *size, device=device)
        for i in range(self.n_T, 0, -1):
            alpha_t = self.alpha_t[i]
            beta_t = self.beta_t[i]

            # First line of loop:
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * self.gt(
                z_t, (i / self.n_T) * _one
            )
            z_t /= torch.sqrt(1 - beta_t)

            if i > 1:
                # Last line of loop:
                z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            # (We don't add noise at the final step - i.e., the last line of the algorithm)

        return z_t


class QR_DDPM(nn.Module):
    """
    The `QR_DDPM` class in Python implements a model that applies QR code patterns to images and
    predicts the original images from the QR-coded ones using a specified schedule for noise intensity.

    @param n_T The `n_T` parameter in the `QR_DDPM` class represents the number of time steps or
    intervals used in the dynamic QR code pattern modulation. It is used to determine the schedule for
    changing the intensity of the QR code pattern applied to the images over time.

    @return defines a PyTorch module `QR_DDPM` that implements a model for applying QR
    code transformations to images and training a CNN to predict the original images from the QR-coded
    images. The module includes methods for applying QR transformations, forward pass computation,
    creating random QR code patterns, and sampling images with QR codes applied.
    """

    def __init__(
        self,
        gt,
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super().__init__()

        self.gt = gt
        self.n_T = n_T

        noise_schedule = create_sigma_schedule(n_T)
        self.register_buffer("sigma_t", noise_schedule)
        self.sigma_t

        self.criterion = criterion

    def apply_qr_transformation(self, x, intensity):
        """
        The function `apply_qr_transformation` applies a QR code pattern onto a batch of images with a
        specified intensity level.

        @param x a batch of images with the shape (B, C, H, W), where:
        @param intensity the strength or opacity of the QR code pattern that will be applied onto the input image. It is
        a scalar value that determines how prominent the QR code pattern will be in the final
        transformed image.

        @return the transformed image `transformed_x` after applying a QR code pattern with a given intensity.
        """
        # x is a batch of images of shape (B, C, H, W)
        batch_size, channels, height, width = x.size()
        qr_codes = self.create_batch_random_qr(batch_size, (channels, height, width))
        # Make sure intensity is broadcastable over the batch
        intensity = intensity.view(-1, 1, 1, 1).to(x.device)
        qr_codes = qr_codes.to(x.device)
        # Apply the QR code pattern with the given intensity
        transformed_x = x * (1 - intensity) + qr_codes * (intensity)

        return transformed_x

    def forward(self, x):
        """
        The `forward` function takes an input image `x`, applies a QR code transformation at a random
        time step `t`, predicts the original image from the transformed image, and calculates the loss
        between the original image and the predictions.

        @param x the input batch of images that will be processed by the neural network.

        @return the loss calculated between the original images x and the predictions preds.
        """
        # Sample a random time step t for each batch element
        t = torch.randint(0, self.n_T, (x.shape[0],), device=x.device)

        # Get the sigma for the QR code transformation at the corresponding time step t
        sigma_t = self.sigma_t[t]

        # Transform each image in the batch x with the corresponding QR code pattern
        z_t = self.apply_qr_transformation(x, sigma_t)

        # The CNN tries to predict the original image x from the QR-coded image z_t
        preds = self.gt(z_t, t / self.n_T)

        # Return the loss between the original images x and the predictions preds
        loss = self.criterion(x, preds)

        return loss

    def create_batch_random_qr(self, batch_size, image_size):
        """
        The function `create_batch_random_qr` generates a batch of random pseudo-QR code images with
        specified size and number of channels.

        @param batch_size the number of pseudo-QR images that will be generated in
        each batch.
        @param image_size a tuple containing the dimensions of the image
        in terms of channels, height, and width.

        @return a PyTorch tensor containing a batch of pseudo-QR images.
        """
        channels, height, width = image_size
        # Initialize an empty list to store the pseudo-QR images
        pseudo_qr_batch_list = []

        for _ in range(batch_size):
            # Create a random binary pattern
            pseudo_qr_array = np.random.choice(
                [0, 255], size=(height, width), p=[0.5, 0.5]
            ).astype(np.uint8)

            # If more than one channel is needed, replicate the grayscale pattern across the channels
            if channels > 1:
                pseudo_qr_array = np.repeat(
                    pseudo_qr_array[..., np.newaxis], channels, axis=-1
                )

            pseudo_qr_batch_list.append(pseudo_qr_array)

        # Convert the list of arrays into a single NumPy array
        pseudo_qr_batch = np.stack(pseudo_qr_batch_list, axis=0)

        # Convert the NumPy array to a PyTorch tensor and normalize to [-0.5, 0.5]
        pseudo_qr_batch_tensor = torch.from_numpy(pseudo_qr_batch).float() / 255.0 - 0.5

        return pseudo_qr_batch_tensor.unsqueeze(
            1
        )  # Add a channel dimension, resulting in shape (B, 1, H, W)

    def sample(self, n_sample, size, device):
        """
        The function generates predictions from a model and gradually reveals an MNIST image by undoing
        a QR code transformation.

        @param n_sample number of samples to generate.
        @param size the size of the images being processed or generated.
        @param device  CPU or GPU.

        @return the final image `z_t`, which is the MNIST image with no QR
        code pattern applied after undergoing a series of transformations and predictions based on the
        input parameters and model predictions.
        """
        # Start with a fully 'QR-coded' random image
        z_t = self.create_batch_random_qr(n_sample, size).to(device)
        _one = torch.ones(n_sample, device=device)
        for t in reversed(range(0, self.n_T)):
            if t > 0:
                sigma_t = self.sigma_t[t]
                sigma_t_minus_1 = self.sigma_t[t - 1]

                # Here we generate the predictions from the model, which are presumably 'less QR-coded' than z_t
                x_0_pred = self.gt(z_t, (t / self.n_T) * _one)

                # Undo the QR code transformation by a schedule, gradually revealing the MNIST image
                z_t = (
                    z_t
                    - self.apply_qr_transformation(x_0_pred, sigma_t)
                    + self.apply_qr_transformation(x_0_pred, sigma_t_minus_1)
                )
            else:
                # The final step should be the MNIST image with no QR code pattern applied
                z_t = x_0_pred

        return z_t


def create_sigma_schedule(n_T):
    """
    The function `create_sigma_schedule` generates a sigmoid schedule that gradually increases from 0 to
    1 over a specified number of time steps.

    @param n_T the number of steps over which the sigmoid schedule will transition from 0 to 1.

    @return a schedule of sigmoid values ranging from 0 to 1 over `n_T` steps.
    """
    # Sigmoid schedule from 0 to 1 over n_T steps
    timesteps = torch.linspace(-4, 6, steps=n_T)  # Wide range for a gradual sigmoid
    sigma_schedule = torch.sigmoid(
        timesteps
    )  # Sigmoid function for non-linear scheduling
    return sigma_schedule
