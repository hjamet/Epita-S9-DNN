import logging
import torch
from matplotlib import pyplot as plt
import numpy as np
from tqdm.notebook import tqdm_notebook

from Model import Model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IntegratedGradients:
    """A class to compute the integrated gradients of a Tensorflow model."""

    def __init__(self, model: Model, baseline_color="black"):
        """Initializes the IntegratedGradients class.

        Args:
            model (Model): The model object. It must be an instance of the Model class.
        """
        self.model = model
        self.intregrated_gradients = None

        # Uninitialized variables
        self.baseline = None

        self.__generate_baseline(baseline_color)

    # ---------------------------------------------------------------------------- #
    #                                PUBLIC METHODS                                #
    # ---------------------------------------------------------------------------- #

    def compute_integrated_gradients(self, steps: int = 50, number_of_trials: int = 1):
        """Computes the integrated gradients of the model.

        Args:
            steps (int, optional): The number of steps between the baseline and the input images. Defaults to 50.
            number_of_trials (int, optional): The number of trials to compute the integrated gradients. Defaults to 1.
            The final result is the average of the integrated gradients of all trials.

        Returns:
            IntegratedGradients: The IntegratedGradients object.
        """
        logger.info("Computing integrated gradients...")
        images_tensor = torch.stack(list(self.model.images_tensors.values()))
        attributions = self.__compute_trials(
            images_tensor,
            self.model.model,
            steps=steps,
            num_random_trials=number_of_trials,
            cuda=True,
        )

        self.intregrated_gradients = {
            name: attributions[i]
            for i, name in enumerate(list(self.model.images_tensors.keys()))
        }

        return self

    def plot_integrated_gradients(
        self, display: list = [], cmap=plt.cm.inferno, overlay_alpha: float = 0.4
    ):
        """Plots the integrated gradients of the model.

        Args:
            display (list, optional): The list of the attributions to display. Default to [].
            cmap (matplotlib.colors.Colormap, optional): The colormap to use. Defaults to plt.cm.inferno.
            overlay_alpha (float, optional): The transparency of the overlay. Defaults to 0.4.

        Returns:
            IntegratedGradients: The IntegratedGradients object.
        """
        if display == []:
            display = list(self.intregrated_gradients.keys())
        for name, integrated_gradients in self.intregrated_gradients.items():
            if name in display:
                logger.info(f"Plotting integrated gradients for image {name}...")

                attribution_mask = np.sum(np.abs(integrated_gradients), axis=2)

                image = self.model.images_tensors[name].numpy().transpose((1, 2, 0))

                fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

                axs[0, 0].set_title("Baseline image")
                axs[0, 0].imshow(self.baseline.numpy().transpose((1, 2, 0)))
                axs[0, 0].axis("off")

                axs[0, 1].set_title("Original image")
                axs[0, 1].imshow(image)
                axs[0, 1].axis("off")

                axs[1, 0].set_title("Attribution mask")
                axs[1, 0].imshow(attribution_mask, cmap=cmap)
                axs[1, 0].axis("off")

                axs[1, 1].set_title("Overlay")
                axs[1, 1].imshow(attribution_mask, cmap=cmap)
                axs[1, 1].imshow(image, alpha=overlay_alpha)
                axs[1, 1].axis("off")

                plt.tight_layout()
        return self

    # ---------------------------------------------------------------------------- #
    #                                PRIVATE METHODS                               #
    # ---------------------------------------------------------------------------- #

    def __generate_baseline(self, color="black"):
        """Generates a baseline image of the same size as the input image.

        Args:
            color (str, optional): The color of the baseline. Defaults to "black".
            can be either "black", "white" "grey" or "random".

        Raises:
            ValueError: Unknown color.
            ValueError: No images loaded.

        Returns:
            IntegratedGradients: The IntegratedGradients object.
        """
        logger.info("Generating baseline image...")
        if self.model.images_tensors is None or len(self.model.images_tensors) == 0:
            raise ValueError("No images loaded.")
        img = self.model.images_tensors[list(self.model.images_tensors.keys())[0]]
        if color == "black":
            self.baseline = torch.zeros_like(img)
        elif color == "white":
            self.baseline = torch.ones_like(img)
        elif color == "grey":
            self.baseline = torch.ones_like(img) * 0.5
        elif color == "random":
            self.baseline = torch.random.uniform(shape=img.shape, minval=0, maxval=1)
        else:
            raise ValueError("Unknown color.")
        return self

    def __compute_gradients(self, inputs: np.array, model: Model, cuda: bool = False):
        """Computes the gradients of the model.

        Args:
            inputs (np.array): The input images. The shape is (N, C, H, W).
            model (Model): The model.
            cuda (bool, optional): Whether to use cuda. Defaults to False.

        Returns:
            np.array: The gradients of the model.
        """
        gradients = []
        for input in inputs:
            input = self.__to_tensor(input, cuda)
            logits = model(input)
            output = torch.nn.functional.softmax(logits, dim=1)
            index = np.zeros(output.shape, dtype=np.int64)
            index = torch.tensor(index, dtype=torch.int64)
            if cuda:
                index = index.cuda()
            output = output.gather(1, index)

            output.sum().backward()
            gradients.append(input.grad.detach().cpu().numpy()[0])
            model.zero_grad()
        return np.array(gradients)

    def __compute_trials(
        self,
        inputs: np.array,
        model: Model,
        steps: int,
        num_random_trials: int,
        cuda: bool = False,
    ):
        """Computes the integrated gradients of the model.

        Args:
            inputs (np.array): The input images. The shape is (N, C, H, W).
            model (Model): The model.
            steps (int): The number of steps.
            num_random_trials (int): The number of random trials.
            cuda (bool, optional): Whether to use cuda. Defaults to False.

        Returns:
            np.array: The integrated gradients of the model.
        """
        trails_gradients_list = []
        for _ in tqdm_notebook(range(num_random_trials)):
            integrated_grad = self.__compute_integrated_gradients(
                inputs,
                model,
                steps=steps,
                cuda=cuda,
            )
            trails_gradients_list.append(integrated_grad)
        ig = np.average(np.array(trails_gradients_list), axis=0)
        return ig

    def __compute_integrated_gradients(
        self,
        inputs: np.array,
        model: Model,
        steps: int = 50,
        cuda: bool = False,
    ):
        """Computes the integrated gradients of the model.

        Args:
            inputs (np.array): The input images. The shape is (N, C, H, W).
            model (Model): The model.
            steps (int, optional): The number of steps. Defaults to 50.
            cuda (bool, optional): Whether to use cuda. Defaults to False.

        Returns:
            np.array: The integrated gradients of the model.
        """
        scaled_inputs = [
            self.baseline + (float(i) / steps) * (inputs - self.baseline)
            for i in range(0, steps + 1)
        ]
        gradients = self.__compute_gradients(scaled_inputs, model, cuda)
        mean_gradients = np.average(gradients[:-1], axis=0)
        mean_gradients = np.transpose(mean_gradients, (1, 2, 0))
        integrated_gradients = (
            np.transpose(
                (
                    (
                        self.__to_tensor(inputs, cuda)
                        - self.__to_tensor(self.baseline, cuda)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                ),
                (0, 2, 3, 1),
            )
            * mean_gradients
        )
        return integrated_gradients

    def __to_tensor(self, array: np.array, cuda: bool = False):
        """Converts a numpy array to a torch tensor.

        Args:
            array (np.array): The numpy array.
            cuda (bool, optional): Whether to use cuda. Defaults to False.

        Returns:
            torch.tensor: The torch tensor.
        """
        if cuda:
            torch_device = torch.device("cuda:0")
        else:
            torch_device = torch.device("cpu")
        obs_tensor = torch.tensor(
            array, dtype=torch.float32, device=torch_device, requires_grad=True
        )
        return obs_tensor
