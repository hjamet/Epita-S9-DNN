import logging
import torch
from matplotlib import pyplot as plt
import numpy as np

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
        self.attributions = None

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
        attributions = self.__random_baseline_integrated_gradients(
            images_tensor,
            self.model.model,
            self.model.labels,
            self.__calculate_outputs_and_gradients,
            steps=50,
            num_random_trials=number_of_trials,
            cuda=True,
        )

        self.attributions = {
            name: attributions[i]
            for i, name in enumerate(list(self.model.images_tensors.keys()))
        }

        return self

    def plot_integrated_gradients(
        self, display: list = [], cmap=plt.cm.inferno, overlay_alpha=0.4
    ):
        if display == []:
            display = list(self.attributions.keys())
        for name, integrated_gradients in self.attributions.items():
            if name in display:
                logger.info(f"Plotting integrated gradients for image {name}...")
                # Sum of the attributions across color channels for visualization.
                # The attribution mask shape is a grayscale image with height and width
                # equal to the original image.
                attribution_mask = np.sum(np.abs(integrated_gradients), axis=2)

                image = self.model.images_tensors[name].numpy().transpose((1, 2, 0))

                fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

                axs[0, 0].set_title("Baseline image")
                axs[0, 0].imshow(self.baseline.numpy().transpose((2, 1, 0)))
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

    def __calculate_outputs_and_gradients(
        self, inputs, model, target_label_idx, cuda=False
    ):
        gradients = []
        for input in inputs:
            input = self.__pre_processing(input, cuda)
            output = model(input)
            output = torch.nn.functional.softmax(output, dim=1)
            index = np.zeros(output.shape, dtype=np.int64)
            index = torch.tensor(index, dtype=torch.int64)
            if cuda:
                index = index.cuda()
            output = output.gather(1, index)
            # clear grad
            model.zero_grad()
            output.sum().backward()
            gradient = input.grad.detach().cpu().numpy()[0]
            gradients.append(gradient)
        gradients = np.array(gradients)
        return gradients, target_label_idx

    def __random_baseline_integrated_gradients(
        self,
        inputs,
        model,
        target_label_idx,
        predict_and_gradients,
        steps,
        num_random_trials,
        cuda,
    ):
        all_intgrads = []
        for i in range(num_random_trials):
            integrated_grad = self.__integrated_gradients(
                inputs,
                model,
                target_label_idx,
                predict_and_gradients,
                steps=steps,
                cuda=cuda,
            )
            all_intgrads.append(integrated_grad)
            print("the trial number is: {}".format(i))
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads

    def __integrated_gradients(
        self,
        inputs,
        model,
        target_label_idx,
        predict_and_gradients,
        steps=50,
        cuda=False,
    ):
        scaled_inputs = [
            self.baseline + (float(i) / steps) * (inputs - self.baseline)
            for i in range(0, steps + 1)
        ]
        grads, _ = predict_and_gradients(scaled_inputs, model, target_label_idx, cuda)
        avg_grads = np.average(grads[:-1], axis=0)
        avg_grads = np.transpose(avg_grads, (1, 2, 0))
        delta_X = (
            (
                self.__pre_processing(inputs, cuda)
                - self.__pre_processing(self.baseline, cuda)
            )
            .detach()
            .squeeze(0)
            .cpu()
            .numpy()
        )
        delta_X = np.transpose(delta_X, (0, 2, 3, 1))
        integrated_grad = delta_X * avg_grads
        return integrated_grad

    def __pre_processing(self, obs, cuda):
        if cuda:
            torch_device = torch.device("cuda:0")
        else:
            torch_device = torch.device("cpu")
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=torch_device, requires_grad=True
        )
        return obs_tensor
