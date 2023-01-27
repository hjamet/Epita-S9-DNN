import logging
import tensorflow as tf
from matplotlib import pyplot as plt

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

        # Uninitialized variables
        self.baseline = None
        self.intepolated_images = None

        self.__generate_baseline(color=baseline_color)

    # ---------------------------------------------------------------------------- #
    #                                PUBLIC METHODS                                #
    # ---------------------------------------------------------------------------- #

    def display_baseline(self):
        """Displays the baseline image.

        Returns:
            IntegratedGradients: The IntegratedGradients object.
        """
        logger.info("Displaying baseline image...")
        plt.imshow(self.baseline)
        plt.axis("off")
        plt.title("Baseline image")
        plt.show()
        return self

    def interpolate_images(self, steps: int = 50, display: dict = {}):
        """Interpolates the images between the baseline and the input images.

        Args:
            steps (int, optional): The number of steps between the baseline and the input images. Defaults to 50.
            display (dict, optional): A dictionary with the image names as keys and the number of images to display as values. Defaults to {} (no images displayed)

        Returns:
            IntegratedGradients: The IntegratedGradients object.
        """
        logger.info("Interpolating images...")
        self.interpolated_images = {}
        for name, img in self.model.images_tensors.items():
            self.interpolated_images[name] = [
                self.baseline + (img - self.baseline) * i / steps
                for i in range(steps + 1)
            ]
        if display != {}:
            display = {
                k: v
                for k, v in display.items()
                if (
                    lambda x: True
                    if x in self.interpolated_images
                    else logger.warning(f"Image {x} not found.")
                )(k)
            }
            for name, nbr in display.items():
                fig, ax = plt.subplots(1, nbr, figsize=(20, 4))
                fig.suptitle(name)
                print(steps // (nbr - 1))
                for column, i in enumerate(range(0, steps + 1, steps // (nbr - 1))):
                    if column > len(ax) - 1:
                        break
                    ax[column].imshow(self.interpolated_images[name][i])
                    ax[column].axis("off")
                    ax[column].set_title(f"Step {i}")
                plt.show()

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
            self.baseline = tf.zeros_like(img)
        elif color == "white":
            self.baseline = tf.ones_like(img)
        elif color == "grey":
            self.baseline = tf.ones_like(img) * 0.5
        elif color == "random":
            self.baseline = tf.random.uniform(shape=img.shape, minval=0, maxval=1)
        else:
            raise ValueError("Unknown color.")
        return self
