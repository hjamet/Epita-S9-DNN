import logging
import tensorflow as tf
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

        # Uninitialized variables
        self.baseline = None
        self.interpolated_images = None
        self.gradients = None
        self.integrated_gradients = None

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
                self.baseline * ((steps - 1) - i) + (img - self.baseline) * i / steps
                for i in range(steps)
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
                    if i > steps - 1:
                        i = steps - 1
                    ax[column].imshow(self.interpolated_images[name][i])
                    ax[column].axis("off")
                    ax[column].set_title(f"Step {i}")
                plt.show()

        return self

    def compute_gradients(self, display: list = []):
        """Computes the gradients of the model for each interpolated image.
        The gradients are stored in the gradients attribute.

        Args:
            images (tf.Tensor): The images.
            display (list, optional): A list of images to display. Defaults to [] (no images displayed).

        Returns:
            IntegratedGradients: The IntegratedGradients object.
        """
        if self.interpolated_images is None:
            logger.warning("No interpolated images found. Interpolating images...")
            self.interpolate_images()
        gradients = {}
        for name, images in self.interpolated_images.items():
            logger.info(f"Computing average gradients for image {name}...")
            with tf.GradientTape() as tape:
                images = tf.stack(images, axis=0)
                tape.watch(images)
                logits = self.model.model(images)
                probs = tf.nn.softmax(logits, axis=-1)[
                    :,
                    np.where(self.model.labels == self.model.predictions[name][0][0])[
                        0
                    ][0],
                ]
                gradients[name] = tape.gradient(probs, images)

            if name in display:
                plt.figure(figsize=(10, 4))

                ax1 = plt.subplot(1, 2, 1)
                ax1.plot(probs)
                ax1.set_title(
                    "Probabilité de prédiction de la classe la plus probable."
                )
                ax1.set_ylabel("Confiance de la prédiction")
                ax1.set_xlabel("Barycentre entre l'image de base et l'image d'origine")
                ax1.set_ylim([0, 1])

                ax2 = plt.subplot(1, 2, 2)
                average_grads = tf.reduce_mean(gradients[name], axis=[1, 2, 3])
                average_grads_norm = (
                    average_grads - tf.math.reduce_min(average_grads)
                ) / (tf.math.reduce_max(average_grads) - tf.reduce_min(average_grads))
                ax2.plot(average_grads_norm)
                ax2.set_title("Moyenne normalisée des gradients des pixels.")
                ax2.set_ylabel("Gradient moyen des pixels")
                ax2.set_xlabel("Barycentre entre l'image de base et l'image d'origine")
                ax2.set_ylim([0, 1])

                # Add title
                plt.suptitle(
                    f"Pour l'image {name}, en fonction du barycentre entre l'image de base et l'image d'origine",
                    fontsize=16,
                )
                # Add space under title
                plt.subplots_adjust(top=0.85)

                plt.show()

        self.gradients = gradients
        return self

    def integrate_gradients(self):
        """Integrates the gradients of the model for each interpolated image.
        The integrated gradients are stored in the integrated_gradients attribute.

        Args:
            images (tf.Tensor): The images.

        Returns:
            IntegratedGradients: The IntegratedGradients object.
        """
        if self.gradients is None:
            logger.warning("No gradients found. Computing gradients...")
            self.compute_gradients()
        integrated_gradients = {}
        for name, gradients in self.gradients.items():
            logger.info(f"Integrating gradients for image {name}...")
            grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
            integrated_gradients[name] = tf.math.reduce_mean(grads, axis=0)

        self.integrated_gradients = integrated_gradients
        return self

    def plot_integrated_gradients(
        self, display: list = [], cmap=plt.cm.inferno, overlay_alpha=0.4
    ):
        for name, integrated_gradients in self.integrated_gradients.items():
            if name in display:
                logger.info(f"Plotting integrated gradients for image {name}...")
                # Sum of the attributions across color channels for visualization.
                # The attribution mask shape is a grayscale image with height and width
                # equal to the original image.
                attribution_mask = tf.reduce_sum(
                    tf.math.abs(integrated_gradients), axis=-1
                )

                image = self.interpolated_images[name][-1]

                fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

                axs[0, 0].set_title("Baseline image")
                axs[0, 0].imshow(self.baseline)
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
