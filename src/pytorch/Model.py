import torch
import torchvision
import logging
import tensorflow as tf
import numpy as np
import PIL
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Model:
    """A class to load and predict images using a pretrained model. This class is based on a Pytorch Model."""

    def __init__(self):
        """Calling this constructor will download the pretrained model and the imagenet labels."""

        # Uninitialized variables
        self.model = None
        self.labels = None
        self.images_tensors = None
        self.predictions = None

        self.__download_model()
        self.__download_labels(
            url="https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )

        # ---------------------------------------------------------------------------- #
        #                                PUBLIC METHODS                                #
        # ---------------------------------------------------------------------------- #

    def load_images(self, img_url: dict, display: bool = False):
        """Loads the images from the urls and stores them in a dictionary. If display is True, it will display the images.
        The dictionary keys are the image names and the values are the image tensors.

        Args:
            img_url (dict): A dictionary with the image names as keys and the image urls as values.
            display (bool, optional): Display the images. Defaults to False.

        Returns:
            Model: The model object.
        """
        logger.info("Loading images...")
        img_paths = {
            name: tf.keras.utils.get_file(name, url) for (name, url) in img_url.items()
        }
        self.images_tensors = {
            name: self.__read_image(path) for (name, path) in img_paths.items()
        }

        if display:
            logger.info("Displaying images...")
            plt.figure(figsize=(8, 8))
            for n, (name, img_tensors) in enumerate(self.images_tensors.items()):
                # permute the dimensions of the image from (224, 224, 3) to (3, 224, 224)
                img_tensors = img_tensors.numpy().transpose((1, 2, 0))
                plt.imshow(img_tensors)
                plt.title(name)
                plt.axis("off")
                plt.show()

        return self

    def predict(self, display: bool = False):
        """Predicts the class of the images. If display is True, it will display the predictions.

        Args:
            display (bool, optional): Display the predictions. Defaults to False.

        Returns:
            Model: The model object.
        """
        logger.info("Predicting images classes...")
        self.predictions = {
            name: self.__predict_image(img_tensor)
            for (name, img_tensor) in self.images_tensors.items()
        }

        if display:
            logger.info("Displaying predictions...")
            for n, (name, (labels, probs)) in enumerate(self.predictions.items()):
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(self.images_tensors[name].numpy().transpose((1, 2, 0)))
                ax[0].set_title(name)
                ax[0].axis("off")
                ax[1].barh(np.arange(5), probs)
                ax[1].set_aspect(0.1)
                ax[1].set_yticks(np.arange(5))
                ax[1].set_yticklabels(labels, size="small")
                ax[1].yaxis.set_ticks_position("right")
                ax[1].set_title("Probability")
                ax[1].set_xlim(0, 1.1)
                plt.show()
        return self

        # ---------------------------------------------------------------------------- #
        #                                PRIVATE METHODS                               #
        # ---------------------------------------------------------------------------- #

    def __download_model(self):
        """Downloads the pretrained model.

        Returns:
            Model: The model object.
        """
        logger.info(f"Downloading pretrained model: InceptionV1")
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "googlenet", pretrained=True
        )
        self.model.eval()

        return self

    def __download_labels(self, url: str):
        """Downloads the imagenet labels.

        Args:
            url (str): The url of the labels to download.

        Returns:
            Model: The model object.
        """
        logger.info("Downloading imagenet labels...")
        labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", url)
        self.labels = np.array(open(labels_path).read().splitlines())
        return self

    def __read_image(self, file_name: str):
        """Reads an image from a file and converts it to a tensor. The image is resized to 224x224.
        They are also normalized to the range [0, 1].

        Args:
            file_name (str): The path of the image to read.

        Returns:
            tf.Tensor: The image tensor.
        """
        input_image = PIL.Image.open(file_name)
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(input_image)
        return input_tensor

    def __predict_image(self, image: torch.Tensor, k: int = 5):
        """Predicts the class of an image. It returns the top k predictions.

        Args:
            image (torch.Tensor): The image tensor.
            k (int, optional): The number of predictions to return. Defaults to 5.

        Returns:
            tuple: A tuple containing the labels and the probabilities of the predictions.
        """
        input_batch = image.unsqueeze(0)
        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")
            self.model.to("cuda")

        with torch.no_grad():
            output = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        topk_prob, topk_class = torch.topk(probabilities, k=k)
        topk_prob = topk_prob.cpu().numpy()
        topk_class = topk_class.cpu().numpy()
        topk_labels = self.labels[topk_class + 1]
        return topk_labels, topk_prob
