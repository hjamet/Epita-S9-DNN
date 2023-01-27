import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Model:
    def __init__(self):

        # Uninitialized variables
        self.model = None
        self.imagenet_labels = None
        self.images_tensors = None
        self.predictions = None

        self.__download_model(
            name="inception_v1",
            handle="https://tfhub.dev/google/imagenet/inception_v1/classification/4",
        )
        self.__download_labels(
            url="https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
        )

    # ---------------------------------------------------------------------------- #
    #                                PUBLIC METHODS                                #
    # ---------------------------------------------------------------------------- #

    def summary(self):
        logger.info("Model summary:")
        self.model.summary()
        return self

    def load_images(self, img_url: dict, display: bool = False):
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
                plt.imshow(img_tensors)
                plt.title(name)
                plt.axis("off")
                plt.show()

        return self

    def predict(self, display: bool = False):
        logger.info("Predicting images classes...")
        self.predictions = {
            name: self.__predict_image(img_tensor)
            for (name, img_tensor) in self.images_tensors.items()
        }

        if display:
            logger.info("Displaying predictions...")
            for n, (name, (labels, probs)) in enumerate(self.predictions.items()):
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(self.images_tensors[name])
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

    def __download_model(self, name: str, handle: str):
        logger.info(f"Downloading pretrained model: {name}")
        self.model = tf.keras.Sequential(
            [
                hub.KerasLayer(
                    name=name,
                    handle=handle,
                    trainable=False,
                ),
            ]
        )
        self.model.build([None, 224, 224, 3])

    def __download_labels(self, url: str):
        logger.info("Downloading imagenet labels...")
        labels_path = tf.keras.utils.get_file("ImageNetLabels.txt", url)
        self.imagenet_labels = np.array(open(labels_path).read().splitlines())
        return self

    def __read_image(self, file_name: str):
        image = tf.io.read_file(file_name)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
        return image

    def __predict_image(self, image: tf.Tensor, k: int = 5):
        image_batch = tf.expand_dims(image, 0)
        predictions = self.model(image_batch)
        probs = tf.nn.softmax(predictions, axis=-1)
        top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
        top_labels = self.imagenet_labels[tuple(top_idxs)]
        return top_labels, top_probs[0]
