import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Model:
    def __init__(self):

        # Uninitialized variables
        self.model = None
        self.images_tensors = None

        self.__download_model(
            name="inception_v1",
            handle="https://tfhub.dev/google/imagenet/inception_v1/classification/4",
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

    def __read_image(self, file_name: str):
        image = tf.io.read_file(file_name)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
        return image
