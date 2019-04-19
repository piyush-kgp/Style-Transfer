"""
Citations:
https://shafeentejani.github.io/2017-01-03/fast-style-transfer/
https://arxiv.org/pdf/1603.08155.pdf
"""


from tensorflow.keras.applications import VGG19 as VGG
from tensorflow.keras.layers import *
from tensorflow.keras.models  import Model, Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
import glob
from PIL import Image
import numpy as np


class Config:
    content_blob = '../datasets/vangogh2photo/*B/*.jpg'
    style_blob = '../datasets/vangogh2photo/*A/*.jpg'
    img_size = [256, 256]
    batch_size = 50

config = Config()

def generate_dataset(blob):
    images = np.array(glob.glob(blob)).astype(np.str)
    idx = np.random.randint(0, len(images), config.batch_size).astype(np.int)
    sel_images = images[idx]
    images_np =  np.array([np.array(Image.open(file).resize(config.img_size, Image.BILINEAR))
            for file in sel_images
            ])
    return images_np.astype(np.float32)/127.5 - 1

def get_content_style_batch():
    return generate_dataset(config.content_blob), generate_dataset(config.style_blob)



###NEURAL NETWORK###
class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = Conv2D(filters1, (1, 1))
    self.bn2a = BatchNormalization()

    self.conv2b = Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = BatchNormalization()

    self.conv2c = Conv2D(filters3, (1, 1))
    self.bn2c = BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.trainable_variables])

######


def image_transformation_network():
    model = Sequential([
    Conv2D(input_shape=(256, 256, 3)),
    BatchNormalization(),
    Conv2DTranspose(),
    ])
    return model

def feature_reconstruction_loss(C, G):
    pass

def style_reconstruction_loss(S, G):
    pass

def total_loss(C, S, G):
    pass

def train():
    pass

train()

# class JohnsonStyleTransfer:
#     def __init__(self, style_img, content_img_dir, loss_network):
#         self.style_img = style_img
#         self.content_img_dir = content_img_dir
#         self.loss_network = loss_network
