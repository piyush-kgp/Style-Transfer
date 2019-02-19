import numpy as np
import cv2
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model

VGG = VGG19(weights='imagenet', include_top=False)

img_path = 'cat.jpeg'
cat = cv2.imread(img_path)
img = image.load_img(img_path, target_size=cat.shape[:2])
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# cat = cv2.imread(img_path)
# x_dash = np.array(cat, dtype = np.float32)
# x_dash = np.expand_dims(x_dash, axis = 0)
# mean = [103.939, 116.779, 123.68]
# x_dash[..., 0] -= mean[0]
# x_dash[..., 1] -= mean[1]
# x_dash[..., 2] -= mean[2]
# np.array_equal(x, x_dash)

class GatysStyleTransfer(object):
    def __init__(self, content_file, style_file, model):
        self.content_image = self.read_image(content_file)
        self.style_image = self.read_image(style_file)
        self.model = model
        self.content_cost_layer = 'block4_conv2'
        self.style_cost_layers = {'block1_conv1': .2,
                                  'block2_conv1': .2,
                                  'block3_conv1': .2,
                                  'block4_conv1': .2,
                                  'block5_conv1': .2
                                 }
        pass

    def read_image(self, file_name, shape=(224, 224)):
        img = image.load_img(img_path, target_size=shape)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    @staticmethod
    def save_image(four_dim_tensor, file_name):
        import cv2
        mean = [103.939, 116.779, 123.68]
        cv2.imwrite(file_name, four_dim_tensor[0]+mean)

    def get_layer_output(self, layer_name, input):
        layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        layer_output = layer_model.predict(input)
        return layer_output

    def content_cost(self):
        pass
