from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import cv2


model = VGG19(weights='imagenet', include_top=False)

img_path = 'cat.jpeg'
cat = cv2.imread(img_path)
x_dash = np.array(cat, dtype = np.float32)
x_dash = np.expand_dims(x_dash, axis = 0)
mean = [103.939, 116.779, 123.68]
x_dash[..., 0] -= mean[0]
x_dash[..., 1] -= mean[1]
x_dash[..., 2] -= mean[2]

img = image.load_img(img_path, target_size=cat.shape[:2])
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
np.array_equal(x, x_dash)

features = model.predict(x)


layers = [l.name for l in model.layers]
