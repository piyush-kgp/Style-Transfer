
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
import tensorflow as tf


class Config:
    img_size = (400, 300)
    num_channels = 3
    noise_ratio = 0.6
    num_iters = 100
    content_cost_layer = 'block4_conv2'
    style_cost_layers = {'block1_conv1': 0.2,
                         'block2_conv1': 0.2,
                         'block3_conv1': 0.2,
                         'block4_conv1': 0.2,
                         'block5_conv1': 0.2
                        }
    pretrained_model = VGG19(weights='imagenet', include_top=False)
    #Total Cost = alpha*J_c + beta*J_s
    alpha = 10
    beta = 40
    logdir = 'logs/'


class GatysStyleTransfer(object):
    """
    Implemenation of Gatys' Style Transfer Algorithm in Keras and Tensorflow
    https://arxiv.org/abs/1508.06576
    """
    def __init__(self, content_file, style_file):
        self.config = Config()
        self.content_img = self.read_image(content_file, name='content_img')
        self.style_img = self.read_image(style_file, name='style_img')
        self.sess = tf.Session()


    def read_image(self, file_name, name):
        img = Image.open(file_name)
        img = img.resize(self.config.img_size)
        img = np.array(img).astype(np.float32)/127.5 - 1
        img = np.expand_dims(img, axis=0)
        return tf.convert_to_tensor(img, name=name, dtype=tf.float32)


    def generate_noise_image(self):
        noise_img = tf.random_normal((1,) + tuple(reversed(self.config.img_size)) + \
                    (self.config.num_channels,), name='noise_img')
        gen_img = self.config.noise_ratio*noise_img + (1-self.config.noise_ratio) * \
                  self.content_img
        gen_img = tf.identity(gen_img, 'gen_img')
        return gen_img


    def get_intermediate_model(self, layer_name):
        layer_model = Model(inputs=self.config.pretrained_model.input, \
                            outputs=self.config.pretrained_model.get_layer(\
                                    layer_name).output
                           )
        return layer_model


    @property
    def content_model(self):
        with tf.name_scope('content_model'):
            model = self.get_intermediate_model(layer_name=self.config.content_cost_layer)
        model.trainable = False
        return model


    @property
    def style_models(self):
        with tf.name_scope('style_models'):
            models = {layer: self.get_intermediate_model(layer_name=layer) for \
                      layer in self.config.style_cost_layers}
        for model in models.values():
            model.trainable = False
        return models


    def compute_content_cost(self):
        a_C = self.content_model(self.content_img)
        a_G = self.content_model(self.gen_img)
        return 1/4 * tf.reduce_mean(tf.pow(a_C - a_G, 2))


    @staticmethod
    def gram_matrix(tensor):
        _, h, w, c = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, (h*w, c))
        return tf.matmul(tensor, tf.transpose(tensor))


    def compute_style_cost(self):
        style_cost = 0
        for layer, coeff in self.config.style_cost_layers.items():
            a_S = self.style_models[layer](self.style_img)
            a_G = self.style_models[layer](self.gen_img)
            _, h, w, c = a_S.shape.as_list()
            GM_S = self.gram_matrix(a_S)
            GM_G = self.gram_matrix(a_G)
            layer_style_cost = 1/(4*(h*w*c)**2) * tf.reduce_sum(tf.pow(GM_S - GM_G, 2))
            style_cost += coeff * layer_style_cost
        return style_cost


    def compute_total_cost(self):
        J_c = self.compute_content_cost()
        J_s = self.compute_style_cost()
        J = self.config.alpha*J_c + self.config.beta*J_s
        optimizer = tf.train.AdamOptimizer(2.0)
        style_op = optimizer.minimize(J)
        return style_op, J


    @staticmethod
    def save_image(iter, np_array):
        fig = plt.figure(figsize=(8, 6))
        np_array = np_array*0.5 + 1
        plt.imshow(np_array)
        plt.axis('off')
        fig.savefig('step_{}.jpg'.format(iter))


    def stylize(self):
        self.gen_img = self.generate_noise_image()
        with self.sess as sess:
            writer = tf.summary.FileWriter(self.config.logdir, sess.graph)
            sess.run(tf.initialize_all_variables())
            for iter in range(1):
                _, cost = sess.run(self.compute_total_cost())
                print('Step = %i --> Cost = %f' %(iter, cost))
            #     if iter%10==0:
            #         gen_img_np = self.gen_img.eval()[0]
            #         self.save_image(iter, gen_img_np)
            # gen_img_np = self.gen_img.eval()[0]
            # self.save_image('final', gen_img_np)
            #

if __name__=='__main__':
    gatys = GatysStyleTransfer(content_file='piyush.jpeg', style_file='starry_nights.jpeg')
    gatys.stylize()
