
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
    num_iters = 1000
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

def read_image(file_name, name):
    """
    Reads image from file path and does basic preprocessing
    """
    img = Image.open(file_name)
    img = img.resize(config.img_size)
    img = np.array(img).astype(np.float32)/127.5 - 1
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img, name=name, dtype=tf.float32)

def generate_noise_image():
    """
    Returns a gen_img which is a linear combination of content_img and noise_img
    """
    noise_img = tf.random_normal((1,) + tuple(reversed(config.img_size)) + \
                (config.num_channels,), name='noise_img')
    gen_img = config.noise_ratio*noise_img + (1-config.noise_ratio) * \
              content_img
    return tf.Variable(gen_img, name='gen_img')

def get_intermediate_model(layer_name):
    """
    Creates Model objects with given layer name as output layer and config's
    pretrained model's input as input layer.
    """
    layer_model = Model(inputs=config.pretrained_model.input, \
                        outputs=config.pretrained_model.get_layer(\
                                layer_name).output
                       )
    return layer_model

def compute_content_cost(content_img, gen_img):
    """
    Calculates content cost between C and G.
    """
    a_C = content_model(content_img)
    a_G = content_model(gen_img)
    return 1/4 * tf.reduce_mean(tf.pow(a_C - a_G, 2))

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_style_cost(style_img, gen_img):
    """
    Calculates style cost between S and G.
    """
    style_cost = 0
    for layer, coeff in config.style_cost_layers.items():
        a_S = style_models[layer](style_img)
        a_G = style_models[layer](gen_img)
        _, n_H, n_W, n_C = a_S.shape.as_list()
        a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        layer_style_cost = 1/(4*(n_H*n_W*n_C)**2) * tf.reduce_sum(tf.pow(GS - GG, 2))
        style_cost += coeff * layer_style_cost
    return style_cost

def compute_total_cost(C, S, G):
    """
    Calculates total cost
    """
    J_c = compute_content_cost(C, G)
    J_s = compute_style_cost(S, G)
    J = config.alpha*J_c + config.beta*J_s
    return J

def save_image(iter, np_array):
    """
    utility method to save image
    """
    fig = plt.figure(figsize=(8, 6))
    np_array = np_array*0.5 + 1
    plt.imshow(np_array)
    plt.axis('off')
    fig.savefig('step_{}.jpg'.format(iter))

config = Config()

with tf.name_scope('content_model'):
    content_model = get_intermediate_model(layer_name=config.content_cost_layer)
    content_model.trainable = False

style_models = {}
for layer in config.style_cost_layers:
    with tf.name_scope('style_model_%s' %layer):
        style_models[layer] =  get_intermediate_model(layer_name=layer)
        style_models[layer].trainable = False


content_img = read_image(file_name='cat.jpeg', name='content_img')
style_img = read_image(file_name='starry_nights.jpeg', name='style_img')
gen_img = generate_noise_image()

total_cost = compute_total_cost(C=content_img, S=style_img, G=gen_img)
optimizer = tf.train.AdamOptimizer(2.0)
style_op = optimizer.minimize(total_cost, var_list=[gen_img])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
for iter in range(config.num_iters):
    cost, _ = sess.run([total_cost, style_op])
    print("Iteration %i Cost %f" %(iter, cost))
    if iter%10==0:
        curr_img = sess.run(gen_img)[0]
        save_image(iter, curr_img)
    print("Iteration %i Cost %f" %(iter, cost))
    curr_img = sess.run(gen_img)[0]
    save_image('final', curr_img)
