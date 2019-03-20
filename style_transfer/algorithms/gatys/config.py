
class Config:
    img_size = (400, 300)
    noise_ratio = 0.6
    content_cost_layer = 'block4_conv2'
    style_cost_layers = {'block1_conv1': 0.2,
                         'block2_conv1': 0.2,
                         'block3_conv1': 0.2,
                         'block4_conv1': 0.2,
                         'block5_conv1': 0.2
                        }
