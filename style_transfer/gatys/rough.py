
sess = tf.keras.backend.get_session()
saver = tf.train.Saver()
saver.save(sess, 'checkpoints/vgg19.ckpt')


v1 = tf.get_variable('block5_conv3/kernel', shape = [3, 3, 512, 512])
with tf.Session() as sess:
    saver.restore(sess, 'checkpoints/vgg19.ckpt')



latest_ckp = tf.train.latest_checkpoint('checkpoints/')
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')

from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/vgg19.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
