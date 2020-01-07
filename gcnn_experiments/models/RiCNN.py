import tensorflow as tf
import ricnn

VALID='VALID'



batch_size = 100

default_filter_size = 3
n_filters = 7
rotations = 8
stride = 1
pool_stride = 2
dft_n_filters=20

def r_e_layer(layer_order, previous_layer,output_size,n_filters_previous_layer, n_rotations=rotations, filter_size=default_filter_size):
    weights = tf.get_variable('weights-conv'+str(layer_order), [filter_size, filter_size,
                                                n_filters_previous_layer, n_filters])
    biases = tf.get_variable('biases-conv'+str(layer_order), [n_filters])
    output = ricnn.rotation_equivariant_conv2d(
        previous_layer, weights,
        [batch_size, output_size, output_size, n_filters_previous_layer],
        [filter_size, filter_size, n_filters_previous_layer, n_filters],
        n_rotations, stride=stride) + biases
    if filter_size != 4:
        output = tf.layers.batch_normalization(output)
        output = tf.nn.relu(output)
    output_size = ricnn.calculate_reconv_output_size(
        output_size, filter_size, pool_stride=stride)
    n_filters_previous_layer = n_filters

    return output, output_size, n_filters_previous_layer

def dft_layer(previous_layer,output_size,n_filters_previous_layer, n_rotations=rotations):
    filter_size = output_size
    weights = tf.get_variable('weights-dft', [filter_size, filter_size,
                                              n_filters_previous_layer, dft_n_filters])
    biases = tf.get_variable('biases-dft', [dft_n_filters])
    output = ricnn.conv_to_circular_shift(previous_layer, weights, biases, n_rotations)
    output = tf.nn.relu(output)
    output = ricnn.dft2d_transition(output, n_rotations, batch_size, output_size,
                                    n_filters_previous_layer, dft_n_filters)
    dft_size = ricnn.calculate_dft_output_size(dft_n_filters, n_rotations)
    return output, dft_size

def get_model(x, x_size, x_depth, y_size):
    output, output_size, n_filters_previous_layer = r_e_layer(1,x, x_size,x_depth)
    output, output_size, n_filters_previous_layer = r_e_layer(2,output, output_size, n_filters_previous_layer)
    output = tf.nn.max_pool(output, [1, pool_stride, pool_stride, 1],
                                [1, pool_stride, pool_stride, 1], padding='VALID')
    output_size = int(output.shape[1])
    output, output_size, n_filters_previous_layer = r_e_layer(3,output, output_size, n_filters_previous_layer)
    output, output_size, n_filters_previous_layer = r_e_layer(4,output, output_size, n_filters_previous_layer)
    output, output_size, n_filters_previous_layer = r_e_layer(5,output, output_size, n_filters_previous_layer)
    output, output_size, n_filters_previous_layer = r_e_layer(6,output, output_size, n_filters_previous_layer, 4)
    output, output_size, n_filters_previous_layer = r_e_layer(7, output, output_size, n_filters_previous_layer, 4, 4)
    output_size = int(output.shape[1])
    output, dft_size = dft_layer(output, output_size, n_filters_previous_layer, 4)
    images_flat = tf.contrib.layers.flatten(output)
    l7 = tf.contrib.layers.fully_connected(images_flat, y_size, tf.nn.softmax)
    return l7


