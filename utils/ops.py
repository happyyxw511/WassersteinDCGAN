import math
import numpy as np 
import tensorflow as tf

print tf.__version__

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

def batch_norm(net, name, phase_train, axes=None, decay = 0.999):
    eps = 1e-3
    if not axes:
        axes = [0, 1, 2]
        channels = net.get_shape()[3].value
        var_shape = [channels]
    else:
        channels = net.get_shape()[1].value
        var_shape = [channels]
    with tf.variable_scope(name):
        beta = tf.get_variable('beta', var_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable('gamma', var_shape, initializer=tf.ones_initializer())
        pop_mean = tf.get_variable('pop_mean', var_shape, initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', var_shape, initializer=tf.zeros_initializer(), trainable=False)

        def mean_var_with_update():
            batch_mean, batch_var = tf.nn.moments(net, axes, name='moments')
            train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1-decay))
            train_var = tf.assign(pop_var, pop_var*decay + batch_var*(1-decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                           mean_var_with_update,
                           lambda: (pop_mean, pop_var))

    #return net
    return tf.nn.batch_normalization(net, mean, var, beta, gamma, eps, name=name)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))

    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
              strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
