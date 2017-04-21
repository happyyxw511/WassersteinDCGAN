import tensorflow as tf
from utils import load_data
from network.WasserstainDCGAN import WasserstainDCGAN
import cv2
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_dir', default_value='.', docstring='directory to save the data')
flags.DEFINE_string('dataset', default_value='mnist', docstring='dataset')
flags.DEFINE_float('learning_rate', default_value=5e-5, docstring='learning rate')
flags.DEFINE_float('clip', default_value=1e-2, docstring='clip value')
flags.DEFINE_integer('num_critic', default_value=5, docstring='number of critic training iterations')
flags.DEFINE_integer('batch_size', default_value=128, docstring='size of mini-batch')
flags.DEFINE_integer('num_samples', default_value=5, docstring='the number of generated samples')
flags.DEFINE_integer('num_initial_dimensions', default_value=100, docstring='the number of input dimensions for generator')
flags.DEFINE_integer('img_size', default_value=28, docstring='the size of the image')
flags.DEFINE_integer('n_epoch', default_value=10, docstring='the number of training epoch')


def main():
    if FLAGS.dataset == 'mnist':
        X_train = load_data.load_mnist()
        c_dim = 1
    elif FLAGS.dataset == 'cifar10':
        X_train, y_train, X_test, y_tes = load_data.load_cifar10(FLAGS.dataset_dir)
        c_dim = 3
    model = WasserstainDCGAN(FLAGS, c_dim=c_dim)
    model.construct_network()
    model.fit(X_train)
    samples = model.generate_samples(FLAGS.num_samples)
    if FLAGS.show:
        for sample in samples:
            cv2.imshow('generated_imgs', sample)
            cv2.waitKey()



if __name__ == '__main__':
    main()
