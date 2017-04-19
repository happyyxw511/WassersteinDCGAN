import tensorflow as tf


class WasserstainDCGAN(object):
    def __init__(self, params):
        self.num_critic = params.num_critic
        self.clip = params.clip
        self.batch_size = params.batch_size
        self.num_initial_dimensions = params.num_initial_dimensions


    def construct_network(self):

        # create generator
        with tf.name_scope('generator'):
            X_g = tf.placeholder(tf.float32, shape=(self.num_initial_dimensions, ), name='X_g')
            generator = self._make_generator(TBD)

    def fit(self, data):
        pass

    def generate_samples(self, num_samples):
        pass

    def _make_descriminator(self):
        pass

    def _make_generator(self):
        pass
