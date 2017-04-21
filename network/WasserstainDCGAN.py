import tensorflow as tf
import math
from utils import ops
import numpy as np
import time
import matplotlib.pyplot as plt

class WasserstainDCGAN(object):
    def __init__(self, params, c_dim=3):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=run_config)
        self.num_critic = params.num_critic
        self.clip = params.clip
        self.batch_size = params.batch_size
        self.num_initial_dimensions = params.num_initial_dimensions
        self.img_size = params.img_size
        self.c_dim = c_dim
        self.gf_dim = 64
        self.df_dim=64
        self.lr = params.learning_rate
        self.clip = params.clip
        self.n_epoch = params.n_epoch
        self.num_critic = params.num_critic


    def construct_network(self):
        self.X_d = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size, self.c_dim), name='X_d')
        self.X_g = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_initial_dimensions), name='X_g')

        # create generator
        with tf.variable_scope('generator'):
            self.P = self._make_generator(self.X_g)

        with tf.variable_scope('discriminator'):
            d_fake = self._make_descriminator(self.P)

        # get the weights
        self.w_g = [w for w in tf.global_variables() if 'generator' in w.name]
        self.w_d = [w for w in tf.global_variables() if 'discriminator' in w.name]

        with tf.variable_scope('discriminator') as scope:
            scope.reuse_variables()
            d_real = self._make_descriminator(self.X_d)
        # create losses
        self.loss_g = tf.reduce_mean(d_fake)
        self.loss_d = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)

        # compute and store discriminator probabilities
        self.d_real = tf.reduce_mean(d_real)
        self.d_fake = tf.reduce_mean(d_fake)
        self.p_real = tf.reduce_mean(tf.sigmoid(d_real))
        self.p_fake = tf.reduce_mean(tf.sigmoid(d_fake))

        # create an optimizer
        optimizer_g = tf.train.RMSPropOptimizer(self.lr)
        optimizer_d = tf.train.RMSPropOptimizer(self.lr)

        # get gradients
        gv_g = optimizer_g.compute_gradients(self.loss_g, self.w_g)
        gv_d = optimizer_d.compute_gradients(self.loss_d, self.w_d)

        # create training operation
        self.train_op_g = optimizer_g.apply_gradients(gv_g)
        self.train_op_d = optimizer_d.apply_gradients(gv_d)

        # clip the weights, so that they fall in [-c, c]
        self.clip_updates = [w.assign(tf.clip_by_value(w, -self.clip, self.clip)) for w in self.w_d]

    def fit(self, data, logdir='dcgan-run'):
        if tf.gfile.Exists(logdir):
            tf.gfile.DeleteRecursively(logdir)
        tf.gfile.MakeDirs(logdir)

        # init model
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # train the model
        step, g_step, epoch = 0, 0, 0
        num_imgs = len(data)
        batch_ids = num_imgs // self.batch_size
        print batch_ids
        while epoch < self.n_epoch:
            epoch += 1
            for idx in xrange(batch_ids):
                n_critic = 100 if g_step < 25 or (g_step + 1) % 500 == 0 else self.num_critic

                start_time = time.time()
                for i in range(n_critic):
                    losses_d = []

                    # load the batch
                    X_batch = data[idx*self.batch_size: (idx+1)*self.batch_size]
                    noise = np.random.rand(self.batch_size, self.num_initial_dimensions).astype('float32')
                    feed_dict = {
                        self.X_g: noise,
                        self.X_d: X_batch
                    }

                    # train the critic/discriminator
                    loss_d = self.train_d(feed_dict)
                    losses_d.append(loss_d)

                loss_d = np.array(losses_d).mean()

                # train the generator
                noise = np.random.rand(self.batch_size, self.num_initial_dimensions).astype('float32')
                # noise = np.random.uniform(-1.0, 1.0, [n_batch, 100]).astype('float32')
                feed_dict = {
                    self.X_g: noise,
                    self.X_d: X_batch
                }
                loss_g = self.train_g(feed_dict)
                g_step += 1

                if g_step < 100 or g_step % 100 == 0:
                    tot_time = time.time() - start_time
                    print 'Epoch: %3d, batch_idx: %3d, Gen step: %4d (%3.1f s), Disc loss: %.6f, Gen loss %.6f' % \
                          (epoch, idx , g_step, tot_time, loss_d, loss_g)

                # take samples
                if g_step % 100 == 0:
                    noise = np.random.rand(1, self.num_initial_dimensions).astype('float32')
                    samples = self.generate_samples(noise)
                    fname = logdir + '.mnist_samples-%d.png' % g_step
                    plt.imsave(fname,
                               samples,
                               cmap='gray')

    def generate_samples(self, noise):
        feed_dict = {self.X_g: noise}
        return self.sess.run(self.P, feed_dict=feed_dict)

    def train_g(self, feed_dict):
        _, loss_g = self.sess.run([self.train_op_g, self.loss_g], feed_dict=feed_dict)
        return loss_g

    def train_d(self, feed_dict):
        # clip the weights, so that they fall in [-c, c]
        self.sess.run(self.clip_updates, feed_dict=feed_dict)

        # take a step of RMSProp
        self.sess.run(self.train_op_d, feed_dict=feed_dict)

        # return discriminator loss
        return self.sess.run(self.loss_d, feed_dict=feed_dict)

    def _conv_out_size_same(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def _make_descriminator(self, input):
        h0 = ops.lrelu(ops.conv2d(input, self.df_dim, name='d_h0_conv'))
        h1 = ops.lrelu(ops.batch_norm(ops.conv2d(h0, self.df_dim*2, name='d_h1_conv'), name='d_bn1'))
        h2 = ops.lrelu(ops.batch_norm(ops.conv2d(h1, self.df_dim*4, name='d_h2_conv'), name='d_bn2'))
        h3 = ops.lrelu(ops.batch_norm(ops.conv2d(h2, self.df_dim*8, name='d_h3_conv'), name='d_bn3'))
        h4 = ops.lrelu(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        return h4

    def _make_generator(self, input):
        s_h, s_w = self.img_size, self.img_size
        s_h2, s_w2 = self._conv_out_size_same(s_h, 2), self._conv_out_size_same(s_w, 2)
        s_h4, s_w4 = self._conv_out_size_same(s_h2, 2), self._conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = self._conv_out_size_same(s_h4, 2), self._conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = self._conv_out_size_same(s_h8, 2), self._conv_out_size_same(s_w8, 2)
        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = ops.linear(
            input, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(ops.batch_norm(self.h0, name='g_bn0'))

        self.h1, self.h1_w, self.h1_b = ops.deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(ops.batch_norm(self.h1, name='g_bn1'))

        h2, self.h2_w, self.h2_b = ops.deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(ops.batch_norm(h2, name='g_bn2'))

        h3, self.h3_w, self.h3_b = ops.deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(ops.batch_norm(h3, name='g_bn3'))

        h4, self.h4_w, self.h4_b = ops.deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
        return h4
