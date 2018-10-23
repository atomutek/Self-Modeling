import time
from collections import deque

import numpy as np
import tensorflow as tf

from env_learners.env_learner import EnvLearner
from misc import losses


def predictor(x, h, z_dim=1024):
    rnn_cell = tf.contrib.rnn.GRUCell(z_dim, name='predictor')
    outputs, states = tf.nn.static_rnn(rnn_cell, x, initial_state=h, dtype=tf.float32)
    return outputs, states

def corrector(x, h=None, z_dim=1024):
    rnn_cell = tf.contrib.rnn.GRUCell(z_dim, name='corrector')
    outputs, states = tf.nn.static_rnn(rnn_cell, x, initial_state=h, dtype=tf.float32)
    return outputs, states

def KLD_loss(latent_var, latent_mean):
    return -0.5 * tf.reduce_mean(1.0 + latent_var - tf.pow(latent_mean, 2) - tf.exp(latent_var))


def decode(x, out_dim, drop_rate=0.5, is_training=False):
    x = tf.layers.batch_normalization(x, name='bn0')
    x = tf.layers.dense(x, 512, name='mlp0')
    x = tf.layers.dropout(x, rate=drop_rate, name='do0')
    x = tf.nn.relu(x, name='rl0')

    x = tf.layers.batch_normalization(x, name='bn1')
    x = tf.layers.dense(x, 256, name='mlp1')
    x = tf.layers.dropout(x, rate=drop_rate, name='do1')
    x = tf.nn.relu(x, name='rl1')

    x = tf.layers.batch_normalization(x, name='bn2')
    x = tf.layers.dense(x, 128, name='mlp2')
    x = tf.layers.dropout(x, rate=drop_rate, name='do2')
    x = tf.nn.relu(x, name='rl2')

    x = tf.layers.batch_normalization(x, name='bn3')
    x = tf.layers.dense(x, 256, name='mlp3')
    x = tf.layers.dropout(x, rate=drop_rate, name='do3')
    x = tf.nn.relu(x, name='rl3')

    x = tf.layers.batch_normalization(x, name='bn4')
    x = tf.layers.dense(x, 512, name='mlp4')
    x = tf.layers.dropout(x, rate=drop_rate, name='do4')
    x = tf.nn.relu(x, name='rl4')

    x = tf.layers.batch_normalization(x, name='bn5')
    x = tf.layers.dense(x, out_dim, name='mlp5')

    return tf.nn.tanh(x, name='th0'), x

class PreCoWAEEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)
        # Initialization

        self.latent_size = 1024
        self.last_r = np.array([0.0]).flatten()
        self.buffer = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
        dropout_rate = 0.5
        lr = 1e-5
        print('General Stats: ')
        print('Drop Rate: ' + str(dropout_rate))
        print('Buffer Len: ' + str(self.buff_len))
        print('PreCo WAE model:')
        print('Learning Rate: ' + str(lr))
        self.max_seq_len = 30

        # WAE configs
        self.e_noise='implicit'
        self.wae_lambda= 100 # Double check in paper to see if they gave a value
        self.pz_scale = 1
        self.noise = tf.placeholder(tf.float32, [None] + [1024], name='noise_ph')

        """ State Prediction """
        self.x = tf.placeholder(dtype=tf.float32, shape=([None, self.state_dim]))
        self.a = tf.placeholder(dtype=tf.float32, shape=([None, self.act_dim]))

        self.a_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.act_dim*self.max_seq_len]))
        self.y_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.state_dim*self.max_seq_len]))

        self.state_in = tf.placeholder(dtype=tf.float32, shape=([None, self.latent_size]))

        self.loss_seq = 0
        self.loss_corr = 0
        self.loss_single = 0

        self.last_state = None

        with tf.variable_scope('PreCoWAE', reuse=tf.AUTO_REUSE):

            tmp_a_seq = tf.split(self.a_seq, self.max_seq_len, 1)
            tmp_y_seq = tf.split(self.y_seq, self.max_seq_len, 1)


            corr_out, self.corr_hidden = corrector([self.x])
            # self.decoded_corr = decode(self.corr_hidden, self.state_dim)
            # self.loss_corr += losses.loss_p(self.decoded_corr, self.x)
            self.loss_corr += self.wae_loss(self.corr_hidden, self.x)

            pred_out, self.pred_hidden = predictor([tmp_a_seq[0]], self.corr_hidden)
            # self.decoded_pred = decode(self.pred_hidden, self.state_dim)
            # self.loss_single += losses.loss_p(self.decoded_pred, tmp_y_seq[0])
            self.loss_single += self.wae_loss(self.pred_hidden, tmp_y_seq[0])
            self.loss_seq += self.loss_single

            pred_out, self.pred_hidden_open = predictor([tmp_a_seq[0]], self.state_in)
            self.decoded_pred_open = decode(self.pred_hidden, self.state_dim)

            last_state = self.pred_hidden
            for i in range(1, self.max_seq_len):
                a = tmp_a_seq[i]
                y = tmp_y_seq[i]
                pred_out, last_state = predictor([a], last_state)
                # decoded_pred = decode(last_state, self.state_dim)
                # l = losses.loss_p(decoded_pred, y)
                # Add WAE Loss here
                l = self.wae_loss(last_state, y)
                self.loss_seq += l

            # Losses for the corrector, and single/sequence predictor, each one should be executed for each train step
            self.corr_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_corr)
            self.pred_seq_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_seq)
            self.pred_single_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_single)

            # Testing
            _, self.corr_hidden_out = corrector([self.x])
            _, self.pred_hidden_out = predictor([self.a], self.corr_hidden_out)
            self.decoded = decode(self.pred_hidden_out, out_dim=self.state_dim)
            # self.decoded = decode(self.pred_hidden_out, self.state_dim)

            _, self.pred_hidden_open = predictor([self.a], self.state_in)
            self.decoded_pred_open = decode(self.pred_hidden_open, out_dim=self.state_dim)
            # self.decoded_pred_open = decode(self.pred_hidden_open, self.state_dim)


    ## WAE
    # Adapted From: https://github.com/tolstikhin/wae
    # Note to self, make this a seperate function as it's too complex here
    # res = encoder(inputs=sample_points, is_training=self.is_training)
    def wae_loss(self, res, sample_points):
        # if self.e_noise in ('deterministic', 'implicit', 'add_noise'):
        enc_mean, enc_sigmas = None, None
        # if self.e_noise == 'implicit':
        #     encoded, encoder_A = res
        # else:
        encoded = res
        # elif e_noise == 'gaussian':
        #     # Encoder outputs means and variances of Gaussian
        #     enc_mean, enc_sigmas = res[0]
        #     enc_sigmas = tf.clip_by_value(enc_sigmas, -50, 50)
        #     self.enc_mean, self.enc_sigmas = enc_mean, enc_sigmas
        #     # zdim=1024
        #     eps = tf.random_normal((-1, 1024),
        #                            0., 1., dtype=tf.float32)
        #     self.encoded = self.enc_mean + tf.multiply(
        #         eps, tf.sqrt(1e-8 + tf.exp(self.enc_sigmas)))
        #     # self.encoded = self.enc_mean + tf.multiply(
        #     #     eps, tf.exp(self.enc_sigmas / 2.))

        # Decode the points encoded above (i.e. reconstruct)
        reconstructed, reconstructed_logits = decode(encoded, out_dim=self.state_dim)

        # Decode the content of sample_noise
        # self.decoded, self.decoded_logits = decode(self.sample_noise, out_dim=self.state_dim, is_training=self.is_training)

        # -- Objectives, losses, penalties

        penalty, self.loss_gan = self.matching_penalty(encoded, self.noise)
        loss_reconstruct = self.reconstruction_loss(sample_points, reconstructed)
        wae_objective = loss_reconstruct + self.wae_lambda * penalty
        return wae_objective

    def matching_penalty(self, encoded, sample_noise):
        loss_gan = None
        sample_qz = encoded
        z_test = 'mmd'
        sample_pz = sample_noise
        # if z_test == 'gan':
        #     loss_gan, loss_match = self.gan_penalty(sample_qz, sample_pz)
        # if z_test == 'mmd':
        loss_match = self.mmd_penalty(sample_qz, sample_pz)
        # elif z_test == 'mmdpp':
        #     loss_match = improved_wae.mmdpp_penalty(
        #         opts, self, sample_pz)
        # elif z_test == 'mmdppp':
        #     loss_match = improved_wae.mmdpp_1d_penalty(
        #         opts, self, sample_pz)
        # else:
        #     assert False, 'Unknown penalty %s' % opts['z_test']
        return loss_match, loss_gan
    def mmd_penalty(self, sample_qz, sample_pz):
        # opts = self.opts
        sigma2_p = self.pz_scale ** 2
        # kernel = 'RBF'

        # get_batch_size
        # n = utils.get_batch_size(sample_qz)
        n = tf.cast(tf.shape(sample_qz)[0], tf.float32)
        n = tf.cast(n, tf.int32)
        nf = tf.cast(n, tf.float32)
        half_size = (n * n - n) / 2

        norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
        dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
        distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

        norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
        dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
        distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

        dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
        distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

        # if kernel == 'RBF':
        #     # Median heuristic for the sigma^2 of Gaussian kernel
        #     sigma2_k = tf.nn.top_k(
        #         tf.reshape(distances, [-1]), half_size).values[half_size - 1]
        #     sigma2_k += tf.nn.top_k(
        #         tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
        #     # if opts['verbose']:
        #     #     sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
        #     res1 = tf.exp( - distances_qz / 2. / sigma2_k)
        #     res1 += tf.exp( - distances_pz / 2. / sigma2_k)
        #     res1 = tf.multiply(res1, 1. - tf.eye(n))
        #     res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        #     res2 = tf.exp( - distances / 2. / sigma2_k)
        #     res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        #     stat = res1 - res2
        # elif kernel == 'IMQ':
        # if opts['pz'] == 'normal':
        Cbase = 2. * 1024 * sigma2_p
        # elif opts['pz'] == 'sphere':
        #     Cbase = 2.
        # elif opts['pz'] == 'uniform':
        #     # E ||x - y||^2 = E[sum (xi - yi)^2]
        #     #               = zdim E[(xi - yi)^2]
        #     #               = const * zdim
        #     Cbase = opts['zdim']
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + distances_qz)
            res1 += C / (C + distances_pz)
            res1 = tf.multiply(res1, 1. - tf.eye(n))
            res1 = tf.reduce_sum(res1) / (nf * nf - nf)
            res2 = C / (C + distances)
            res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
            stat += res1 - res2
        return stat

    # def gan_penalty(self, sample_qz, sample_pz):
    #     # opts = self.opts
    #     # Pz = Qz test based on GAN in the Z space
    #     logits_Pz = z_adversary(opts, sample_pz)
    #     logits_Qz = z_adversary(opts, sample_qz, reuse=True)
    #     loss_Pz = tf.reduce_mean(
    #         tf.nn.sigmoid_cross_entropy_with_logits(
    #             logits=logits_Pz, labels=tf.ones_like(logits_Pz)))
    #     loss_Qz = tf.reduce_mean(
    #         tf.nn.sigmoid_cross_entropy_with_logits(
    #             logits=logits_Qz, labels=tf.zeros_like(logits_Qz)))
    #     loss_Qz_trick = tf.reduce_mean(
    #         tf.nn.sigmoid_cross_entropy_with_logits(
    #             logits=logits_Qz, labels=tf.ones_like(logits_Qz)))
    #     loss_adversary = self.wae_lambda * (loss_Pz + loss_Qz)
    #     # Non-saturating loss trick
    #     loss_match = loss_Qz_trick
    #     return (loss_adversary, logits_Pz, logits_Qz), loss_match

    @staticmethod
    def reconstruction_loss(real, reconstr):
        # real = self.sample_points
        # reconstr = self.reconstructed
        # if opts['cost'] == 'l2':
        #     # c(x,y) = ||x - y||_2
        #     loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
        #     loss = 0.2 * tf.reduce_mean(tf.sqrt(1e-08 + loss))
        # elif opts['cost'] == 'l2sq':
            # c(x,y) = ||x - y||_2^2
        # loss = tf.reduce_sum(tf.square(real - reconstr), axis=[1, 2, 3])
        loss = tf.reduce_sum(tf.square(real - reconstr))
        loss = 0.05 * tf.reduce_mean(loss)
        # elif opts['cost'] == 'l1':
        #     # c(x,y) = ||x - y||_1
        #     loss = tf.reduce_sum(tf.abs(real - reconstr), axis=[1, 2, 3])
        #     loss = 0.02 * tf.reduce_mean(loss)
        # else:
        #     assert False, 'Unknown cost function %s' % opts['cost']
        return loss
    def sample_pz(self, num=32):
        noise = None
        # distr = opts['pz']
        # if distr == 'uniform':
        #     noise = np.random.uniform(
        #         -1, 1, [num, opts["zdim"]]).astype(np.float32)
        # elif distr in ('normal', 'sphere'):
        mean = np.zeros(1024)
        cov = np.identity(1024)
        noise = np.random.multivariate_normal(
            mean, cov, num).astype(np.float32)
            # if distr == 'sphere':
            #     noise = noise / np.sqrt(
            #         np.sum(noise * noise, axis=1))[:, np.newaxis]
        return self.pz_scale * noise
    ## End WAE

    def initialize(self, session, load=False):
        self.sess = session
        if not load:
            self.sess.run(tf.global_variables_initializer())

    def train_epoch(self, data):
        batch_len = 32
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=batch_len)
        Single = 0.0
        Seq = 0.0
        Corr = 0.0
        for i in range(len(X)):
            batch_noise = self.sample_pz(batch_len)
            single, seq, corr, _, _, _ = self.sess.run([self.loss_single, self.loss_seq, self.loss_corr,
                                                        self.pred_seq_train_step, self.pred_single_train_step, self.corr_train_step],
                                     feed_dict={
                                                self.x: X[i][:,:self.state_dim],
                                                self.a_seq: A[i],
                                                self.y_seq: S[i],
                                                self.noise: batch_noise,
                                               })
            Single += single
            Seq += seq
            Corr += corr

        return Single / len(X), Seq / len(X), Corr / len(X)

    def train(self, train, total_steps, valid=None, log_interval=10, early_stopping=-1, saver=None, save_str=None, verbose=True):
        min_loss = 10000000000
        stop_count = 0

        seq_i = 0
        seq_idx = [1] * (self.max_seq_len - self.seq_len + 1)
        for j in range(1, self.max_seq_len - self.seq_len + 1):
            seq_tmp = self.max_seq_len - j
            seq_idx[j] = (seq_tmp + 1) * seq_idx[j - 1] / seq_tmp
        seq_idx.reverse()
        mul_const = total_steps / sum(seq_idx)
        for j in range(len(seq_idx)):
            seq_idx[j] = round(mul_const * seq_idx[j])
            if j > 0:
                seq_idx[j] += seq_idx[j - 1]
        for i in range(total_steps):
            if i == seq_idx[seq_i] and self.seq_len < self.max_seq_len:
                self.seq_len += 1
                seq_i += 1

            if i % log_interval == 0 and i > 0 and valid is not None:
                (single, seq, corr) = self.get_loss(valid)
                print('Epoch: ' + str(i) + '/' + str(total_steps))
                print('Valid Single: ' + str(single))
                print('Valid Seq: ' + str(seq))
                print('Valid Corr: ' + str(corr))
                print('')
                if saver is not None and save_str is not None:
                    save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
                    print("Model saved in path: %s" % save_path)
            start = time.time()
            (single, seq, corr) = self.train_epoch(train)
            duration = time.time() - start
            if stop_count > early_stopping and early_stopping > 0:
                break
            if i % log_interval != 0 and i > 0:
                print('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's')
                print('Train Single: ' + str(single))
                print('Train Seq: ' + str(seq))
                print('Train Corr: ' + str(corr))
                print('')
        if valid is not None:
            (single, seq, corr) = self.get_loss(valid)
            print('Final Epoch')
            print('Valid Single: ' + str(single))
            print('Valid Seq: ' + str(seq))
            print('Valid Corr: ' + str(corr))
            print('')
        if saver is not None and save_str is not None:
            save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
            print("Final Model saved in path: %s" % save_path)

    def get_loss(self, data):
        batch_len=32
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=batch_len)
        Single = 0.0
        Seq = 0.0
        Corr = 0.0
        for i in range(len(X)):
            batch_noise = self.sample_pz(batch_len)
            single, seq, corr, tmp_y_seq = self.sess.run([self.loss_single, self.loss_seq, self.loss_corr, tf.split(self.y_seq, self.max_seq_len, 1)],
                                     feed_dict={
                                                self.x: X[i][:,:self.state_dim],
                                                self.a_seq: A[i],
                                                self.y_seq: S[i],
                                                self.noise: batch_noise,
                                               })
            # diff = out-tmp_y_seq[0]
            Single += single
            Seq += seq
            Corr += corr
        return Single / len(X), Seq / len(X), Corr / len(X)

    def reset(self, obs_in):
        obs = obs_in/self.state_mul_const
        obs = np.array([obs])
        self.last_state = self.sess.run([self.corr_hidden_out], feed_dict={self.x:obs})[0]
        return obs_in

    def step(self, action_in, obs_in=None, episode_step=None, save=True, buff=None):
        if obs_in is not None:
            # action = np.zeros(self.act_dim*self.max_seq_len)
            # action[:action_in.shape[0]] = action_in/self.act_mul_const
            # action = np.array([action])
            # obs = obs_in/self.state_mul_const
            # obs = np.array([obs])
            # new_obs, self.last_state = self.sess.run([self.decoded_pred, self.pred_hidden],
            #                                          feed_dict={self.x: obs,self.a_seq: action,})
            # new_obs = new_obs[0]
            action = np.array([action_in/self.act_mul_const])
            obs = np.array([obs_in/self.state_mul_const])
            new_obs = self.sess.run([self.decoded], feed_dict={self.x: obs, self.a: action})
            new_obs = new_obs[0][0][0]*self.state_mul_const
            return new_obs
        else:
            action = np.array([action_in/self.act_mul_const])
            new_obs, self.last_state = self.sess.run([self.decoded_pred_open, self.pred_hidden_open],
                                                     feed_dict={self.a: action,
                                                                self.state_in:self.last_state
                                                                })
            new_obs = new_obs[0][0]*self.state_mul_const
            return new_obs

