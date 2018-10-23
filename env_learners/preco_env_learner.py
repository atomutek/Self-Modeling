import time
from collections import deque

import numpy as np
import tensorflow as tf

from env_learners.env_learner import EnvLearner
from misc import losses


def predictor(x, h):
    # multiply = tf.shape(x[0])[0]
    # initial_h = tf.ones(multiply,1)*h
    rnn_cell = tf.contrib.rnn.GRUCell(1024, name='predictor')
    outputs, states = tf.nn.static_rnn(rnn_cell, x, initial_state=h, dtype=tf.float32)
    return outputs, states

def corrector(x, h=None):
    rnn_cell = tf.contrib.rnn.GRUCell(1024, name='corrector')
    outputs, states = tf.nn.static_rnn(rnn_cell, x, initial_state=h, dtype=tf.float32)
    return outputs, states

def decode(x, state_dim, drop_rate=0.5, num_components=2):
    x = tf.layers.batch_normalization(x, name='bn0')

    stdev = tf.layers.dense(x, 128, name='stdev_hd1')
    stdev = tf.layers.dense(stdev, state_dim*num_components, name='stdev_hd2')
    stdev = tf.nn.tanh(stdev, name='stdev_out')

    mean = tf.layers.dense(x, 128, name='mean_hd1')
    mean = tf.layers.dense(mean, state_dim*num_components, name='mean_hd2')
    mean = tf.nn.tanh(mean, name='mean_out')

    mix = tf.layers.dense(x, 128, name='mix_hd1')
    mix = tf.layers.dense(mix, num_components, name='mix_hd2')
    mix = tf.nn.tanh(mix, name='mix_out')

    # Normalization
    max_pi = tf.reduce_max(mix, 1, keep_dims=True)
    mix = tf.subtract(mix, max_pi)

    mix = tf.exp(mix)

    normalize_pi = tf.reciprocal(tf.reduce_sum(mix, 1, keep_dims=True))
    mix = tf.multiply(normalize_pi, mix)

    stdev = tf.exp(stdev)
    return mix, stdev, mean


class PreCoEnvLearner(EnvLearner):
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
        print('PreCo model:')
        print('Learning Rate: ' + str(lr))
        self.max_seq_len = 30

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

        with tf.variable_scope('PreCo', reuse=tf.AUTO_REUSE):

            tmp_a_seq = tf.split(self.a_seq, self.max_seq_len, 1)
            tmp_y_seq = tf.split(self.y_seq, self.max_seq_len, 1)


            corr_out, self.corr_hidden = corrector([self.x])
            # self.decoded_corr = decode(self.corr_hidden, self.state_dim)
            self.corr_pi, self.corr_sigma, self.corr_mu = decode(self.corr_hidden, self.state_dim)
            l = self.get_lossfunc(self.corr_pi, self.corr_sigma, self.corr_mu, self.x)
            self.loss_corr += l

            pred_out, self.pred_hidden = predictor([tmp_a_seq[0]], self.corr_hidden)
            pi, sigma, mu = decode(self.pred_hidden, self.state_dim)
            l = self.get_lossfunc(pi, sigma, mu, tmp_y_seq[0])
            self.loss_single = l
            self.loss_seq += l

            pred_out, self.pred_hidden_open = predictor([tmp_a_seq[0]], self.state_in)
            self.decoded_pred_open = decode(self.pred_hidden, self.state_dim)

            last_state = self.pred_hidden
            for i in range(1, self.max_seq_len):
                a = tmp_a_seq[i]
                y = tmp_y_seq[i]
                pred_out, last_state = predictor([a], last_state)
                pi, sigma, mu = decode(last_state, self.state_dim)
                l = self.get_lossfunc(pi, sigma, mu, y)
                self.loss_seq += l

            # Losses for the corrector, and single/sequence predictor, each one should be executed for each train step
            self.corr_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_corr)
            self.pred_seq_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_seq)
            self.pred_single_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_single)

            # Testing
            _, self.corr_hidden_out = corrector([self.x])
            _, self.pred_hidden_out = predictor([self.a], self.corr_hidden_out)
            self.pi_decoded, self.sigma_decoded, self.mu_decoded = decode(self.pred_hidden_out, self.state_dim)
            # self.decoded = self.generate_ensemble(self.pi_decoded, self.sigma_decoded, self.mu_decoded)

            _, self.pred_hidden_open = predictor([self.a], self.state_in)
            # self.decoded_pred_open = decode(self.pred_hidden_open, self.state_dim)
            self.pi_decoded_open, self.sigma_decoded_open, self.mu_decoded_open = \
                decode(self.pred_hidden_open, self.state_dim)
            # self.decoded_pred_open = self.generate_ensemble(self.pi_decoded_open, self.sigma_decoded_open, self.mu_decoded_open)

    # Fromhttp://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/
    def tf_normal(self, y, mu, sigma, n_mix):
        import math
        mu_split = tf.split(mu, num_or_size_splits=n_mix, axis=1)
        sigma_split = tf.split(sigma, num_or_size_splits=n_mix, axis=1)
        result = tf.subtract(y, mu_split)
        result = tf.multiply(result,tf.reciprocal(sigma_split))
        result = -tf.square(result)/2
        result = tf.multiply(tf.exp(result),tf.reciprocal(sigma_split)) / math.sqrt(2*math.pi)
        # results = [result[i] for i in range(result.get_shape().as_list()[0])]
        # result = tf.concat(results, axis=1)
        result = tf.transpose(result)
        return result

    def get_lossfunc(self, out_pi, out_sigma, out_mu, y):
        n_mix = out_pi.get_shape().as_list()[1]
        result = self.tf_normal(y, out_mu, out_sigma, n_mix)
        result = tf.multiply(result, out_pi)
        result = tf.reduce_sum(result, 2, keep_dims=True)
        result = -tf.log(result)
        result = tf.reduce_mean(result)
        return result
    def get_pi_idx(self, x, pdf):
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x): #? '>' was '&gt;'
                return i
        print 'error with sampling ensemble'
        return -1

    def generate_ensemble(self, out_pi, out_sigma, out_mu, M = 1):
        NTEST = out_pi.shape[0]
        split_mu = np.split(out_mu, out_pi.shape[1], axis=1)
        split_sigma = np.split(out_sigma, out_pi.shape[1], axis=1)
        result = np.random.rand(NTEST, M, split_mu[0].shape[1]) # initially random [0, 1]
        rn = np.random.randn(NTEST, M, split_mu[0].shape[1]) # normal random matrix (0.0, 1.0)

        # transforms result into random ensembles
        for j in range(0, M):
            for i in range(0, NTEST):
                for k in range(split_mu[0].shape[1]):
                    idx = self.get_pi_idx(result[i, j, k], out_pi[i])
                    mu = split_mu[idx][i, k]
                    std = split_sigma[idx][i, k]
                    result[i, j, k] = mu + rn[i, j, k]*std
        return result
    # End From http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/

    def initialize(self, session, load=False):
        self.sess = session
        if not load:
            self.sess.run(tf.global_variables_initializer())

    def train_epoch(self, data):
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=32)
        Single = 0.0
        Seq = 0.0
        Corr = 0.0
        for i in range(len(X)):
            pi, sigma, mu, single, seq, corr, _, _, _ = self.sess.run([ self.corr_pi, self.corr_sigma, self.corr_mu, self.loss_single, self.loss_seq, self.loss_corr,
                                                        self.pred_seq_train_step, self.pred_single_train_step,
                                                        self.corr_train_step],
                                     feed_dict={
                                                self.x: X[i][:,:self.state_dim],
                                                self.a_seq: A[i],
                                                self.y_seq: S[i]
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
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=32)
        Single = 0.0
        Seq = 0.0
        Corr = 0.0
        for i in range(len(X)):
            single, seq, corr, tmp_y_seq = self.sess.run([self.loss_single, self.loss_seq, self.loss_corr, tf.split(self.y_seq, self.max_seq_len, 1)],
                                     feed_dict={
                                                self.x: X[i][:,:self.state_dim],
                                                self.a_seq: A[i],
                                                self.y_seq: S[i]
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
            pi, sigma, mu = self.sess.run([self.pi_decoded, self.sigma_decoded, self.mu_decoded], feed_dict={self.x: obs, self.a: action})
            new_obs = self.generate_ensemble(pi, sigma, mu)
            new_obs = new_obs[0][0]*self.state_mul_const
            return new_obs
        else:
            action = np.array([action_in/self.act_mul_const])
            pi, sigma, mu, self.last_state = self.sess.run([self.pi_decoded_open, self.sigma_decoded_open, self.mu_decoded_open, self.pred_hidden_open],
                                                     feed_dict={self.a: action,
                                                                self.state_in:self.last_state})
            new_obs = self.generate_ensemble(pi, sigma, mu)
            new_obs = new_obs[0][0]*self.state_mul_const
            return new_obs

