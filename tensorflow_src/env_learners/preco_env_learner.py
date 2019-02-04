import time
from collections import deque

import numpy as np
import tensorflow_src as tf

from tensorflow_src.env_learners import EnvLearner


def predictor(x, h):
    # multiply = tf.shape(x[0])[0]
    # initial_h = tf.ones(multiply,1)*h
    rnn_cell = tf.contrib.rnn.GRUCell(128, name='predictor')
    outputs, states = tf.nn.static_rnn(rnn_cell, x, initial_state=h, dtype=tf.float32)
    return outputs, states

def corrector(x, h=None):
    rnn_cell = tf.contrib.rnn.GRUCell(128, name='corrector')
    outputs, states = tf.nn.static_rnn(rnn_cell, x, initial_state=h, dtype=tf.float32)
    return outputs, states

def decode(x, state_dim, num_components=None, drop_rate=0.5,):

    # x = tf.layers.batch_normalization(x, name='bn0')
    x = tf.layers.dense(x, 128, name='mlp0')
    # x = tf.layers.dropout(x, rate=drop_rate, name='do0')
    # x = tf.nn.relu(x, name='rl0')

    # # x = tf.layers.batch_normalization(x, name='bn1')
    # x = tf.layers.dense(x, 256, name='mlp1')
    # # x = tf.layers.dropout(x, rate=drop_rate, name='do1')
    # x = tf.nn.tanh(x, name='rl1')

    stdev = tf.layers.dense(x, 128, name='stdev_hd1')
    stdev = tf.layers.dense(stdev, num_components, name='stdev_hd2')
    # stdev = tf.nn.tanh(stdev, name='stdev_out')
    stdev = tf.maximum(tf.exp(stdev), 0.0001)

    mean = tf.layers.dense(x, 128, name='mean_hd1')
    mean = tf.layers.dense(mean, state_dim*num_components, name='mean_hd2')
    # mean = tf.nn.tanh(mean, name='mean_out')

    mix = tf.layers.dense(x, 128, name='mix_hd1')
    mix = tf.layers.dense(mix, num_components, name='mix_hd2')
    mix = tf.nn.softmax(mix, name='mix_out')

    # Normalization
    # max_pi = tf.reduce_max(mix, 1, keep_dims=True)
    # mix = tf.subtract(mix, max_pi)
    #
    # mix = tf.exp(mix)
    #
    # normalize_pi = tf.reciprocal(tf.reduce_sum(mix, 1, keep_dims=True))
    # mix = tf.multiply(normalize_pi, mix)
    #
    # stdev = tf.exp(stdev)
    return mix, stdev, mean


class PreCoEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)
        # Initialization

        self.latent_size = 128
        self.last_r = np.array([0.0]).flatten()
        self.buffer = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
        dropout_rate = 0.0
        lr = 0.00025
        print('General Stats: ')
        print('Drop Rate: ' + str(dropout_rate))
        print('Buffer Len: ' + str(self.buff_len))
        print('PreCo model:')
        print('Learning Rate: ' + str(lr))
        self.max_seq_len = 30
        self.batch_size = 32
        num_components = 2
        """ State Prediction """
        self.x = tf.placeholder(dtype=tf.float32, shape=([self.batch_size, self.state_dim]))
        self.a = tf.placeholder(dtype=tf.float32, shape=([self.batch_size, self.act_dim]))

        self.a_seq = tf.placeholder(dtype=tf.float32, shape=([self.batch_size, self.act_dim*self.max_seq_len]))
        self.y_seq = tf.placeholder(dtype=tf.float32, shape=([self.batch_size, self.state_dim*self.max_seq_len]))

        self.state_in = tf.placeholder(dtype=tf.float32, shape=([self.batch_size, self.latent_size]))

        self.loss_seq = 0
        self.loss_corr = 0
        self.loss_single = 0

        self.last_state = None

        with tf.variable_scope('PreCo', reuse=tf.AUTO_REUSE):

            tmp_a_seq = tf.split(self.a_seq, self.max_seq_len, 1)
            tmp_y_seq = tf.split(self.y_seq, self.max_seq_len, 1)

            corr_out, self.corr_hidden = corrector([self.x])
            self.corr_pi, self.corr_sigma, self.corr_mu = decode(self.corr_hidden, self.state_dim, num_components)
            l = self.MDN_loss(self.corr_pi, self.corr_sigma, self.corr_mu, self.x)
            self.loss_corr += l

            pred_out, self.pred_hidden = predictor([tmp_a_seq[0]], self.corr_hidden)
            pi, sigma, mu = decode(self.pred_hidden, self.state_dim, num_components)
            l = self.MDN_loss(pi, sigma, mu, tmp_y_seq[0])
            self.loss_single = l
            self.loss_seq += l

            # pred_out, self.pred_hidden_open = predictor([tmp_a_seq[0]], self.state_in)
            # self.decoded_pred_open = decode(self.pred_hidden, self.state_dim, num_components)

            last_state = self.pred_hidden
            for i in range(1, self.max_seq_len):
                a = tmp_a_seq[i]
                y = tmp_y_seq[i]
                pred_out, last_state = predictor([a], last_state)
                pi, sigma, mu = decode(last_state, self.state_dim, num_components)
                l = self.MDN_loss(pi, sigma, mu, y)
                self.loss_seq += l

            # Losses for the corrector, and single/sequence predictor, each one should be executed for each train step
            self.corr_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_corr)
            self.pred_seq_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_seq/self.max_seq_len)
            self.pred_single_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_single)

            # Testing
            _, self.corr_hidden_out = corrector([self.x])
            _, self.pred_hidden_out = predictor([self.a], self.corr_hidden_out)
            self.pi_decoded, self.sigma_decoded, self.mu_decoded = decode(self.pred_hidden_out, self.state_dim, num_components)

            _, self.pred_hidden_open = predictor([self.a], self.state_in)

            self.pi_decoded_open, self.sigma_decoded_open, self.mu_decoded_open = \
                decode(self.pred_hidden_open, self.state_dim, num_components)



    def pdf(self, y, mu, sigma):
        c = 1
        return (1/(((2*np.pi)**(c/2))*tf.pow(sigma, c))) * tf.exp( -(tf.pow(tf.norm(y-mu, 2, 1), 2) / tf.pow(2*sigma, 2)) )

    def loss(self, y, pi, sigma, mu):
        n_mix = pi.get_shape().as_list()[1]
        errors = tf.zeros(self.batch_size)
        for i in range(n_mix):
            errors += pi[:, i] * self.pdf(y, mu[:, i*self.state_dim:(i+1)*self.state_dim], sigma[:,i])
        error = -tf.log(tf.reduce_sum(errors))
        return error

    # Taken from https://github.com/dusenberrymw/mixture-density-networks/blob/master/mixture_density_networks.ipynb
    def gaussian_pdf(self, y, mu, sigmasq):
        # return (1/torch.sqrt(2*np.pi*sigmasq)) * torch.exp((-1/(2*sigmasq)) * torch.norm((x-mu), 2, 1)**2)
        return (1/tf.sqrt(2*np.pi*sigmasq)) * tf.exp((-1/2*sigmasq)) * (tf.norm((y-mu), 2, 1))**2

    def loss_fn(self, y, in_pi, in_sigma, in_mu):
        n_mix = in_pi.get_shape().as_list()[1]
        # pi = tf.split(in_pi, num_or_size_splits=n_mix, axis=1)
        # mu = tf.split(in_mu, num_or_size_splits=n_mix, axis=1)
        # sigma = tf.split(in_sigma, num_or_size_splits=n_mix, axis=1)
        pi = in_pi
        sigma = in_sigma
        mu = in_mu
        losses = tf.zeros(self.batch_size)

        for i in range(n_mix):
            likelihood_z_x = self.gaussian_pdf(y, mu[:, i*self.state_dim:(i+1)*self.state_dim], (sigma[:,i]))
            prior_z = pi[:, i]
            losses += prior_z * likelihood_z_x
        loss = tf.reduce_mean(-tf.log(losses))
        return loss

    def sample_pred(self, pi, sigma, mu, is_sq=False):

        # rather than sample the single conditional mode at each
        # point, we could sample many points from the GMM produced
        # by the model for each point, yielding a dense set of
        # predictions
        N, K = pi.shape
        _, KT = mu.shape
        T = int(KT / K)
        # out = Variable(torch.zeros(N, samples, T))  # s samples per example
        out = np.zeros((N,T))
        for i in range(N):
            # pi must sum to 1, thus we can sample from a uniform
            # distribution, then transform that to select the component
            u = np.random.uniform()  # sample from [0, 1)
            # split [0, 1] into k segments: [0, pi[0]), [pi[0], pi[1]), ..., [pi[K-1], pi[K])
            # then determine the segment `u` that falls into and sample from that component
            prob_sum = 0
            for k in range(K):
                prob_sum += pi[i, k]
                if u < prob_sum:
                    # sample from the kth component
                    for t in range(T):
                        if is_sq:
                            sample = np.random.normal(mu[i, k*T+t], np.sqrt(sigma[i, k]))
                        else:
                            sample = np.random.normal(mu[i, k*T+t], sigma[i, k])
                        out[i, t] = sample
                    break
        return out

    def MDN_loss(self, out_pi, out_sigma, out_mu, y):
        # return self.loss_fn(y, out_pi, out_sigma, out_mu)
        return self.loss(y, out_pi, out_sigma, out_mu)

    def MDN_out(self, out_pi, out_sigma, out_mu):
        return self.sample_pred(out_pi, out_sigma, out_mu, is_sq=False)

    def initialize(self, session, load=False):
        self.sess = session
        if not load:
            self.sess.run(tf.global_variables_initializer())

    def train_epoch(self, data):
        G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=self.batch_size)
        Single = 0.0
        Seq = 0.0
        Corr = 0.0
        MSE = 0.0
        for i in range(len(X)):
            pi, sigma, mu, single, seq, corr, _, _, _ = self.sess.run([ self.pi_decoded, self.sigma_decoded, self.mu_decoded, self.loss_single, self.loss_seq, self.loss_corr,
                                                        self.pred_seq_train_step, self.pred_single_train_step,
                                                        self.corr_train_step],
                                     feed_dict={
                                                self.x: X[i][:,:self.state_dim],
                                                self.a_seq: A[i],
                                                self.y_seq: S[i],
                                                self.a: A[i][:,:self.act_dim]
                                               })
            out = self.MDN_out(pi, sigma, mu)
            mse = np.linalg.norm(out-S[i][:,:self.state_dim])
            MSE += mse
            Single += single
            Seq += seq
            Corr += corr

        return Single / len(X), Seq / len(X), Corr / len(X), MSE / len(X)

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
                (single, seq, corr, MSE) = self.get_loss(valid)
                print('Epoch: ' + str(i) + '/' + str(total_steps))
                print('Valid Single: ' + str(single))
                print('Valid Single MSE: ' + str(MSE))
                print('Valid Seq: ' + str(seq))
                print('Valid Corr: ' + str(corr))
                print('')
                if saver is not None and save_str is not None:
                    save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
                    print("Model saved in path: %s" % save_path)
            start = time.time()
            (single, seq, corr, MSE) = self.train_epoch(train)
            duration = time.time() - start
            if stop_count > early_stopping and early_stopping > 0:
                break
            if i % log_interval != 0 and i > 0:
                print('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's')
                print('Train Single: ' + str(single))
                print('Train Single MSE: ' + str(MSE))
                print('Train Seq: ' + str(seq))
                print('Train Corr: ' + str(corr))
                print('')
        if valid is not None:
            (single, seq, corr, MSE) = self.get_loss(valid)
            print('Final Epoch')
            print('Valid Single: ' + str(single))
            print('Valid Single MSE: ' + str(MSE))
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
        MSE = 0.0
        for i in range(len(X)):
            pi, sigma, mu, single, seq, corr = self.sess.run([ self.pi_decoded, self.sigma_decoded, self.mu_decoded,
                                                           self.loss_single, self.loss_seq, self.loss_corr],
                                     feed_dict={
                                                self.x: X[i][:,:self.state_dim],
                                                self.a_seq: A[i],
                                                self.y_seq: S[i],
                                                self.a: A[i][:,:self.act_dim]
                                               })
            # diff = out-tmp_y_seq[0]
            out = self.MDN_out(pi, sigma, mu)
            MSE += np.linalg.norm(out-S[i][:,:self.state_dim])
            Single += single
            Seq += seq
            Corr += corr
        return Single / len(X), Seq / len(X), Corr / len(X), MSE / len(X)

    def reset(self, obs_in):
        obs = obs_in/self.state_mul_const
        obs = np.array([obs])
        obs_array = np.zeros((self.batch_size, self.state_dim))
        obs_array[0, :] = obs[0, :]
        self.last_state = self.sess.run([self.corr_hidden_out], feed_dict={self.x:obs_array})[0]
        return obs_in

    def step(self, action_in, obs_in=None, episode_step=None, save=True, buff=None, num=None):
        if obs_in is not None:
            action = np.array([action_in/self.act_mul_const])
            act_array = np.zeros((self.batch_size, self.act_dim))
            act_array[0,:] = action[0,:]
            obs = np.array([obs_in/self.state_mul_const])
            obs_array = np.zeros((self.batch_size, self.state_dim))
            obs_array[0, :] = obs[0, :]

            pi, sigma, mu = self.sess.run([self.pi_decoded, self.sigma_decoded, self.mu_decoded], feed_dict={self.x: obs_array, self.a: act_array})
            new_obs = self.MDN_out(pi, sigma, mu)
            new_obs = new_obs[0]*self.state_mul_const
            return new_obs
        else:
            action = np.array([action_in/self.act_mul_const])
            act_array = np.zeros((self.batch_size, self.act_dim))
            act_array[0,:] = action[0,:]

            pi, sigma, mu, self.last_state = self.sess.run([self.pi_decoded_open, self.sigma_decoded_open, self.mu_decoded_open, self.pred_hidden_open],
                                                     feed_dict={self.a: act_array,
                                                                self.state_in:self.last_state})
            new_obs = self.MDN_out(pi, sigma, mu)
            new_obs = new_obs[0]*self.state_mul_const
            return new_obs

    def uncertainty(self, action_in, obs_in=None):
        if obs_in is not None:
            action = np.array([action_in/self.act_mul_const])
            act_array = np.zeros((self.batch_size, self.act_dim))
            act_array[0,:] = action[0,:]
            obs = np.array([obs_in/self.state_mul_const])
            obs_array = np.zeros((self.batch_size, self.state_dim))
            obs_array[0, :] = obs[0, :]

            pi, sigma, mu = self.sess.run([self.pi_decoded, self.sigma_decoded, self.mu_decoded], feed_dict={self.x: obs_array, self.a: act_array})
        else:
            action = np.array([action_in/self.act_mul_const])
            act_array = np.zeros((self.batch_size, self.act_dim))
            act_array[0,:] = action[0,:]

            pi, sigma, mu, self.last_state = self.sess.run([self.pi_decoded_open, self.sigma_decoded_open, self.mu_decoded_open, self.pred_hidden_open],
                                                     feed_dict={self.a: act_array,
                                                                self.state_in:self.last_state})
        return np.mean(sigma)

    def next_move(self, obs_in, episode_step):
        max_act = None
        max_score = 0
        num_tries = 100

        for i in range(num_tries):
            act = np.random.uniform(-1, 1, self.act_dim)
            score = self.uncertainty(act, obs_in)
            if score > max_score:
                max_act = act
                max_score = score

        return max_act
