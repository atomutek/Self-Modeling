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

def encode_mean(x, out_dim, drop_rate=0.5):
    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 256)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 128)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, out_dim)
    return x

def encode_var(x, out_dim, drop_rate=0.5):
    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 256)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 128)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.leaky_relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, out_dim)
    return x

def decode(x, out_dim, drop_rate=0.5):
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

    return tf.nn.tanh(x, name='th0')

def KLD_loss(latent_var, latent_mean):
    return -0.5 * tf.reduce_mean(1.0 + latent_var - tf.pow(latent_mean, 2) - tf.exp(latent_var))


class PreCoGenEnvLearner(EnvLearner):
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
        print('PreCo Gen model:')
        print('Learning Rate: ' + str(lr))
        self.max_seq_len = 300

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

        with tf.variable_scope('PreCoVAE', reuse=tf.AUTO_REUSE):

            tmp_a_seq = tf.split(self.a_seq, self.max_seq_len, 1)
            tmp_y_seq = tf.split(self.y_seq, self.max_seq_len, 1)


            corr_out, self.corr_hidden = corrector([self.x])
            self.decoded_corr = decode(self.corr_hidden, self.state_dim)
            self.loss_corr += losses.loss_p(self.decoded_corr, self.x)

            pred_out, self.pred_hidden = predictor([tmp_a_seq[0]], self.corr_hidden)
            self.decoded_pred = decode(self.pred_hidden, self.state_dim)
            self.loss_single += losses.loss_p(self.decoded_pred, tmp_y_seq[0])
            self.loss_seq += self.loss_single

            pred_out, self.pred_hidden_open = predictor([tmp_a_seq[0]], self.state_in)
            self.decoded_pred_open = decode(self.pred_hidden, self.state_dim)

            last_state = self.pred_hidden
            for i in range(1, self.max_seq_len):
                a = tmp_a_seq[i]
                y = tmp_y_seq[i]
                pred_out, last_state = predictor([a], last_state)
                decoded_pred = decode(last_state, self.state_dim)
                l = losses.loss_p(decoded_pred, y)
                self.loss_seq += l

            # Losses for the corrector, and single/sequence predictor, each one should be executed for each train step
            self.corr_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_corr)
            self.pred_seq_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_seq)
            self.pred_single_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_single)

            # Testing
            _, self.corr_hidden_out = corrector([self.x])
            _, self.pred_hidden_out = predictor([self.a], self.corr_hidden_out)
            self.decoded = decode(self.pred_hidden_out, self.state_dim)

            _, self.pred_hidden_open = predictor([self.a], self.state_in)
            self.decoded_pred_open = decode(self.pred_hidden_open, self.state_dim)

    def __vae_loss__(self, out, y):
        train_latent_mean = encode_mean(out[0], self.latent_size)
        train_latent_var = encode_var(out[0], self.latent_size)
        train_latent = tf.random_uniform(shape=([self.latent_size]))
        train_latent = train_latent*tf.exp(0.5*train_latent_var)+train_latent_mean
        decoded_train = decode(train_latent, self.state_dim)
        l = losses.loss_p(decoded_train, y) + KLD_loss(train_latent_var, train_latent_mean)
        return l

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
            single, seq, corr, corr_out, pred_out, _, _, _ = self.sess.run([self.loss_single, self.loss_seq, self.loss_corr, self.decoded_corr, self.decoded_pred,
                                                        self.pred_seq_train_step, self.pred_single_train_step, self.corr_train_step],
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
            single, seq, corr, out, tmp_y_seq = self.sess.run([self.loss_single, self.loss_seq, self.loss_corr, self.decoded_pred, tf.split(self.y_seq, self.max_seq_len, 1)],
                                     feed_dict={
                                                self.x: X[i][:,:self.state_dim],
                                                self.a_seq: A[i],
                                                self.y_seq: S[i]
                                               })
            diff = out-tmp_y_seq[0]
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
        # print('Stepping!')
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
            new_obs = new_obs[0][0]*self.state_mul_const
            return new_obs
        else:
            action = np.array([action_in/self.act_mul_const])
            new_obs, self.last_state = self.sess.run([self.decoded_pred_open, self.pred_hidden_open],
                                                     feed_dict={self.a: action,
                                                                self.state_in:self.last_state})
            new_obs = new_obs[0]*self.state_mul_const
            return new_obs

