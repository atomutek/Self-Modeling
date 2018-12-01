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

def decode(x, out_dim, num_components, drop_rate=0.5):

    i = 0
    # x = tf.layers.batch_normalization(x, name='bn3')
    # x = tf.layers.dense(x, 256, name='mlp3')
    # x = tf.layers.dropout(x, rate=drop_rate, name='do3')
    # x = tf.nn.relu(x, name='rl3')
    #
    # x = tf.layers.batch_normalization(x, name='bn4')
    # x = tf.layers.dense(x, 512, name='mlp4')
    # x = tf.layers.dropout(x, rate=drop_rate, name='do4')
    # x = tf.nn.relu(x, name='rl4')
    out = tf.layers.batch_normalization(x, name='bn0_'+str(i))
    out = tf.layers.dense(out, 512, name='mlp0_'+str(i))
    out = tf.layers.dropout(out, rate=drop_rate, name='do0_'+str(i))
    out = tf.nn.relu(out, name='rl0_'+str(i))

    # out = tf.layers.batch_normalization(x, name='bn0c_'+str(i))
    # out = tf.layers.dense(out, 512, name='mlp0c_'+str(i))
    # out = tf.layers.dropout(out, rate=drop_rate, name='do0c_'+str(i))
    # out = tf.nn.relu(out, name='rl0c_'+str(i))

    out = tf.layers.batch_normalization(out, name='bn1_'+str(i))
    out = tf.layers.dense(out, 256, name='mlp1_'+str(i))
    out = tf.layers.dropout(out, rate=drop_rate, name='do1_'+str(i))
    out = tf.nn.relu(out, name='rl1_'+str(i))

    # out = tf.layers.batch_normalization(out, name='bn1c_'+str(i))
    # out = tf.layers.dense(out, 256, name='mlp1c_'+str(i))
    # out = tf.layers.dropout(out, rate=drop_rate, name='do1c_'+str(i))
    # out = tf.nn.relu(out, name='rl1c_'+str(i))

    outputs = []
    for i in range(num_components):

        out = tf.layers.batch_normalization(out, name='bn2_'+str(i))
        out = tf.layers.dense(out, 128, name='mlp2_'+str(i))
        out = tf.layers.dropout(out, rate=drop_rate, name='do2_'+str(i))
        out = tf.nn.relu(out, name='rl2_'+str(i))

        # out = tf.layers.batch_normalization(out, name='bn2c_'+str(i))
        # out = tf.layers.dense(out, 128, name='mlp2c_'+str(i))
        # out = tf.layers.dropout(out, rate=drop_rate, name='do2c_'+str(i))
        # out = tf.nn.relu(out, name='rl2c_'+str(i))

        out = tf.layers.batch_normalization(out, name='bn_out_'+str(i))
        out = tf.layers.dense(out, out_dim, name='mlp_out_'+str(i))

        out = tf.nn.tanh(out, name='th_out')
        outputs.append(out)
    outputs = tf.stack(outputs)
    return outputs

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
        self.max_seq_len = 30
        self.batch_size = 32

        self.rand_on_start = True
        self.num_components = 8

        if self.rand_on_start:
            self.chosen = np.random.random_integers(0, self.num_components-1, 1)[0]
        else:
            self.chosen = 0


        """ State Prediction """
        self.x = tf.placeholder(dtype=tf.float32, shape=([None, self.state_dim]))
        self.a = tf.placeholder(dtype=tf.float32, shape=([None, self.act_dim]))
        self.y = tf.placeholder(dtype=tf.float32, shape=([None, self.state_dim]))

        self.a_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.act_dim*self.max_seq_len]))
        self.y_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.state_dim*self.max_seq_len]))

        self.state_in = tf.placeholder(dtype=tf.float32, shape=([None, self.latent_size]))

        self.loss_seq = tf.zeros(shape=(self.num_components, self.state_dim))
        self.loss_corr = tf.zeros(shape=(self.num_components, self.state_dim))
        self.loss_single = tf.zeros(shape=(self.num_components, self.state_dim))

        self.last_state = None

        with tf.variable_scope('PreCoGen', reuse=tf.AUTO_REUSE):

            tmp_a_seq = tf.split(self.a_seq, self.max_seq_len, 1)
            tmp_y_seq = tf.split(self.y_seq, self.max_seq_len, 1)


            corr_out, self.corr_hidden = corrector([self.x])
            self.decoded_corrs = decode(self.corr_hidden, self.state_dim, self.num_components)
            self.loss_corr += self.loss(self.decoded_corrs, self.x)

            pred_out, self.pred_hidden = predictor([tmp_a_seq[0]], self.corr_hidden)
            self.decoded_preds = decode(self.pred_hidden, self.state_dim, self.num_components)
            self.loss_single += self.loss(self.decoded_preds, tmp_y_seq[0])
            self.loss_seq += self.loss_single

            last_state = self.pred_hidden

            self.units = 0.0
            tmp_loss_seq = 0.0
            l = 0.0
            tmp_last_loss = 0.0
            last_loss = 0.0
            for i in range(1, self.max_seq_len):
                a = tmp_a_seq[i]
                y = tmp_y_seq[i]
                pred_out, last_state = predictor([a], last_state)
                decoded_preds = decode(last_state, self.state_dim, self.num_components)
                l = self.loss(decoded_preds, y, avg=False)

                # This term is a hack such that when the action is all 0s (i.e. for an invalid action) it isn't counted
                is_on = tf.minimum(tf.ceil(tf.reduce_mean(a, 1)), 1.0)
                is_on = tf.stack([tf.stack([is_on]*self.state_dim, axis=1)]*self.num_components)
                l = l * is_on

                # TODO Double check
                # Only add if the action isn't null
                last_loss += l
                if i > 1:
                    # Only remove the previous one if this one isn't null
                    last_loss -= tmp_last_loss*is_on
                tmp_last_loss = l

                self.units = self.units + is_on
                tmp_loss_seq += l

            # summed seq loss
            # self.loss_seq += tf.reduce_mean(tmp_loss_seq, 1)

            # avg seq loss
            tmp_loss_seq = tmp_loss_seq/tf.maximum(self.units, 1.0)
            self.loss_seq += tf.reduce_mean(tmp_loss_seq, 1)

            # Last element seq loss
            # self.loss_seq = tf.reduce_mean(last_loss, 1)

            # Losses for the corrector, and single/sequence predictor, each one should be executed for each train step
            self.corr_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_corr)
            self.pred_seq_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_seq)
            self.pred_single_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_single)


            corr_lambda = 0.1
            seq_lambda = 10

            cumulative_loss = self.loss_single + corr_lambda*self.loss_corr + seq_lambda*self.loss_seq
            # self.corr_train_step = tf.zeros(1)
            # self.pred_single_train_step = tf.zeros(1)
            # self.cumulative_train_step = tf.train.AdamOptimizer(lr).minimize(cumulative_loss)

            # Testing
            _, self.corr_hidden_out = corrector([self.x])
            self.autoencoded = decode(self.corr_hidden_out, self.state_dim, self.num_components)

            _, self.pred_hidden_out = predictor([self.a], self.corr_hidden_out)
            self.decodeds = decode(self.pred_hidden_out, self.state_dim, self.num_components)

            _, self.pred_hidden_open = predictor([self.a], self.state_in)
            self.decoded_preds_open = decode(self.pred_hidden_open, self.state_dim, self.num_components)

            # simple loss
            self.simple_loss_corr = self.loss(self.autoencoded, self.x)
            self.simple_loss_single = self.loss(self.decodeds, self.y)
            self.simple_loss_seq = self.loss(self.decoded_preds_open, self.y)

            self.simple_corr_train_step = tf.train.AdamOptimizer(lr).minimize(self.simple_loss_corr)
            self.simple_pred_seq_train_step = tf.train.AdamOptimizer(lr).minimize(self.simple_loss_seq)
            self.simple_pred_single_train_step = tf.train.AdamOptimizer(lr).minimize(self.simple_loss_single)

    def loss(self, outs, y, avg=True):
        loss = tf.abs(y - outs)
        # loss = tf.reduce_mean(loss, 2)
        if avg:
            loss = tf.reduce_mean(loss, 1)
        return loss

    def initialize(self, session, load=False):
        self.sess = session
        if not load:
            self.sess.run(tf.global_variables_initializer())

    def train_epoch(self, data):
        # X = []
        # A = []
        # Y = []
        # S = []
        # D = []
        # reset=True
        # for tuple in data:
        #     x = np.array([tuple[0]/self.state_mul_const])
        #     a = np.array([tuple[1]/self.act_mul_const])
        #     y = np.array([tuple[2]/self.state_mul_const])
        #     X.append(x)
        #     A.append(a)
        #     Y.append(y)
        #     D.append(tuple[4])
        # import sys
        # start = time.time()
        # Single = 0.0
        # Seq = 0.0
        # Corr = 0.0
        # reset = True
        # for i in range(len(X)):
        #     if reset: state_tmp = self.sess.run([self.corr_hidden_out], feed_dict={self.x:X[i]})[0]
        #     if D[i]: reset = True
        #
        #     single, seq, corr, new_state, _, _, _ = self.sess.run([self.simple_loss_single, self.simple_loss_seq, self.simple_loss_corr,
        #                                                                     self.pred_hidden_open,
        #                                                                     self.simple_pred_seq_train_step, self.simple_pred_single_train_step, self.simple_corr_train_step],
        #                              feed_dict={
        #                                         self.x: X[i],
        #                                         self.a: A[i],
        #                                         self.y: Y[i],
        #                                         self.state_in: state_tmp
        #                                        })
        #     state_tmp = new_state
        #     Single += single
        #     Seq += seq
        #     Corr += corr
        #     # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
        #     sys.stdout.write(str(round(float(100*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        # # sys.stdout.write('Done\r\n')
        # return Single / len(X), Seq / len(X), Corr / len(X)

        G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=self.batch_size)
        Single = 0.0
        Seq = 0.0
        Corr = 0.0
        import sys
        start = time.time()
        for i in range(min(len(X), len(A), len(S))):
            single, seq, corr, corr_out, pred_out, _, _, _ = self.sess.run([self.loss_single, self.loss_seq, self.loss_corr, self.decoded_corrs,self.decoded_preds,
                                                        self.pred_seq_train_step, self.pred_single_train_step, self.corr_train_step],
                                     feed_dict={
                                                self.x: X[i][:,:self.state_dim],
                                                self.a_seq: A[i],
                                                self.y_seq: S[i]
                                               })
            Single += np.mean(single, 1)
            Seq += np.mean(seq, 1)
            Corr += np.mean(corr, 1)
            # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
            sys.stdout.write(str(round(float(100*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        # sys.stdout.write('Done\r\n')
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
            if i % log_interval != 0 or i == 0:
                print('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's                                          ')
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
        # X = []
        # A = []
        # Y = []
        # S = []
        # D = []
        # reset=True
        # for tuple in data:
        #     x = np.array([tuple[0]/self.state_mul_const])
        #     a = np.array([tuple[1]/self.act_mul_const])
        #     y = np.array([tuple[2]/self.state_mul_const])
        #     X.append(x)
        #     A.append(a)
        #     Y.append(y)
        #     D.append(tuple[4])
        #
        # Single = 0.0
        # Seq = 0.0
        # Corr = 0.0
        # reset = True
        # # self.autoencoded = decode(self.corr_hidden_out, self.state_dim, self.num_components)
        # #
        # # _, self.pred_hidden_out = predictor([self.a], self.corr_hidden_out)
        # # self.decodeds = decode(self.pred_hidden_out, self.state_dim, self.num_components)
        # #
        # # _, self.pred_hidden_open = predictor([self.a], self.state_in)
        # # self.decoded_preds_open = decode(self.pred_hidden_open, self.state_dim, self.num_components)
        #
        #
        # for i in range(len(X)):
        #     if reset: state_tmp = self.sess.run([self.corr_hidden_out], feed_dict={self.x:X[i]})[0]
        #     if D[i]: reset = True
        #
        #     single, seq, corr, new_state, corr_out, pred_out, pred_open = self.sess.run([self.simple_loss_single, self.simple_loss_seq, self.simple_loss_corr,
        #                                                                     self.pred_hidden_open, self.autoencoded, self.decodeds, self.decoded_preds_open],
        #                              feed_dict={
        #                                         self.x: X[i],
        #                                         self.a: A[i],
        #                                         self.y: Y[i],
        #                                         self.state_in: state_tmp
        #                                        })
        #     corr_calc_loss = np.abs(corr_out-X[i])
        #     corr_calc_loss = np.mean(corr_calc_loss, 2)
        #     corr_calc_loss = np.mean(corr_calc_loss, 1)
        #
        #     single_calc_loss = np.abs(pred_out-Y[i])
        #     single_calc_loss = np.mean(single_calc_loss, 2)
        #     single_calc_loss = np.mean(single_calc_loss, 1)
        #
        #     seq_calc_loss = np.abs(pred_open-Y[i])
        #     seq_calc_loss = np.mean(seq_calc_loss, 2)
        #     seq_calc_loss = np.mean(seq_calc_loss, 1)
        #
        #     state_tmp = new_state
        #     Single += single
        #     Seq += seq
        #     Corr += corr
        # return Single / len(X), Seq / len(X), Corr / len(X)

        # G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=self.batch_size)
        # Single = 0.0
        # Seq = 0.0
        # Corr = 0.0
        # for i in range(len(X)):
        #     single, seq, corr, out = self.sess.run([self.loss_single, self.loss_seq, self.loss_corr, self.decoded_preds],
        #                              feed_dict={
        #                                         self.x: X[i][:,:self.state_dim],
        #                                         self.a_seq: A[i],
        #                                         self.y_seq: S[i]
        #                                        })
        #     # diff = out-tmp_y_seq[0]
        #     Single += single
        #     Seq += seq
        #     Corr += corr
        # return Single / len(X), Seq / len(X), Corr / len(X)



        G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=self.batch_size)
        Single = 0.0
        Seq = 0.0
        Corr = 0.0
        import sys
        start = time.time()
        for i in range(min(len(X), len(A), len(S))):
            single, seq, corr, corr_out, pred_out = self.sess.run([self.loss_single, self.loss_seq, self.loss_corr,
                                                                            self.decoded_corrs,self.decoded_preds],
                                     feed_dict={
                                                self.x: X[i][:,:self.state_dim],
                                                self.a_seq: A[i],
                                                self.y_seq: S[i]
                                               })
            Single += single
            Seq += seq
            Corr += corr
            # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
            # sys.stdout.write(str(round(float(100*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        # sys.stdout.write('Done\r\n')
        return Single / len(X), Seq / len(X), Corr / len(X)

    def reset(self, obs_in):
        obs = obs_in/self.state_mul_const
        obs = np.array([obs])
        self.last_state = self.sess.run([self.corr_hidden_out], feed_dict={self.x:obs})[0]
        if self.rand_on_start:
            self.chosen = np.random.random_integers(0, self.num_components-1, 1)[0]
        print(str(self.chosen)+' chosen as num')
        return obs_in

    def step(self, action_in, obs_in=None, episode_step=None, save=True, buff=None, num=None):
        # print('Stepping!')
        if num is None:
            num = self.chosen
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
            new_obs = self.sess.run([self.decodeds], feed_dict={self.x: obs, self.a: action})
            new_obs = new_obs[0]*self.state_mul_const
        else:
            action = np.array([action_in/self.act_mul_const])
            new_obs, tmp_state = self.sess.run([self.decoded_preds_open, self.pred_hidden_open],
                                                     feed_dict={self.a: action,
                                                                self.state_in:self.last_state})
            self.last_state = tmp_state
            new_obs = new_obs*self.state_mul_const
        new_obs = new_obs[:,0]
        if num > 0:
            return new_obs[num]
        return new_obs


    def uncertainty(self, action_in, obs_in=None):
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
            new_obs = self.sess.run([self.decodeds], feed_dict={self.x: obs, self.a: action})
            new_obs = new_obs[0]*self.state_mul_const
        else:
            action = np.array([action_in/self.act_mul_const])
            new_obs, self.last_state = self.sess.run([self.decoded_preds_open, self.pred_hidden_open],
                                                     feed_dict={self.a: action,
                                                                self.state_in:self.last_state})
            new_obs = new_obs[0]*self.state_mul_const

        diffs = np.zeros(shape=(self.num_components, self.num_components))
        row_means = np.zeros(self.num_components)
        for i in range(self.num_components):
            for j in range(i, self.num_components):
                diffs[i][j] = np.linalg.norm(new_obs[i]-new_obs[j])
            row_means[i] = np.mean(diffs[i])

        return np.mean(row_means)

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