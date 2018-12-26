import time
from collections import deque

import numpy as np
import tensorflow as tf

from env_learners.env_learner import EnvLearner
from misc import losses


def predictor(x, h, latent_size, out_dim, num_components):
    number_of_layers = 2
    out = []

    # rnn = tf.contrib.rnn.LSTMCell(latent_size, name='predictor', state_is_tuple=False)
    # rnn = tf.contrib.rnn.GRUCell(latent_size, name='predictor')
    rnn = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(latent_size, name='predictor'+str(i)) for i in range(number_of_layers)], state_is_tuple=False)
    outputs, state = tf.nn.static_rnn(rnn, x, initial_state=h, dtype=tf.float32)

    for output in outputs:
        out.append(decode(output, out_dim, num_components))
    return out, state

def corrector(x, latent_size, out_dim, num_components, h=None):
    number_of_layers = 2
    out = []
    # rnn = tf.contrib.rnn.LSTMCell(latent_size, name='corrector', state_is_tuple=False)
    # rnn = tf.contrib.rnn.GRUCell(latent_size, name='corrector')
    rnn = tf.nn.rnn_cell.MultiRNNCell([tf.contrib.rnn.GRUCell(latent_size, name='corrector'+str(i)) for i in range(number_of_layers)], state_is_tuple=False)

    outputs, state = tf.nn.static_rnn(rnn, x, initial_state=h, dtype=tf.float32)

    for output in outputs:
        out.append(decode(output, out_dim, num_components))
    return out, state

def decode(x, out_dim, num_components, drop_rate=0.5):

    i = 0

    out = x

    outputs = []

    for i in range(num_components):

        out = tf.layers.batch_normalization(out, name='bn0_'+str(i))
        out = tf.layers.dense(out, 1024, name='mlp0_'+str(i))
        out = tf.nn.relu(out, name='rl0_'+str(i))
        out = tf.layers.dropout(out, rate=drop_rate, name='do0_'+str(i))

        out = tf.layers.batch_normalization(out, name='bn1_'+str(i))
        out = tf.layers.dense(out, 512, name='mlp1_'+str(i))
        out = tf.nn.relu(out, name='rl1_'+str(i))
        out = tf.layers.dropout(out, rate=drop_rate, name='do1_'+str(i))

        out = tf.layers.batch_normalization(out, name='bn2_'+str(i))
        out = tf.layers.dense(out, 256, name='mlp2_'+str(i))
        out = tf.nn.relu(out, name='rl2_'+str(i))
        out = tf.layers.dropout(out, rate=drop_rate, name='do2_'+str(i))

        out = tf.layers.batch_normalization(out, name='bns_0_'+str(i))
        out = tf.layers.dense(out, 128, name='mlps_0_'+str(i))
        out = tf.nn.relu(out, name='rls_0_'+str(i))
        out = tf.layers.dropout(out, rate=drop_rate, name='dos_0_'+str(i))

        out = tf.layers.batch_normalization(out, name='bn_out_'+str(i))
        out = tf.layers.dense(out, out_dim, name='mlp_out_'+str(i))

        # out = tf.nn.tanh(out, name='th_out')
        outputs.append(out)
    outputs = tf.stack(outputs)
    return outputs

def KLD_loss(latent_var, latent_mean):
    return -0.5 * tf.reduce_mean(1.0 + latent_var - tf.pow(latent_mean, 2) - tf.exp(latent_var))


## TODO ##
# Implement attention encoder decoder as separate class and use that
# Have a seperate loss value for the difference in components and minimize that loss as well as a secondary loss term
# Try training it so that it just learns the sequential loss (thats the most important one)
# Implement a priority replay like system in the active learning

class PreCoGenEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)
        # Initialization

        self.latent_size = 2048
        self.last_r = np.array([0.0]).flatten()
        self.buffer = deque(self.buff_init * self.buff_len, maxlen=self.buff_len)
        dropout_rate = 0.5
        lr = 1e-4
        print('General Stats: ')
        print('Drop Rate: ' + str(dropout_rate))
        print('Buffer Len: ' + str(self.buff_len))
        print('PreCo Gen model:')
        print('Learning Rate: ' + str(lr))
        self.max_seq_len = 100
        self.batch_size = 256

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


        self.loss_seq = tf.zeros(shape=(self.num_components, self.state_dim))
        self.loss_corr = tf.zeros(shape=(self.num_components, self.state_dim))
        self.loss_single = tf.zeros(shape=(self.num_components, self.state_dim))

        self.last_state = None

        with tf.variable_scope('PreCoGen', reuse=tf.AUTO_REUSE):

            tmp_a_seq = tf.split(self.a_seq, self.max_seq_len, 1)
            tmp_y_seq = tf.split(self.y_seq, self.max_seq_len, 1)


            self.autoencoded, self.corr_hidden_out = corrector([self.x], self.latent_size, self.state_dim, self.num_components)
            self.autoencoded = self.autoencoded[0]
            self.loss_corr = self.loss(self.autoencoded, self.x)

            self.decodeds, self.pred_hidden_out = predictor([self.a], self.corr_hidden_out, self.latent_size, self.state_dim, self.num_components)
            self.decodeds = self.decodeds[0]
            self.loss_single = self.loss(self.decodeds, self.y)


            self.decoded_preds, self.pred_hidden = predictor(tmp_a_seq, self.corr_hidden_out, self.latent_size, self.state_dim, self.num_components)
            self.loss_seq = 0.0

            self.units = 0
            for i in range(1, self.max_seq_len):
                y = tmp_y_seq[i]
                decoded_preds = self.decoded_preds[i]
                l = self.loss(decoded_preds, y, avg=True)
                # This term is a hack such that when the action is all 0s (i.e. for an invalid action) it isn't counted
                # is_on = tf.minimum(tf.ceil(tf.reduce_mean(a, 1)), 1.0)
                # is_on = tf.stack([tf.stack([is_on]*self.state_dim, axis=1)]*self.num_components)
                # self.loss_seq += l * is_on
                # self.units += is_on
                self.loss_seq += l
            self.loss_seq /= self.max_seq_len-1
            # self.loss_seq = tf.reduce_mean(self.loss_seq/tf.maximum(self.units, 1.0))

            # avg seq loss
            # tmp_loss_seq = tmp_loss_seq/tf.maximum(self.units, 1.0)
            # self.loss_seq += tf.reduce_mean(tmp_loss_seq, 1)

            # Last element seq loss
            # self.loss_seq = tf.reduce_mean(last_loss, 1)

            # Losses for the corrector, and single/sequence predictor, each one should be executed for each train step
            self.corr_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_corr)
            self.pred_seq_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_seq)
            self.pred_single_train_step = tf.train.AdamOptimizer(lr).minimize(self.loss_single)

            self.optimizer = tf.train.AdamOptimizer(lr)

            self.seq_grads = self.optimizer.compute_gradients(self.loss_seq)
            self.single_grads = self.optimizer.compute_gradients(self.loss_single)
            self.corr_grads = self.optimizer.compute_gradients(self.loss_corr)[:-4]

            self.seq_apply = self.optimizer.apply_gradients(self.seq_grads)
            self.single_apply = self.optimizer.apply_gradients(self.single_grads)
            self.corr_apply = self.optimizer.apply_gradients(self.corr_grads)

            corr_lambda = 0.1
            seq_lambda = 10

            # cumulative_loss = self.loss_single + corr_lambda*self.loss_corr + seq_lambda*self.loss_seq
            # self.corr_train_step = tf.zeros(1)
            # self.pred_single_train_step = tf.zeros(1)
            # self.cumulative_train_step = tf.train.AdamOptimizer(lr).minimize(cumulative_loss)

            # Testing
            # self.autoencoded, self.corr_hidden_out = corrector([self.x], self.latent_size, self.state_dim, self.num_components)
            # self.autoencoded = self.autoencoded[0]

            # self.decodeds, self.pred_hidden_out = predictor([self.a], self.corr_hidden, self.latent_size, self.state_dim, self.num_components)
            # self.decodeds = self.decodeds[0]

            self.state_in = tf.placeholder(dtype=tf.float32, shape=self.corr_hidden_out.shape)
            self.decoded_preds_open, self.pred_hidden_open = predictor([self.a], self.state_in, self.latent_size, self.state_dim, self.num_components)
            self.decoded_preds_open = self.decoded_preds_open[0]
            #
            # # simple loss
            # self.simple_loss_corr = self.loss(self.autoencoded, self.x)
            # self.simple_loss_single = self.loss(self.decodeds, self.y)
            # self.simple_loss_seq = self.loss(self.decoded_preds_open, self.y)
            #
            # self.simple_corr_train_step = tf.train.AdamOptimizer(lr).minimize(self.simple_loss_corr)
            # self.simple_pred_seq_train_step = tf.train.AdamOptimizer(lr).minimize(self.simple_loss_seq)
            # self.simple_pred_single_train_step = tf.train.AdamOptimizer(lr).minimize(self.simple_loss_single)

    def loss(self, outs, y, avg=True):
        # loss = tf.pow(tf.abs(y - outs), 2)
        loss = tf.abs(y - outs)
        loss = tf.reduce_mean(loss, 2)
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
        #     y = np.array([tuple[3]/self.state_mul_const])
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
        #     if reset:
        #         state_tmp = self.sess.run([self.corr_hidden_out], feed_dict={self.x:X[i]})[0]
        #         reset = False
        #     if D[i]: reset = True
        #
        #     single, seq, corr, new_state, corr_state, _, _, _ = self.sess.run([self.simple_loss_single, self.simple_loss_seq, self.simple_loss_corr,
        #                                                                     self.pred_hidden_open, self.corr_hidden_out,
        #                                                                     self.simple_pred_seq_train_step, self.simple_pred_single_train_step, self.simple_corr_train_step],
        #                              feed_dict={
        #                                         self.x: X[i],
        #                                         self.a: A[i],
        #                                         self.y: Y[i],
        #                                         self.state_in: state_tmp
        #                                        })
        #     state_tmp = new_state
        #     Single += np.mean(single)
        #     Seq += np.mean(seq)
        #     Corr += np.mean(corr)
        #     # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
        #     sys.stdout.write(str(round(float(100*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        # # sys.stdout.write('Done\r\n')
        # return Single / len(X), Seq / len(X), Corr / len(X)


        # G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=self.batch_size)
        X = []
        A = []
        Y = []

        Xs = []
        As = []
        Ys = []
        reset=True
        a = deque(maxlen=self.max_seq_len)
        y = deque(maxlen=self.max_seq_len)
        x = deque(maxlen=self.max_seq_len)
        for i in range(len(data)):
            obs = data[i][0]/self.state_mul_const
            act = data[i][1]/self.act_mul_const
            new_obs = data[i][3]/self.state_mul_const

            if reset:
                x = deque(maxlen=self.max_seq_len)
                a = deque(maxlen=self.max_seq_len)
                y = deque(maxlen=self.max_seq_len)
                a.append(act)
                y.append(new_obs)
                x.append(obs)
            else:
                # x = copy.copy(X[i-1])
                # x.append(obs)
                a.append(act)
                y.append(new_obs)
                x.append(obs)
                assert len(x) == len(y) == len(a)
                if len(a) == self.max_seq_len:
                    As.append(np.concatenate(a))
                    Ys.append(np.concatenate(y))
                    Xs.append(x[0])

            reset = data[i][4]
            X.append(obs)
            Y.append(new_obs)
            A.append(act)

        assert len(X) == len(Y) == len(A)
        assert len(Xs) == len(Ys) == len(As)

        p = np.random.permutation(len(X))
        X = self.__batch__(np.array(X)[p], self.batch_size)
        Y = self.__batch__(np.array(Y)[p], self.batch_size)
        A = self.__batch__(np.array(A)[p], self.batch_size)

        ps = np.random.permutation(len(Xs))
        Xs = self.__batch__(np.array(Xs)[ps], self.batch_size)
        Ys = self.__batch__(np.array(Ys)[ps], self.batch_size)
        As = self.__batch__(np.array(As)[ps], self.batch_size)


        Single = 0.0
        Seq = 0.0
        Corr = 0.0
        import sys
        start = time.time()
        # for i in range(len(X)):
        #     single, corr, corr_out, pred_out, _, _= self.sess.run([self.loss_single, self.loss_corr, self.autoencoded,self.decodeds,
        #                                                 self.pred_single_train_step, self.corr_train_step],
        #                              feed_dict={
        #                                         self.x: X[i],
        #                                         self.a: A[i],
        #                                         self.y: Y[i]
        #                                        })
        #     Single += np.mean(single)
        #     Corr += np.mean(corr)
        #     # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
        #     sys.stdout.write(str(round(float(50*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')

        for i in range(len(Xs)):
            seq, pred_open, _ = self.sess.run([self.loss_seq, self.decoded_preds,
                                                        self.pred_seq_train_step,],
                                     feed_dict={
                                                self.x: Xs[i],
                                                self.a_seq: As[i],
                                                self.y_seq: Ys[i]
                                               })
            Seq += np.mean(seq)
            # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
            sys.stdout.write(str(round(float(i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
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

        (single, seq, corr) = self.get_loss(valid)
        print('Valid Single: ' + str(single))
        print('Valid Seq: ' + str(seq))
        print('Valid Corr: ' + str(corr))
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
        #     y = np.array([tuple[3]/self.state_mul_const])
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
        #     if reset:
        #         state_tmp = self.sess.run([self.corr_hidden_out], feed_dict={self.x:X[i]})[0]
        #         reset = False
        #     if D[i]: reset = True
        #
        #     single, seq, corr, new_state, corr_state= self.sess.run([self.simple_loss_single, self.simple_loss_seq, self.simple_loss_corr,
        #                                                                     self.pred_hidden_open, self.corr_hidden_out],
        #                              feed_dict={
        #                                         self.x: X[i],
        #                                         self.a: A[i],
        #                                         self.y: Y[i],
        #                                         self.state_in: state_tmp
        #                                        })
        #     state_tmp = new_state
        #     Single += np.mean(single)
        #     Seq += np.mean(seq)
        #     Corr += np.mean(corr)
        #     # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
        #     sys.stdout.write(str(round(float(100*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        # # sys.stdout.write('Done\r\n')
        # return Single / len(X), Seq / len(X), Corr / len(X)


        X = []
        A = []
        Y = []

        Xs = []
        As = []
        Ys = []
        reset=True
        a = deque(maxlen=self.max_seq_len)
        y = deque(maxlen=self.max_seq_len)
        x = deque(maxlen=self.max_seq_len)
        for i in range(len(data)):
            obs = data[i][0]/self.state_mul_const
            act = data[i][1]/self.act_mul_const
            new_obs = data[i][3]/self.state_mul_const

            if reset:
                x = deque(maxlen=self.max_seq_len)
                a = deque(maxlen=self.max_seq_len)
                y = deque(maxlen=self.max_seq_len)
                a.append(act)
                y.append(new_obs)
                x.append(obs)
            else:
                # x = copy.copy(X[i-1])
                # x.append(obs)
                a.append(act)
                y.append(new_obs)
                x.append(obs)
                assert len(x) == len(y) == len(a)
                if len(a) == self.max_seq_len:
                    As.append(np.concatenate(a))
                    Ys.append(np.concatenate(y))
                    Xs.append(x[0])

            reset = data[i][4]
            X.append(obs)
            Y.append(new_obs)
            A.append(act)

        assert len(X) == len(Y) == len(A)
        assert len(Xs) == len(Ys) == len(As)

        p = np.random.permutation(len(X))
        X = self.__batch__(np.array(X)[p], self.batch_size)
        Y = self.__batch__(np.array(Y)[p], self.batch_size)
        A = self.__batch__(np.array(A)[p], self.batch_size)

        ps = np.random.permutation(len(Xs))
        Xs = self.__batch__(np.array(Xs)[ps], self.batch_size)
        Ys = self.__batch__(np.array(Ys)[ps], self.batch_size)
        As = self.__batch__(np.array(As)[ps], self.batch_size)


        Single = 0.0
        Seq = 0.0
        Corr = 0.0
        import sys
        start = time.time()
        for i in range(len(X)):
            single, corr, corr_out, pred_out = self.sess.run([self.loss_single, self.loss_corr, self.autoencoded,self.decodeds],
                                     feed_dict={
                                                self.x: X[i],
                                                self.a: A[i],
                                                self.y: Y[i]
                                               })
            Single += np.mean(single)
            Corr += np.mean(corr)
            # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
            sys.stdout.write(str(round(float(50*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')

        for i in range(len(Xs)):
            seq, pred_open = self.sess.run([self.loss_seq, self.decoded_preds],
                                     feed_dict={
                                                self.x: Xs[i],
                                                self.a_seq: As[i],
                                                self.y_seq: Ys[i]
                                               })
            Seq += np.mean(seq)
            # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
            sys.stdout.write(str(50+round(float(50*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        # sys.stdout.write('Done\r\n')
        return Single / len(X), Seq / len(X), Corr / len(X)



        # G, yS, yR, yD, X, S, A = self.__prep_data__(data, batch_size=self.batch_size)
        # Single = 0.0
        # Seq = 0.0
        # Corr = 0.0
        # import sys
        # start = time.time()
        # for i in range(min(len(X), len(A), len(S))):
        #     single, seq, corr, corr_out, pred_out = self.sess.run([self.loss_single, self.loss_seq, self.loss_corr,
        #                                                                     self.decoded_corrs,self.decoded_preds],
        #                              feed_dict={
        #                                         self.x: X[i][:,:self.state_dim],
        #                                         self.a_seq: A[i],
        #                                         self.y_seq: S[i]
        #                                        })
        #     Single += np.mean(single)
        #     Seq += np.mean(seq)
        #     Corr += np.mean(corr)
        #     # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
        #     # sys.stdout.write(str(round(float(100*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        # # sys.stdout.write('Done\r\n')
        # return Single / len(X), Seq / len(X), Corr / len(X)

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
        if num >= 0:
            return new_obs[num]
        return new_obs


    def uncertainty(self, action_in, obs_in=None):
        # print('Stepping!')
        if obs_in is not None:
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

        if self.num_components < 1:
            import sys; sys.exit('Error insufficient components (needs to be greater than 1)')
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
