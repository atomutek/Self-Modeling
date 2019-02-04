import tensorflow as tf
import numpy as np
import time
from collections import deque

def encode_var(s, latent_size, h=None):
    if h is not None:
        state0_in = h[0]
        state1_in = h[1]
        state2_in = h[2]
    else:
        state0_in = None
        state1_in = None
        state2_in = None
    # state1 = None
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as scope:
        # sequence = tf.concat([x, a], axis=2)
        sequence = s
        output0, state0 = tf.nn.dynamic_rnn(
            tf.contrib.rnn.LSTMCell(latent_size*4, activation=tf.nn.relu),
            sequence,
            dtype=tf.float32,
            sequence_length=length(sequence),
            scope='var_encoder0',
            initial_state = state0_in
        )
        sequence = tf.concat([sequence, output0], axis=2)
        output1, state1 = tf.nn.dynamic_rnn(
            tf.contrib.rnn.LSTMCell(latent_size*2, activation=tf.nn.relu),
            sequence,
            dtype=tf.float32,
            sequence_length=length(sequence),
            scope='var_encoder1',
            initial_state = state1_in
        )
        sequence = tf.concat([output0, output1], axis=2)
        output2, state2 = tf.nn.dynamic_rnn(
            tf.contrib.rnn.LSTMCell(latent_size),
            sequence,
            dtype=tf.float32,
            sequence_length=length(sequence),
            scope='var_encoder2',
            initial_state = state2_in
        )
    return output2, [state0, state1, state2]

def encode_mean(s, latent_size, h=None):
    if h is not None:
        state0_in = h[0]
        state1_in = h[1]
        state2_in = h[2]
    else:
        state0_in = None
        state1_in = None
        state2_in = None
    # state1 = None
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as scope:
        # sequence = tf.concat([x, a], axis=2)
        sequence = s
        output0, state0 = tf.nn.dynamic_rnn(
            tf.contrib.rnn.LSTMCell(latent_size*4, activation=tf.nn.relu),
            sequence,
            dtype=tf.float32,
            sequence_length=length(sequence),
            scope='var_encoder0',
            initial_state = state0_in
        )
        sequence = tf.concat([sequence, output0], axis=2)
        output1, state1 = tf.nn.dynamic_rnn(
            tf.contrib.rnn.LSTMCell(latent_size*2, activation=tf.nn.relu),
            sequence,
            dtype=tf.float32,
            sequence_length=length(sequence),
            scope='mean_encoder1',
            initial_state = state1_in
        )
        sequence = tf.concat([output0, output1], axis=2)
        output2, state2 = tf.nn.dynamic_rnn(
            tf.contrib.rnn.LSTMCell(latent_size),
            sequence,
            dtype=tf.float32,
            sequence_length=length(sequence),
            scope='mean_encoder2',
            initial_state = state2_in
        )
        return output2, [state0, state1, state2]

def decoder(latent, out_dim, h=None):
    # if h is not None:
    #     state0_in = h[0]
    #     state1_in = h[1]
    #     state2_in = h[2]
    # else:
    #     state0_in = None
    #     state1_in = None
    #     state2_in = None
    # # state1 = None
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as scope:
        out = latent
        out = tf.nn.relu(tf.layers.dense(out, 256, name='mlp0'))
        out = tf.nn.relu(tf.layers.dense(out, 512, name='mlp1'))
        out = tf.nn.tanh(tf.layers.dense(out, out_dim, name='mlp2'))
        return out, None
    #     sequence = latent
    #     output0, state0 = tf.nn.dynamic_rnn(
    #         tf.contrib.rnn.LSTMCell(256, activation=tf.nn.relu),
    #         sequence,
    #         dtype=tf.float32,
    #         sequence_length=length(sequence),
    #         scope='var_encoder0',
    #         initial_state = state0_in
    #     )
    #     sequence = tf.concat([sequence, output0], axis=2)
    #     output1, state1 = tf.nn.dynamic_rnn(
    #         tf.contrib.rnn.LSTMCell(512, activation=tf.nn.relu),
    #         sequence,
    #         dtype=tf.float32,
    #         sequence_length=length(sequence),
    #         scope='decoder1',
    #         initial_state = state1_in
    #     )
    #     sequence = tf.concat([sequence, output1], axis=2)
    #     output2, state2 = tf.nn.dynamic_rnn(
    #         tf.contrib.rnn.LSTMCell(out_dim),
    #         sequence,
    #         dtype=tf.float32,
    #         sequence_length=length(sequence),
    #         scope='decoder2',
    #         initial_state = state2_in
    #     )
    #     return output2, [state0, state1, state2]

def predictor(latent, action, out_dim):
    with tf.variable_scope('predictor', reuse=tf.AUTO_REUSE) as scope:
        out = tf.concat([latent, action], axis=1)
        # out = tf.layers.dense(out, 512, name='mlp0')
        out = tf.nn.tanh(tf.layers.dense(out, out_dim, name='mlp1'))
        return out

def KLD_loss(latent_var, latent_mean):
    return -0.5 * tf.reduce_mean(1.0 + latent_var - tf.pow(latent_mean, 2) - tf.exp(latent_var))

# taken from https://danijar.com/variable-sequence-lengths-in-tensorflow/
def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length

## Note to self: Double check
def cost(output, target):
    mse = tf.reduce_mean(tf.square(output-target), axis=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
    mse *= mask
    # Average over actual sequence lengths.
    mse = tf.reduce_sum(mse, 1)
    mse /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(mse)

def last_relevant(output):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length(output) - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


# maybe encode the last n state/actions in the encoder trained as a WAE/VAE this would be a measure of how familiar it is with the current state
    # if this was done well latent states could be passed from "memory" to leverage other experiences
# have a secondary decoder that takes the encoding as well as an action to output the next real obs this would be the forward speculation

from tensorflow_src.env_learners.env_learner import EnvLearner
class SeqEnvLearner(EnvLearner):
    def __init__(self, env_in):
        EnvLearner.__init__(self, env_in)

        self.latent_size = 128
        self.max_seq_len = 300
        self.batch_size = 256

        lr = 1e-4

        # self.x = tf.placeholder(dtype=tf.float32, shape=([None, self.state_dim]))
        self.a = tf.placeholder(dtype=tf.float32, shape=([None, self.act_dim]))
        self.y = tf.placeholder(dtype=tf.float32, shape=([None, self.state_dim]))

        # self.a_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.max_seq_len, self.act_dim]))
        # self.y_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.max_seq_len, self.state_dim]))
        self.x_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.max_seq_len, self.state_dim]))
        self.state_seq = tf.placeholder(dtype=tf.float32, shape=([None, self.max_seq_len, self.act_dim+2*self.state_dim]))

        self.latent_mean, o = encode_mean(self.state_seq, self.latent_size)
        self.latent_var, _ = encode_var(self.state_seq, self.latent_size)

        self.train_latent = tf.random_normal(dtype=tf.float32, shape=([self.batch_size, self.max_seq_len, self.latent_size]))
        self.train_latent = self.train_latent*tf.exp(0.5*self.latent_var)+self.latent_mean

        self.decoded, _ = decoder(self.train_latent, self.state_dim)
        self.decoded_out, _ = decoder(self.latent_mean, self.state_dim)

        self.latent_in = tf.placeholder(dtype=tf.float32, shape=([None, None, self.latent_size]))
        self.decoded_test, _ = decoder(self.latent_in, self.state_dim)



            # latent, self.encoder_states = self.sess.run(encode_mean(
            #     np.array([[np.float32(np.concatenate([self.last_obs, a, new_obs]))]]),
            #     self.latent_size,
            #     h=self.encoder_states
            # ))
        # self.test_state_seq = tf.placeholder(dtype=tf.float32, shape=([1, 1, self.act_dim+2*self.state_dim]))
        # self.encoder_states_in = tf.placeholder(dtype=tf.float32, shape=([1, 1, self.latent_size]))
        # self.test_encoding, self.encoder_states_out = encode_mean(self.test_state_seq, self.latent_size, h=self.encoder_states_in)



        self.out = predictor(last_relevant(self.train_latent), self.a, self.state_dim)

        # self.loss_seq = cost(self.outputs, self.y_seq)
        # self.train_step_seq = tf.train.AdamOptimizer(lr).minimize(self.loss_seq)

        self.loss_single = tf.reduce_mean(tf.square(self.out-self.y))
        self.train_step_single = tf.train.AdamOptimizer(lr).minimize(self.loss_single)

        self.kl_loss = cost(self.decoded, self.x_seq) + KLD_loss(self.latent_var, self.latent_mean)
        self.train_step_kl = tf.train.AdamOptimizer(lr).minimize(self.kl_loss)

    def __prep_data__(self, data, batch_size=32):
        X = []
        A = []
        Y = []
        reset=True

        S = []
        Xs = []

        s = deque([np.zeros(2*self.state_dim+self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        x = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        for i in range(len(data)):
            # obs = data[i][0]/self.state_mul_const
            # act = data[i][1]/self.act_mul_const
            # new_obs = data[i][3]/self.state_mul_const
            obs = data[i][0]/self.state_mul_const
            act = data[i][1]/self.act_mul_const
            new_obs = data[i][3]/self.state_mul_const

            if reset:
                s = deque([np.zeros(2*self.state_dim+self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                x = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                # s.append([np.concatenate([obs, act, new_obs])])
            else:
                X.append(obs)
                A.append(act)
                Y.append(new_obs)
                x.appendleft(obs)
                Xs.append(np.array(x))
            s.appendleft(np.concatenate([obs, act, new_obs]))
            reset = data[i][4]
            if not reset:
                S.append(np.array(s))

        assert len(X) == len(Y) == len(A) == len(S)

        p = np.random.permutation(len(X))
        # p = np.arange(len(X))
        X = self.__batch__(np.array(X)[p], self.batch_size)
        Y = self.__batch__(np.array(Y)[p], self.batch_size)
        A = self.__batch__(np.array(A)[p], self.batch_size)
        S = self.__batch__(np.array(S)[p], self.batch_size)
        Xs = self.__batch__(np.array(Xs)[p], self.batch_size)
        return X, Y, A, S, Xs

    def get_loss(self, data):
        import sys
        start = time.time()
        X, Y, A, S, Xs = self.__prep_data__(data)
        Seq = 0
        KL = 0
        for i in range(len(S)):
            seq, kl, decoded = self.sess.run([self.loss_single, self.kl_loss, self.decoded],
                                     feed_dict={
                                                self.state_seq: S[i],
                                                self.x_seq: Xs[i],
                                                self.a: A[i],
                                                self.y: Y[i]
                                               })
            Seq += np.mean(seq)
            KL += np.mean(kl)
            # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
            sys.stdout.write(str(round(float(100*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
        # sys.stdout.write('Done\r\n')
        return Seq / len(X), KL/len(X)

    def train_epoch(self, data, eager=True):
        import sys, math
        start = time.time()
        Seq = 0
        KL = 0
        if eager:
            reset=True
            Xs = []
            S = []
            A = []
            Y = []
            s = deque([np.zeros(2*self.state_dim+self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
            x = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)

            batches_done = 0.0

            for i in range(len(data)):
                obs = data[i][0]/self.state_mul_const
                act = data[i][1]/self.act_mul_const
                new_obs = data[i][3]/self.state_mul_const

                if reset:
                    s = deque([np.zeros(2*self.state_dim+self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                    x = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                    # s.append([np.concatenate([obs, act, new_obs])])
                else:
                    A.append(act)
                    Y.append(new_obs)
                    x.append(obs)
                    Xs.append(x)
                s.appendleft(np.concatenate([obs, act, new_obs]))
                reset = data[i][4]
                if not reset:
                    S.append(np.array(s))

                if min(len(Xs), len(S), len(A), len(Y)) > self.batch_size:
                    batch_S = S[:self.batch_size]
                    batch_Xs = Xs[:self.batch_size]
                    batch_A = A[:self.batch_size]
                    batch_Y = Y[:self.batch_size]

                    S = S[self.batch_size:]
                    Xs = Xs[self.batch_size:]
                    A = A[self.batch_size:]
                    Y = Y[self.batch_size:]

                    seq, kl, _, = self.sess.run([self.loss_single, self.kl_loss, self.train_step_kl],
                                             feed_dict={
                                                        self.state_seq: np.array(batch_S),
                                                        self.x_seq: np.array(batch_Xs),
                                                        self.a: np.array(batch_A),
                                                        self.y: np.array(batch_Y)
                                                       })
                    # if math.isnan(seq) or math.isnan(kl):
                    if math.isnan(kl):
                        print('Error NaN Exception')
                        # if math.isnan(seq): print('\'single\' loss returned NaN')
                        if math.isnan(kl): print('\'KL\' loss returned NaN')
                        exit()
                    Seq += np.mean(seq)
                    KL += np.mean(kl)
                    sys.stdout.write(str(round(float(100*i)/float(len(data)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
                    batches_done += 1.0
            return Seq / batches_done, KL / batches_done
        else:
            X, Y, A, S, Xs = self.__prep_data__(data)
            for i in range(len(S)):
                seq, kl, _, = self.sess.run([self.loss_single, self.kl_loss, self.train_step_kl],
                                         feed_dict={
                                                    self.state_seq: S[i],
                                                    self.x_seq: Xs[i],
                                                    self.a: A[i],
                                                    self.y: Y[i]
                                                   })
                if math.isnan(seq) or math.isnan(kl):
                    print('Error NaN Exception')
                    if math.isnan(seq): print('\'single\' loss returned NaN')
                    else: print('\'KL\' loss returned NaN')
                    exit()
                Seq += np.mean(seq)
                KL += np.mean(kl)
                # sys.stdout.write('{0:.2f}% Done in {0:.2f} s                                    \r'.format(float(100*i)/float(len(X)), time.time()-start))
                sys.stdout.write(str(round(float(100*i)/float(len(X)), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')
            # sys.stdout.write('Done\r\n')

            # mean, var = self.sess.run([self.latent_mean, self.latent_var],
            #                              feed_dict={
            #                                         self.state_seq: S[i],
            #                                         self.x_seq: Xs[i],
            #                                         self.a: A[i],
            #                                         self.y: Y[i]
            #                                        })
            # print('Mean: ')
            # print(mean)
            # print('Var: ')
            # print(var)
            return Seq / len(X), KL / len(X)

    def train(self, train, total_steps, valid=None, log_interval=1, early_stopping=-1, saver=None, save_str=None, verbose=True):
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

        single, kl = self.get_loss(valid)
        print('Valid Single: ' + str(single))
        # print('Valid Seq: ' + str(seq))
        print('Valid KL: ' + str(kl))
        for i in range(total_steps):
            if i == seq_idx[seq_i] and self.seq_len < self.max_seq_len:
                self.seq_len += 1
                seq_i += 1

            if i % log_interval == 0 and i > 0 and valid is not None:
                single, kl = self.get_loss(valid)
                print('Epoch: ' + str(i) + '/' + str(total_steps))
                print('Valid Single: ' + str(single))
                # print('Valid Seq: ' + str(seq))
                print('Valid KL: ' + str(kl))
                print('')
                if saver is not None and save_str is not None:
                    save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
                    print("Model saved in path: %s" % save_path)
            start = time.time()
            single, kl = self.train_epoch(train)
            duration = time.time() - start
            if stop_count > early_stopping and early_stopping > 0:
                break
            # if i % log_interval != 0 or i == 0:
            print('Epoch: ' + str(i) + '/' + str(total_steps) + ' in ' + str(duration) + 's                                          ')
            print('Train Single: ' + str(single))
            # print('Train Seq: ' + str(seq))
            print('Train KL: ' + str(kl))
            print('')
        if valid is not None:
            single, kl = self.get_loss(valid)
            print('Final Epoch')
            print('Valid Single: ' + str(single))
            # print('Valid Seq: ' + str(seq))
            print('Valid KL: ' + str(kl))
            print('')
        if saver is not None and save_str is not None:
            save_path = saver.save(self.sess, 'models/' + str(save_str) + '.ckpt')
            print("Final Model saved in path: %s" % save_path)

    def autoencode(self, a, obs):
        start = time.time()
        new_obs = obs/self.state_mul_const
        stop1 = time.time()-start
        state_in = np.array([[np.float32(np.concatenate([self.last_obs, a, new_obs]))]])
        latent, self.encoder_states = self.sess.run(encode_mean(
            state_in,
            self.latent_size,
            h=self.encoder_states
        ))
        stop2 = time.time()-start
        # autoencoded, _ = self.sess.run(decoder(latent, self.state_dim))
        autoencoded = self.sess.run(self.decoded_test, feed_dict={self.latent_in: latent})*self.state_mul_const
        stop3 = time.time()-start
        self.last_obs = new_obs
        stop4 = time.time()-start
        out = autoencoded[0][0]
        stop5 = time.time()-start
        diff = np.mean(np.abs(out-obs))
        stop6 = time.time()-start
        print('Diff: '+str(diff))
        print('Rand: '+str(np.mean(np.abs(obs-np.random.uniform(-1, 1, 21)*self.state_mul_const))))
        print('Zeros: '+str(np.mean(np.abs(obs-np.zeros_like(obs)))))
        stop7 = time.time()-start
        return out

    def reset(self, obs_in):
        # self.x_buff = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        # self.a_buff = deque([np.zeros(self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        self.buff_fill = 0
        # self.x_buff.appendleft(obs_in)
        self.state_buff = deque([np.zeros(2*self.state_dim+self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        # self.state_buff.appendleft(np.concatenate([obs_in, np.zeros(self.act_dim), obs_in]))
        self.last_obs = obs_in/self.state_mul_const
        self.encoder_states = None
        self.decoder_states = None

    def step(self, action_in, obs_in=None, episode_step=None, save=True, buff=None, num=None):
        outs = self.sess.run(self.out, feed_dict={
            self.state_seq: np.array([self.state_buff]),
            self.a: np.array([action_in])
        })
        out = outs[0]
        if obs_in is None:
            new_obs = out
        else:
            new_obs = obs_in/self.state_mul_const
        self.state_buff.appendleft(np.concatenate([self.last_obs, action_in, new_obs]))
        self.last_obs = new_obs

        return out
