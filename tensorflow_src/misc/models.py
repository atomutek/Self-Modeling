import tensorflow as tf

def generator_model(x, out_dim, drop_rate=0.5):
    return generator_model_arm(x, out_dim, drop_rate)
    # return generator_model_walker(x, out_dim, drop_rate)

def discriminator_model(x, drop_rate=0.5):
    return discriminator_model_arm(x, drop_rate)
    # return discriminator_model_walker(x, drop_rate)


def generator_model_walker(x, out_dim, drop_rate=0.5):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
        x_seq = []
        for x_tmp in x:
            x_tmp = tf.layers.batch_normalization(x_tmp)
            x_seq.append(x_tmp)

        # x_seq = tf.split(x, buff_len, 1)
        # rnn_cell = tf.contrib.rnn.BasicLSTMCell(1024)
        rnn_cell = tf.contrib.rnn.GRUCell(1024)
        outputs, states = tf.nn.static_rnn(rnn_cell, x_seq, dtype=tf.float32)
        x = outputs[-1]

        x_new = []
        # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
        for x in outputs:
            x = tf.expand_dims(x, -1)
            x_new.append(x)
        x = tf.concat(x_new, axis=2)
        x = tf.layers.conv1d(x, 64, 3)
        x = tf.layers.conv1d(x, 32, 1)
        x = tf.layers.flatten(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 512)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 256)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 128)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 256)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 512)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, out_dim)

        return tf.nn.tanh(x)

def discriminator_model_walker(x, drop_rate=0.5):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
        # drop_rate=0.0
        # x_seq = tf.split(x, buff_len, 1)
        x_seq = []
        for x_tmp in x:
            x_tmp = tf.layers.batch_normalization(x_tmp)
            x_seq.append(x_tmp)

        # rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
        rnn_cell = tf.contrib.rnn.GRUCell(1024)
        outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
        x = outputs[-1]

        x_new = []
        # # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
        for x in outputs:
            x = tf.expand_dims(x, -1)
            x_new.append(x)
        x = tf.concat(x_new, axis=2)
        x = tf.layers.conv1d(x, 64, 3)
        x = tf.layers.conv1d(x, 32, 1)
        x = tf.layers.flatten(x)

        # x = tf.layers.batch_normalization(x)
        # x = tf.layers.dense(x, 1024)
        # x = tf.layers.dropout(x, rate=drop_rate)
        # x = tf.nn.leaky_relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 512)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.leaky_relu(x)

        # x = tf.layers.batch_normalization(x)
        # x = tf.layers.dense(x, 512)
        # x = tf.layers.dropout(x, rate=drop_rate)
        # x = tf.nn.leaky_relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 256)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.leaky_relu(x)
        
        # x = tf.layers.batch_normalization(x)
        # x = tf.layers.dense(x, 256)
        # x = tf.layers.dropout(x, rate=drop_rate)
        # x = tf.nn.leaky_relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 128)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.leaky_relu(x)

        # x = tf.layers.batch_normalization(x)
        # x = tf.layers.dense(x, 128)
        # x = tf.layers.dropout(x, rate=drop_rate)
        # x = tf.nn.leaky_relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 1)
        # return x

        # return tf.nn.leaky_relu(x)
        return tf.nn.sigmoid(x)

def generator_model_arm(x, out_dim, drop_rate=0.5):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
        x_seq = []
        for x_tmp in x:
            x_tmp = tf.layers.batch_normalization(x_tmp)
            x_seq.append(x_tmp)

        # x_seq = tf.split(x, buff_len, 1)
        # rnn_cell = tf.contrib.rnn.BasicLSTMCell(1024)
        rnn_cell = tf.contrib.rnn.GRUCell(1024)
        outputs, states = tf.nn.static_rnn(rnn_cell, x_seq, dtype=tf.float32)
        x = outputs[-1]

        x_new = []
        # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
        for x in outputs:
            x = tf.expand_dims(x, -1)
            x_new.append(x)
        x = tf.concat(x_new, axis=2)
        x = tf.layers.conv1d(x, 64, 3)
        x = tf.layers.conv1d(x, 32, 1)
        x = tf.layers.flatten(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 512)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 256)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 128)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 256)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, 512)
        x = tf.layers.dropout(x, rate=drop_rate)
        x = tf.nn.relu(x)

        x = tf.layers.batch_normalization(x)
        x = tf.layers.dense(x, out_dim)

        return tf.nn.tanh(x)

def discriminator_model_arm(x, drop_rate=0.5):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
        # x_seq = tf.split(x, buff_len, 1)
        x_seq = []
        for x_tmp in x:
            x_tmp = tf.layers.batch_normalization(x_tmp)
            x_seq.append(x_tmp)

        rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
        # rnn_cell = tf.contrib.rnn.GRUCell(512)
        outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
        x = outputs[-1]

        # x_new = []
        # # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
        # for x in x_seq:
        #     x = tf.expand_dims(x, -1)
        #     x_new.append(x)
        # x = tf.concat(x_new, axis=2)
        # x = tf.layers.conv1d(x, 64, 3)
        # x = tf.layers.conv1d(x, 32, 1)
        # x = tf.layers.flatten(x)

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
        x = tf.layers.dense(x, 1)
        # return x
        # return tf.nn.leaky_relu(x)
        return tf.nn.sigmoid(x)


def simple_fk_learner(x, drop_rate=0.5):
    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 64)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 64)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 2)

    return tf.nn.tanh(x, 'output')


def fk_learner(x, drop_rate=0.5):
    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 2048)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 2048)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 1024)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 1024)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 1024)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 512)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 1024)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 1024)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 2048)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 2048)
    x = tf.layers.dropout(x, rate=drop_rate)
    x = tf.nn.relu(x)

    x = tf.layers.batch_normalization(x)
    x = tf.layers.dense(x, 3)

    return tf.nn.tanh(x)


#
# RL Wrappers
#

# PPO

import baselines.common.tf_util as U
import gym
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd

class GenPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            # for i in range(num_hid_layers):
            #     last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            # self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]
            self.vpred = discriminator_model([last_out], drop_rate=0.5)

        with tf.variable_scope('pol'):
            last_out = obz
            # for i in range(num_hid_layers):
            #     last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))

            pdparam = generator_model([last_out], pdtype.param_shape()[0], drop_rate=0.5)

            # if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            #     mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
            #     logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            #     pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            # else:
            #     pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []



## DDPG
class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            x = generator_model([obs], self.nb_actions, drop_rate=0.5)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            x = discriminator_model([obs], drop_rate=0.5)
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


# #
# # OUTDATED
# #
# def simple_gen(x, out_dim, buff_len=None, drop_rate=None):
#     with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
#         x = tf.layers.batch_normalization(x)
#         x = tf.layers.dense(x, 64)
#         x = tf.layers.dropout(x, rate=drop_rate)
#         x = tf.nn.relu(x)
#         x = tf.layers.dense(x, out_dim)
#         return tf.nn.tanh(x)
# def simple_disc(x, drop_rate=None, buff_len=None):
#     with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
#
#         x_seq = []
#         for x_tmp in x:
#             x_tmp = tf.layers.batch_normalization(x_tmp)
#             x_seq.append(x_tmp)
#
#         rnn_cell = tf.contrib.rnn.BasicLSTMCell(64)
#         # rnn_cell = tf.contrib.rnn.GRUCell(512)
#         outputs, states = tf.nn.static_rnn(rnn_cell, x_seq, dtype=tf.float32)
#         x = outputs[-1]
#
#         x = tf.layers.dense(x, 64)
#         x = tf.layers.dropout(x, rate=drop_rate)
#         x = tf.nn.leaky_relu(x)
#         x = tf.layers.dense(x, 1)
#         # return x
#         return tf.nn.sigmoid(x)
#
# def done_model(x, out_dim, buff_len, drop_rate=0.5):
#     # x = tf.split(x, buff_len, 1)
#     # # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
#     # # for x in x_all:
#     # #     x = tf.expand_dims(x, -1)
#     # #     x_new.append(x)
#     # # x = tf.concat(x_new, axis=2)
#     # # # x = tf.layers.conv1d(x, 128, 7)
#     # # x = tf.layers.conv1d(x, 64, 5)
#     # # x = tf.layers.conv1d(x, 32, 3)
#     # # x = tf.layers.flatten(x)
#     # rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
#     #
#     # # rnn_cell = tf.contrib.rnn.GRUCell(512)
#     # outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
#     # x = outputs[-1]
#     #
#     # # x_new = []
#     # # for x in outputs:
#     # #     x = tf.expand_dims(x, -1)
#     # #     x_new.append(x)
#     # # x = tf.concat(x_new, axis=2)
#     # #
#     # # x = tf.layers.conv1d(x, 128, 7)
#     # # x = tf.layers.conv1d(x, 64, 5)
#     # # x = tf.layers.conv1d(x, 32, 3)
#     # # x = tf.layers.flatten(x)
#     # # x = tf.layers.batch_normalization(x)
#     #
#     #
#     # x = tf.layers.dense(x, 1024)
#     # # x = tf.nn.relu(x)
#     # x = tf.layers.dropout(x, rate=drop_rate)
#     #
#     # # # x = tf.layers.batch_normalization(x)
#     # # x = tf.layers.dense(x, 1024)
#     # # # x = tf.nn.relu(x)
#     # # x = tf.layers.dropout(x, rate=drop_rate)
#     #
#     # x = tf.layers.dense(x, 512)
#     # # x = tf.nn.relu(x)
#     # x = tf.layers.dropout(x, rate=drop_rate)
#
#     # x = tf.layers.batch_normalization(x)
#     x = tf.layers.dense(x, 256)
#     # x = tf.nn.relu(x)
#     x = tf.layers.dropout(x, rate=drop_rate)
#
#     # x = tf.layers.batch_normalization(x)
#     x = tf.layers.dense(x, 128)
#     # x = tf.nn.relu(x)
#     x = tf.layers.dropout(x, rate=drop_rate)
#
#     # x = tf.layers.batch_normalization(x)
#     x = tf.layers.dense(x, out_dim)
#     # return x
#     return tf.nn.sigmoid(x)
#
#
# def explore_model(state, act_dim, drop_rate=0.5):
#     with tf.variable_scope('explore') as scope:
#         x = state
#         x = tf.layers.batch_normalization(x)
#         x = tf.layers.dense(x, 1024)
#         x = tf.nn.relu(x)
#         x = tf.layers.dropout(x, rate=drop_rate)
#
#         x = tf.layers.batch_normalization(x)
#         x = tf.layers.dense(x, 1024)
#         x = tf.nn.relu(x)
#         x = tf.layers.dropout(x, rate=drop_rate)
#
#         x = tf.layers.batch_normalization(x)
#         x = tf.layers.dense(x, 512)
#         x = tf.nn.relu(x)
#         x = tf.layers.dropout(x, rate=drop_rate)
#
#         x = tf.layers.dense(x, 512)
#         x = tf.nn.relu(x)
#         x = tf.layers.dropout(x, rate=drop_rate)
#
#         x = tf.layers.dense(x, act_dim)
#         return tf.nn.tanh(x)
#
#
# def gen_state(x, out_dim, buff_len, drop_rate=0.0):
#     # # x = tf.layers.batch_normalization(x)
#     # #
#     # # x_all = tf.split(x, buff_len, 1)
#     # # x_new = []
#     # # for x in x_all:
#     # #     x = tf.layers.batch_normalization(x)
#     # #     x = tf.layers.dense(x, 1024)
#     # #     x = tf.layers.dropout(x, rate=drop_rate)
#     # #
#     # #     x = tf.layers.batch_normalization(x)
#     # #     x = tf.layers.dense(x, 1024)
#     # #     x = tf.layers.dropout(x, rate=drop_rate)
#     # #
#     # #     # x = tf.layers.batch_normalization(x)
#     # #     # x = tf.layers.dense(x, 512)
#     # #     # x = tf.layers.dropout(x, rate=drop_rate)
#     # #
#     # #     x_new.append(x)
#     # # # # rnn_cell = tf.contrib.rnn.BasicLSTMCell(512)
#     # #
#     # # x = x_new
#     # #
#     # x = tf.split(x, buff_len, 1)
#     # lstm = tf.contrib.rnn.MultiRNNCell([
#     #     tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(2048)),
#     #     #     #tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(1024), output_keep_prob=1.0-drop_rate),
#     #     # tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(512), output_keep_prob=1.0-drop_rate),
#     #     tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(1024), output_keep_prob=1.0 - drop_rate),
#     #     #     tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(256), output_keep_prob=1.0-drop_rate),
#     #     #     #tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(256), output_keep_prob=1.0-drop_rate),
#     #     #     tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(128), output_keep_prob=1.0-drop_rate),
#     #     #     #tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(128), output_keep_prob=1.0-drop_rate),
#     #     #     # tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(64), output_keep_prob=1.0-drop_rate),
#     #     #     #tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(64), output_keep_prob=1.0-drop_rate)
#     # ])
#     # #
#     # outputs, states = tf.nn.static_rnn(lstm, x, dtype=tf.float32)
#     # x = outputs[-1]
#
#
#
#
#     # x_seq = tf.split(x, buff_len, 1)
#
#     x_seq = tf.split(x, buff_len, 1)
#
#     # rnn_cell = tf.contrib.rnn.BasicLSTMCell(2048)
#     # # rnn_cell = tf.contrib.rnn.GRUCell(2048)
#     # outputs, states = tf.nn.static_rnn(rnn_cell, x_seq, dtype=tf.float32)
#     # # x = outputs[-1]
#     # x_seq = outputs
#
#     # x_new = []
#     # # CNNs structured according to https://wiki.eecs.yorku.ca/lab/MLL/projects:cnn4asr:start
#     # for x in x_seq:
#     #     x = tf.expand_dims(x, -1)
#     #     x_new.append(x)
#     # x = tf.concat(x_new, axis=2)
#     # x = tf.layers.conv1d(x, 64, 3)
#     # x = tf.layers.conv1d(x, 32, 1)
#     # x = tf.layers.flatten(x)
#
#     x = tf.layers.batch_normalization(x)
#     x = tf.layers.dense(x, 1024)
#     x = tf.nn.relu(x)
#     x = tf.layers.dropout(x, rate=drop_rate)
#     #
#     x = tf.layers.batch_normalization(x)
#     x = tf.layers.dense(x, 512)
#     x = tf.nn.relu(x)
#     x = tf.layers.dropout(x, rate=drop_rate)
#
#     x = tf.layers.batch_normalization(x)
#     x = tf.layers.dense(x, 256)
#     x = tf.nn.relu(x)
#     x = tf.layers.dropout(x, rate=drop_rate)
#
#     x = tf.layers.batch_normalization(x)
#     x = tf.layers.dense(x, 128)
#     x = tf.nn.relu(x)
#     x = tf.layers.dropout(x, rate=drop_rate)
#
#     x = tf.layers.batch_normalization(x)
#     x = tf.layers.dense(x, 64)
#     x = tf.nn.relu(x)
#     x = tf.layers.dropout(x, rate=drop_rate)
#
#     x = tf.layers.batch_normalization(x)
#     x = tf.layers.dense(x, out_dim)
#
#     return tf.nn.tanh(x)
