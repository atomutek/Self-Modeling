import datetime
import math
import time

import gym
import numpy as np
import tensorflow as tf
from gym import spaces


class RealerAntWrapper(gym.Env):
    def __init__(self, ant_env):
        self.env = ant_env
        # self.env.render(mode="human")
        # self.env.seed(0)
        # self.env.render()
        self.action_space = self.env.action_space
        self.action_space.high *= 7
        self.action_space.low *= 7

        self.front = 3
        self.back = 4
        obs_ones = np.ones(shape=(self.env.observation_space.shape[0]-self.front-self.back,))
        self.observation_space = spaces.Box(high=5*obs_ones, low=-5*obs_ones)
        print('State Dim: '+str(obs_ones.shape[0]))

        # State Summary (dim=25):
        # state[0] = vx
        # state[1] = vy
        # state[2] = vz
        # state[3] = roll
        # state[4] = pitch
        # state[5-20] = Joint relative positions
        #    even elements [0::2] position, scaled to -1..+1 between limits
        #    odd elements  [1::2] angular speed, scaled to show -1..+1
        # state[21-24] = feet contacts

        pass
    def reset(self):
        obs = self.env.reset()
        # if self.back > 0:
        #     return obs[self.front:-self.back]
        # else:
        #     return obs[self.front:]
        obs =  obs[self.front:-self.back]

        # x, y, z, r, p, positions
        # obs = np.concatenate([obs[self.front:self.front+4], obs[self.front+4::2][:-self.back/2]])
        return obs

    def step(self, action):
        new_obs, r, done, info = self.env.step(action)
        new_obs =  new_obs[self.front:-self.back]

        # x, y, z, r, p, positions
        # new_obs = np.concatenate([new_obs[self.front:self.front+4], new_obs[self.front+4::2][:-self.back/2]])
        return new_obs, r, done, info

    def reset_raw(self):
        obs = self.env.reset()
        return obs
    def step_raw(self, action):
        new_obs, r, done, info = self.env.step(action)
        return new_obs, r, done, info
    def render(self, mode='human'):
        return self.env.render(mode)

def test(env, env_learner, epochs=100, train_episodes=10, test_episodes=100, loop='open', show_model=False, load=None):
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    print('scaling actions by {} before executing in env'.format(max_action))
    print('Done Env Learner')
    print('Using agent with the following configuration:')
    try:
        saver = tf.train.Saver()
    except:
        saver=None
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    gpu_options = None
    num_cpu = 4
    if gpu_options is None:
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
    else:
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu,
            gpu_options=gpu_options)

    tf_config.gpu_options.allow_growth = True
    episode_duration = -1
    nb_valid_episodes = 5
    episode_step = 0
    episode_reward = 0.0
    max_ep_rew = -10000


    train = []
    valid = []

    with tf.Session(config=tf_config) as sess:
        sess_start = time.time()
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        data_log = open('logs/'+datetime_str+'_log.txt', 'w+')

        if load is not None:
            saver.restore(sess, load)
            print('Model: ' + load + ' Restored')
            env_learner.initialize(sess, load=True)


        # generic data gathering
        obs = env.reset()
        # import random

        # print('Episode 0')
        # print(obs)
        amp_std = 0.1
        off_std = 0.1
        np.random.seed(0)

        amplitude = 1.
        offset = 0.

        best_rew = 0.0
        best_amp = amplitude
        best_off = offset


        if load is not None:
            saver.restore(sess, load)
            print('Model: ' + load + ' Restored')
            env_learner.initialize(sess, load=True)

        if load is None  or train_episodes > 0:
            env_learner.initialize(sess)
            # sess.graph.finalize()
            i = 0
            while i < nb_valid_episodes:
                action = np.random.uniform(-1, 1, env.action_space.shape[0])
                new_obs, r, done, info = env.step(max_action * action)
                if episode_duration > 0:
                    done = (done or (episode_step >= episode_duration))
                valid.append([obs, max_action * action, r, new_obs, done, episode_step])
                episode_step += 1
                obs = new_obs

                episode_reward += r
                if done:
                    obs = env.reset()
                    max_ep_rew = max(max_ep_rew, episode_reward)
                    episode_reward = 0.0
                    i += 1
            i = 0


            train_batch_size = 1000
            max_steps = train_episodes*1000
            update_interval = 2*max_steps/epochs

            epoch = 0
            train_subst = []
            while len(train) < max_steps:
                action = env_learner.next_move(obs, episode_step)
                new_obs, r, done, info = env.step(max_action * action)
                train.append([obs, max_action * action, r, new_obs, done, episode_step])
                train_subst.append([obs, max_action * action, r, new_obs, done, episode_step])
                episode_step += 1
                obs = new_obs

                episode_reward += r
                if done:
                    episode_step = 0.0
                    obs = env.reset()
                    max_ep_rew = max(max_ep_rew, episode_reward)
                    episode_reward = 0.0
                    print(len(train))
                    i += 1

                if len(train) % update_interval == 0 and len(train) > 0:
                    epoch += 1




                    # replay based on uncertainty TODO: Replace with proper priority replay
                    # train_subst = []
                    # scores = []
                    # for i in range(len(train)):
                    #     scores.append( (env_learner.uncertainty(train[i][1]/max_action, train[i][0]), i) )
                    # scores.sort(key=lambda tup: tup[0])
                    # for i in range(min(train_batch_size, len(scores))):
                    #     train_subst.append(train[scores[i][1]])



                    start = time.time()
                    for i in range(10):
                        (single, seq, corr) = env_learner.train_epoch(train_subst)
                    duration = time.time() - start
                    print('Epoch: ' + str(epoch) + '/' + str(epochs) + ' in ' + str(duration) + 's')
                    print('Train Single: ' + str(single))
                    print('Train Seq: ' + str(seq))
                    print('Train Corr: ' + str(corr))
                    print('')

            save_path = saver.save(sess, 'models/' + str(datetime_str) + '.ckpt')
            print("Model saved in path: %s" % save_path)
            print('Train Size: ' + str(len(train)))
            print('Valid Size: ' + str(len(valid)))

            env_learner.train(train, epochs-epoch, valid, saver=saver, save_str=datetime_str, verbose=True)
            print('Trained Self Model')
