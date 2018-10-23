#!/usr/bin/env python3
import os
from baselines.common import tf_util as U
from envs.widowx_arm import WidowxROS
from test_planners.test_plan_walker import AntWrapper

import time,datetime, math, argparse
import gym
from gym import spaces
import pybullet, pybullet_envs
pybullet.connect(pybullet.DIRECT)
import numpy as np
import tensorflow as tf

from env_learners.dnn_env_learner import DNNEnvLearner
from env_learners.preco_wae_env_learner import PreCoWAEEnvLearner
from env_learners.preco_env_learner import PreCoEnvLearner
from env_learners.preco_gen_env_learner import PreCoGenEnvLearner

class LearnedAntWrapperRew(gym.Env):
    def __init__(self, ant_env, env_learner=None, loop='closed'):
        self.env = ant_env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        if env_learner is not None:
            print('Loading Learned Env')
        else:
            print('Loading Real Env')

        self.env_learner = env_learner
        self.loop = loop

    def reset(self):
        obs = self.env.reset()
        if self.env_learner is not None:
            self.env_learner.reset(obs)
        self.frame = 0
        self.new_obs = obs
        return obs

    def step(self, action):
        if self.env_learner is not None:
            if self.loop == 'open':
                pred_obs = self.env_learner.step(action_in=action, episode_step=self.frame)
            elif self.loop == 'closed' or self.loop == 'open2':
                pred_obs = self.env_learner.step(action_in=action, obs_in=self.new_obs, episode_step=self.frame)
            else:
                pred_obs = self.env_learner.step(action_in=action, obs_in=self.new_obs, episode_step=self.frame)

            obs_in, r, done, info = self.env.step(action)
            if self.loop == 'open2':
                self.new_obs = pred_obs
            else:
                self.new_obs = obs_in
            obs = pred_obs
        else:
            obs, r, done, info = self.env.step(action)
        self.frame += 1
        done = self.frame > 999 or (self.env_learner is None and done)
        r = (obs[0]/0.3)/60
        return obs, r, done, info

    def render(self, mode='human'):
        self.env.render(mode)

def train(num_timesteps, seed, model_path=None, load=None, self=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    # env = WidowxROS(test=True)

    if self is None:
        env_in = AntWrapper(gym.make("AntBulletEnv-v0"))
        print('Running for '+str(num_timesteps)+' steps')
        # parameters below were the best found in a simple random search
        # these are good enough to make humanoid walk, but whether those are
        # an absolute best or not is not certain
        env = LearnedAntWrapperRew(env_in)
        env = RewScale(env, 100)
        pi = pposgd_simple.learn(env, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_actorbatch=2048,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10,
                optim_stepsize=3e-4,
                optim_batchsize=64,
                gamma=0.99,
                lam=0.95,
                schedule='linear',load=load
            )
        env.close()
        if model_path:
            model_path += '-'+datetime_str
            print('Model Saved to: '+str(model_path))
            U.save_state(model_path)

        return pi
    else:
        env_in = AntWrapper(gym.make("AntBulletEnv-v0"))
        env_learner = PreCoGenEnvLearner(env_in)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        gpu_options = None
        num_cpu = 1
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)

        with tf.Session(config=tf_config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self)
            print('Model: ' + self + ' Restored')
            env_learner.initialize(sess, load=True)

            env = LearnedAntWrapperRew(env_in, env_learner)
            env = RewScale(env, 100)
            print('Running for '+str(num_timesteps)+' steps')
            pi = pposgd_simple.learn(env, policy_fn,
                    max_timesteps=num_timesteps,
                    timesteps_per_actorbatch=2048,
                    clip_param=0.2, entcoeff=0.0,
                    optim_epochs=10,
                    optim_stepsize=3e-4,
                    optim_batchsize=64,
                    gamma=0.99,
                    lam=0.95,
                    schedule='linear',load=load
                )
            env.close()
            if model_path:
                model_path += '-'+datetime_str
                print('Model Saved to: '+str(model_path))
                U.save_state(model_path)
            return pi

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale

def main():
    # screen -r 2 == closed
    # screen -r 1 == Real
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', default='models/ant_policy')
    parser.add_argument('--load', default=None)
    parser.add_argument('--self', default=None)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--play', default=False)
    parser.add_argument('--num_timesteps', default=int(2e7))

    args = parser.parse_args()

    if not args.play:
        # train the model
        # if args.load:
        #     train(num_timesteps=1, seed=args.seed)
        train(num_timesteps=int(args.num_timesteps), seed=args.seed, model_path=args.model_path, load=args.load, self=args.self)
    else:
        # construct the model object, load pre-trained model and render
        pi = train(num_timesteps=1, seed=args.seed)
        if args.load is None:
            U.load_state(args.model_path)
        else:
            U.load_state(args.load)

        # env = WidowxROS(test=True)
        env = AntWrapper(gym.make("AntBulletEnv-v0"))
        # env = gym.make("AntBulletEnv-v0")
        ob = env.reset()


        test_iters = 10
        i = 0
        pos = np.zeros(3)
        real_pos_chart = []
        while i < test_iters:
            action = pi.act(stochastic=False, ob=ob)[0]
            ob, _, done, _ =  env.step(action)
            # env.render()
            pos += (ob[0:3]/0.3)/60
            real_pos_chart.append(pos.copy())
            # pos += (ob[3:6]/0.3)/60
            if done:
                print(pos)
                ob = env.reset()
                pos = np.zeros(3)
                i += 1

                # from matplotlib import pyplot as plt
                import math
                # print(np.amax(acts, axis=0))
                # print(np.amin(acts, axis=0))

                # real_pos_chart = np.array(real_pos_chart)
                # xr, yr, zr = np.hsplit(real_pos_chart, 3)
                # plt.plot(xr, yr)
                #
                # max_lim = 1.1*max(np.max(xr), np.max(yr))
                # min_lim = 1.1*min(np.min(xr), np.min(yr))
                # plt.xlim(min_lim, max_lim)
                # plt.ylim(min_lim, max_lim)
                # plt.show()
                # plt.clf()
                # real_pos_chart = []



if __name__ == '__main__':
    main()
