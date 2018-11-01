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

from matplotlib import pyplot as plt

from env_learners.dnn_env_learner import DNNEnvLearner
from env_learners.preco_wae_env_learner import PreCoWAEEnvLearner
from env_learners.preco_env_learner import PreCoEnvLearner
from env_learners.preco_gen_env_learner import PreCoGenEnvLearner

class LearnedAntWrapperRew(gym.Env):
    def __init__(self, ant_env, env_learner=None, loop='open'):
        self.env = ant_env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        if env_learner is not None:
            print('Loading Learned Env')
        else:
            print('Loading Real Env')

        self.env_learner = env_learner
        self.loop = loop
        print(str(self.loop)+' loop')

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

            info = {}
            done = False
            if self.loop == 'open2':
                self.new_obs = pred_obs
            elif self.loop == 'closed':
                obs_in, r, done, info = self.env.step(action)
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

def train(num_timesteps, seed, model_path=None, load=None, self=None, loop=None):
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
        env_learner = PreCoEnvLearner(env_in)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        gpu_options = None
        num_cpu = 4
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu)

        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self)
            print('Model: ' + self + ' Restored')
            env_learner.initialize(sess, load=True)

            env = LearnedAntWrapperRew(env_in, env_learner, loop)
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

# Doesn't seem to work, returns all white
def get_image_from_env():
    cam_dist = 3
    cam_yaw = 0
    cam_pitch = -30
    cam_roll=0
    render_width =320
    render_height = 240
    upAxisIndex = 2
    nearPlane = 0.01
    farPlane = 100
    camTargetPos = [0,0,0]
    fov = 60
    pybullet.connect(pybullet.DIRECT)
    start = time.time()
    viewMatrix = pybullet.computeViewMatrixFromYawPitchRoll(camTargetPos, cam_dist, cam_yaw, cam_pitch, cam_roll, upAxisIndex)
    aspect = render_width / render_height
    projectionMatrix = pybullet.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
    img_arr = pybullet.getCameraImage(render_width, render_height, viewMatrix,projectionMatrix, shadow=1,lightDirection=[1,1,1],renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    stop = time.time()

    w=img_arr[0] #width of the image, in pixels
    h=img_arr[1] #height of the image, in pixels
    rgb=img_arr[2] #color data RGB
    dep=img_arr[3] #depth data
    print('Imshow start!')
    plt.imshow(rgb)
    plt.pause(0.5)
    print('Imshow end!')

def main():
    # screen -r 2 == closed
    # screen -r 1 == Real
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', default='models/ant_policy')
    parser.add_argument('--load', default=None)
    parser.add_argument('--self', default='models/2018-10-29-00:26:01.ckpt')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--play', default=0)
    parser.add_argument('--num_timesteps', default=int(2e7))
    parser.add_argument('--visualize', default=False)
    parser.add_argument('--loop', default='open')

    args = parser.parse_args()

    if args.play < 1:
        # train the model
        # if args.load:
        #     train(num_timesteps=1, seed=args.seed)
        train(num_timesteps=int(args.num_timesteps), seed=args.seed, model_path=args.model_path, load=args.load,
              self=args.self, loop=args.loop)
    else:
        # construct the model object, load pre-trained model and render
        pi = train(num_timesteps=1, seed=args.seed)
        if args.load is None:
            U.load_state(args.model_path)
        else:
            U.load_state(args.load)

        # env = WidowxROS(test=True)
        env = AntWrapper(gym.make("AntBulletEnv-v0"))
        if args.visualize:
            a = env.render(mode="human")
        ob = env.reset()

        test_iters = int(args.play)
        i = 0
        pos = np.zeros(3)
        real_pos_chart = []
        x_poses = []
        n = 0
        import pickle as pkl
        acts = pkl.load(open('acts.pkl','r'))
        while i < test_iters:
            # action = pi.act(stochastic=False, ob=ob)[0]
            action = acts[n]
            # action = np.random.uniform(-1, 1, env.action_space.shape[0])
            acts.append(action)
            ob, _, done, _ =  env.step(action)
            if args.visualize:
                im = env.render('rgb_array')
                import cv2
                cv2.imwrite('open_walk_'+str(n)+'.png', im)
            n += 1

            time.sleep(0.0001)
            # env.render()
            pos += (ob[0:3]/0.3)/60
            real_pos_chart.append(pos.copy())
            # pos += (ob[3:6]/0.3)/60
            if done:
                print(str(i)+'/'+str(test_iters)+' in '+str(n)+' steps: '+str(pos))

                if pos[0] > 3:
                    import pickle as pkl

                    pkl.dump(acts, open('acts.pkl', 'w'))
                    exit(0)

                n = 0
                acts = []
                ob = env.reset()


                # plt.imshow(im)
                x_poses.append(pos[0])
                pos = np.zeros(3)
                i += 1
                if args.visualize:

                    real_pos_chart = np.array(real_pos_chart)
                    xr, yr, zr = np.hsplit(real_pos_chart, 3)
                    plt.plot(xr, yr)

                    max_lim = 1.1*max(np.max(xr), np.max(yr))
                    min_lim = 1.1*min(np.min(xr), np.min(yr))
                    plt.xlim(min_lim, max_lim)
                    plt.ylim(min_lim, max_lim)
                    plt.show()
                    plt.clf()
                    real_pos_chart = []

        x_poses = np.array(x_poses)
        print('Mean: '+str(np.mean(x_poses)))
        print('Median: '+str(np.median(x_poses)))
        print('Stdev: '+str(np.std(x_poses)))


if __name__ == '__main__':
    main()
