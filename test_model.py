import numpy as np
import tensorflow as tf
import gym
from test_planners import test_plan_walker as testing
from test_planners import test_plan_walker_active as active_testing
from env_learners.preco_gen_env_learner import PreCoGenEnvLearner
import pybullet, pybullet_envs
pybullet.connect(pybullet.DIRECT)

nb_valid_episodes = 25
i = 0
episode_duration = -1
episode_step = 0
episode_reward = 0.
max_ep_rew = -1000
valid = []


# env = testing.AntWrapper(gym.make("AntBulletEnv-v0"))
env = active_testing.RealerAntWrapper(gym.make("AntBulletEnv-v0"))
max_action = env.action_space.high



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


# load = 'models/2018-12-11-18:38:18.ckpt'
# load = 'models/2018-12-11-20:22:44.ckpt'
# load = 'models/2018-12-11-23:22:43.ckpt'
# load = 'models/2018-12-14-02:45:00.ckpt' # 1 layer lstm batch 32

# load = 'models/2018-12-14-12:47:39.ckpt' # 1 layer lstm batch 512
# load = 'models/2018-12-14-12:47:14.ckpt' # 1 layer gru batch 512
# load = 'models/2018-12-14-16:17:23.ckpt' # 2 layer gru batch 512

# load = 'models/2018-12-15-02:52:27.ckpt'
# load = 'models/2018-12-15-23:55:22.ckpt'
# load = 'models/2018-12-16-00:44:21.ckpt' # gru 2 layer drop 0.5 tanh 8 comp
# load = 'models/2018-12-16-16:53:48.ckpt' # gru 2 layer drop 0.5 no tanh 8 comp
load = 'models/2018-12-18-12:38:01.ckpt' # very self consistent only latent shared all other layers seperate
load = 'models/2018-12-18-12:38:07.ckpt'
load = 'models/2018-12-18-18:19:42.ckpt'
load = 'models/2018-12-22-15:44:09.ckpt'

env_learner = PreCoGenEnvLearner(env)

try:
    saver = tf.train.Saver()
except:
    print('Failed to open Saver')
    saver=None
with tf.Session(config=tf_config) as sess:
    if load is not None:
        saver.restore(sess, load)
        print('Model: ' + load + ' Restored')
        env_learner.initialize(sess, load=True)
    else:
        env_learner.initialize(sess)

    obs = env.reset()
    env_learner.reset(obs)
    open2_obs = obs

    closed = []
    open2 = []
    open = []

    while i < nb_valid_episodes:
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        new_obs, r, done, info = env.step(max_action * action)
        if episode_duration > 0:
            done = (done or (episode_step >= episode_duration))
        valid.append([obs, max_action * action, r, new_obs, done, episode_step])
        episode_step += 1
        obs = new_obs

        closed_obs = env_learner.step(obs_in=obs, action_in=action, episode_step=episode_step, num=-1)
        open2_obs = env_learner.step(obs_in=open2_obs, action_in=action, episode_step=episode_step)
        open_obs = env_learner.step(action_in=action, episode_step=episode_step, num=-1)

        closed.append(closed_obs)
        open2.append(open2_obs)
        open.append(open_obs)

        episode_reward += r
        if done:
            obs = env.reset()
            env_learner.reset(obs)
            open2_obs = obs

            max_ep_rew = max(max_ep_rew, episode_reward)
            episode_reward = 0.0
            i += 1

    (single, seq, corr) = env_learner.get_loss(valid)
    print('Valid Single: ' + str(single))
    print('Valid Seq: ' + str(seq))
    print('Valid Corr: ' + str(corr))

    def step_diff(a, b):
        if len(b[0].shape) > 1:
            diffs = np.zeros(b[0].shape[0])
        else:
            diffs = 0
        for i in range(len(b)):
            diff = np.abs(b[i] - a[i][3])
            if len(b[0].shape) > 1:
                diffs += np.mean(diff, axis=1)
            else:
                diffs += np.mean(diff)
        diffs = diffs/len(b)
        # avg_diff = np.mean(diffs, 1)
        return diffs


    print('Closed Avg Diff: '+str(step_diff(valid, closed)))
    print('Open2 Avg Diff: '+str(step_diff(valid, open2)))
    print('Open Avg Diff: '+str(step_diff(valid, open)))

    print('Closed Final Diff: '+str(step_diff([valid[-1]], [closed[-1]])))
    print('Open2 Final Diff: '+str(step_diff([valid[-1]], [open2[-1]])))
    print('Open Final Diff: '+str(step_diff([valid[-1]], [open[-1]])))

    print('')
    print('Done')
