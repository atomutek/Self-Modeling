import numpy as np
import tensorflow as tf
import gym
from tensorflow_src.test_planners import test_plan_walker_active as active_testing
from tensorflow_src.env_learners.seq_env_learner import SeqEnvLearner
from tensorflow_src.env_learners.preco_gen_env_learner import PreCoGenEnvLearner
import pybullet, pybullet_envs

pybullet.connect(pybullet.DIRECT)

nb_valid_episodes = 25
i = 0
episode_duration = -1


# env = testing.AntWrapper(gym.make("AntBulletEnv-v0"))
env = active_testing.RealerAntWrapper(gym.make("AntBulletEnv-v0"))
max_action = env.action_space.high


# def step_diff(a, b):
#     if len(b[0].shape) > 1:
#         diffs = np.zeros(b[0].shape[0])
#     else:
#         diffs = 0
#     for i in range(len(b)):
#         diff = np.square(b[i] - a[i][3])
#         if len(b[0].shape) > 1:
#             diffs += np.mean(diff)
#         else:
#             diffs += np.mean(diff)
#     diffs = diffs/len(b)
#     # avg_diff = np.mean(diffs, 1)
#     return diffs
def step_diff(a, b):
    if len(b[0].shape) > 1:
        diffs = np.zeros(b[0].shape[0])
    else:
        diffs = 0
    for i in range(len(b)):
        diff = np.square(b[i] - a[i][3])
        diffs += np.mean(diff)
    diffs = diffs/len(b)
    # avg_diff = np.mean(diffs, 1)
    return diffs

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

# load = None
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
# load = 'models/2018-12-18-12:38:01.ckpt' # very self consistent only latent shared all other layers seperate
# load = 'models/2018-12-18-12:38:07.ckpt'
# load = 'models/2018-12-18-18:19:42.ckpt'
# load = 'models/2018-12-22-15:44:09.ckpt'
# load = 'models/2019-01-01-23:16:18.ckpt' # sequential and single and uncertainty loss

# load = 'models/2019-01-09-02:29:00.ckpt' # new seq no preco

# load = 'models/2019-01-12-00:28:01.ckpt' # vae pretrain

# load = 'models/2018-12-24-11:12:45.ckpt' # len 300 comp 1 all seperate big network



# load = 'models/2019-01-29-23:31:10.ckpt' # 1 layer fc with bn and relu and tanh out len 100
# load = 'models/2019-02-01-21:38:39.ckpt' # short decoder using outs instead of states
load = 'models/2019-02-02-18:29:18.ckpt' # 2 layer gru with outputs decoded

env_learner = PreCoGenEnvLearner(env)
# env_learner = SeqEnvLearner(env)

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
    real = []
    none = []
    episode_step = 0
    episode_reward = 0.
    max_ep_rew = -1000
    valid = []

    max_action = env.action_space.high
    while i < nb_valid_episodes:
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        new_obs, r, done, info = env.step(max_action * action)
        if episode_duration > 0:
            done = (done or (episode_step >= episode_duration))
        valid.append([obs, max_action * action, r, new_obs, done, episode_step])
        episode_step += 1

        closed_obs = env_learner.step(obs_in=obs, action_in=action, episode_step=episode_step, num=-1)
        # open2_obs = env_learner.step(obs_in=open2_obs, action_in=action, episode_step=episode_step)
        # open_obs = env_learner.step(action_in=action, episode_step=episode_step, num=-1)

        # closed.append(closed_obs)
        closed.append(obs)
        # open2.append(open2_obs)
        # open.append(open_obs)

        real.append(new_obs)
        none.append(obs)

        obs = new_obs
        episode_reward += r
        if done:
            done = False
            S = []
            X = []
            real = []
            real_pos = np.zeros(3)
            pred_pos = np.zeros(3)
            real_pos_chart = []
            pred_pos_chart = []
            decoded = []
            # for j in range(len(valid)):
            #     step = valid[j]
            #     real.append(step[3]/env_learner.state_mul_const)
            #     S.append(np.array([np.concatenate([step[0], step[1], step[3]])]))
                # if len(S) == env_learner.batch_size:
            # d = env_learner.sess.run(env_learner.decoded_out,
            #                          feed_dict={env_learner.state_seq: np.array(S)})
            # decoded.extend(d)
            # decoded = np.array(decoded)
            # real = np.array(real)
            # d = env_learner.sess.run(env_learner.decoded,
            #                          feed_dict={env_learner.state_seq: np.array(S)})
            # decoded.extend(d)
            # assert len(decoded) == len(valid)

            diffs = []

            for j in range(len(valid)):
                diffs.append(valid[j][3]-closed[j])
                real_pos += (valid[j][3][0:3]/0.3)/60
                real_pos_chart.append(real_pos.copy())
                pred_pos += (closed[j][0:3]/0.3)/60
                pred_pos_chart.append(pred_pos.copy())

            print(env_learner.get_loss(valid))
            print(np.mean(np.abs(np.array(diffs))))
            # print(np.mean(np.abs(decoded-real)))

            import matplotlib.pyplot as plt
            real_pos_chart = np.array(real_pos_chart)
            xr, yr, zr = np.hsplit(real_pos_chart, 3)
            plt.plot(xr, yr)
            pred_pos_chart = np.array(pred_pos_chart)
            xp, yp, zp = np.hsplit(pred_pos_chart, 3)
            plt.plot(xp, yp, color='black')

            max_lim = 1.1*max(np.max(xr), np.max(yr),np.max(xp),np.max(yp))
            min_lim = 1.1*min(np.min(xr), np.min(yr),np.min(xp),np.min(yp))
            plt.xlim(min_lim, max_lim)
            plt.ylim(min_lim, max_lim)
            plt.show()
            plt.clf()

            obs = env.reset()
            env_learner.reset(obs)
            open2_obs = obs

            closed = []
            open2 = []
            open = []
            real = []
            none = []
            episode_step = 0
            episode_reward = 0.
            max_ep_rew = -1000
            valid = []
            i += 1

    # (single, seq, corr) = env_learner.get_loss(valid)
    seq = env_learner.get_loss(valid)
    # print('Valid Single: ' + str(single))
    print('Valid Seq: ' + str(seq))
    # print('Valid Corr: ' + str(corr))

    # print('Scaled Valid Single: ' + str(5*single))
    print('Scaled Valid Seq: ' + str(5*seq))
    # print('Scaled Valid Corr: ' + str(5*corr))

    # print('Closed Avg Diff: '+str(step_diff(valid, closed)))
    # print('Open2 Avg Diff: '+str(step_diff(valid, open2)))
    # print('Open Avg Diff: '+str(step_diff(valid, open)))

    # print('Closed Final Diff: '+str(step_diff([valid[-1]], [closed[-1]])))
    # print('Open2 Final Diff: '+str(step_diff([valid[-1]], [open2[-1]])))
    # print('Open Final Diff: '+str(step_diff([valid[-1]], [open[-1]])))

    print('')
    print('Done')
