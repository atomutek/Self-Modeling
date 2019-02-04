import numpy as np
import gym
from tensorflow_src.test_planners import test_plan_walker_active as active_testing
from pytorch_src.env_learners.seq_env_learner import SeqEnvLearner
import pybullet, pybullet_envs
import torch

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



load = None
# load = 'models/2019-01-17-23:12:55'
# load = 'models/2019-01-17-23:16:36' # best dense 3 layer lstm lr 1e-5 latent 32
# load = 'models/2019-01-18-00:10:59'
# load = 'models/2019-01-18-08:33:19' # 6 layer latent 32
# load = 'models/2019-01-18-08:33:58'
# load = 'models/2019-01-18-08:35:27'
# load = 'models/2019-01-19-12:35:03'
# load = 'models/2019-01-19-11:29:48'
# load = 'models/2019-01-19-11:41:48'
# load = 'models/2019-01-19-17:53:05'
# load = 'models/2019-01-19-22:02:50'
load = 'models/2019-01-20-01:35:22' # 200 examples abs error

env_learner = SeqEnvLearner(env)

if load is not None:
    env_learner.model = torch.load(load)

saver = None

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

    # closed_obs = env_learner.step(obs_in=obs, action_in=action, episode_step=episode_step, num=-1)
    # open2_obs = env_learner.step(obs_in=open2_obs, action_in=action, episode_step=episode_step)
    # open_obs = env_learner.step(action_in=action, episode_step=episode_step, num=-1)

    # closed.append(closed_obs)
    # open2.append(open2_obs)
    # open.append(open_obs)

    episode_reward += r
    if done:
        S = []
        X = []
        real = []
        real_pos = np.zeros(3)
        pred_pos = np.zeros(3)
        real_pos_chart = []
        pred_pos_chart = []
        decoded = []
        for j in range(len(valid)):
            step = valid[j]
            real.append(step[3]/env_learner.state_mul_const)

            S.append(np.array([np.concatenate([step[0]/env_learner.state_mul_const, step[1]/env_learner.act_mul_const])]))
            # if len(S) == env_learner.batch_size:


        # d = env_learner.sess.run(env_learner.decoded_out,
        #                          feed_dict={env_learner.state_seq: np.array(S)})
        env_learner.model.eval()
        d = env_learner.model.generate(torch.Tensor(S).cuda())[0].cpu().detach().numpy()

        gen, disc, mse = env_learner.get_loss(valid)
        print('Gen: '+str(gen))
        print('Disc: '+str(disc))
        print('MSE: '+str(mse))
        decoded.extend(d)
        decoded = np.array(decoded)
        decoded = np.squeeze(decoded)
        # decoded = decoded[:, -21:]
        real = np.array(real)
        # d = env_learner.sess.run(env_learner.decoded,
        #                          feed_dict={env_learner.state_seq: np.array(S)})
        # decoded.extend(d)
        assert len(decoded) == len(valid)
        for j in range(len(valid)):
            real_pos += (valid[j][3][0:3]/0.3)/60
            real_pos_chart.append(real_pos.copy())
            pred_pos += 7*(decoded[j][0:3]/0.3)/60
            pred_pos_chart.append(pred_pos.copy())

        print(np.mean(np.abs(decoded-real)))

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
        valid = []
        obs = env.reset()
        env_learner.reset(obs)
        open2_obs = obs
        max_ep_rew = max(max_ep_rew, episode_reward)
        episode_reward = 0.0
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
