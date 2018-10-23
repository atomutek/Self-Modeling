import tensorflow as tf
import numpy as np
import time, datetime, pickle
from misc import losses
from misc.models import fk_learner
import matplotlib.pyplot as plt
from test_planners.control_widowx import calc_end_effector_pos

def batch(data, batch_size):
    batches = []
    while len(data) >= batch_size:
        batches.append(data[:batch_size])
        data = data[batch_size:]
    return batches

if __name__ == '__main__':
    # load = 'models/2018-10-01-20:58:39.ckpt'
    load = None

    from envs.widowx_arm import WidowxROS
    env = WidowxROS()
    gpu_options = None
    num_cpu = 1
    if gpu_options is None:
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu)
    else:
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            intra_op_parallelism_threads=num_cpu,
            gpu_options=gpu_options)

    episode_duration = -1
    nb_valid_episodes = 50
    episode_step = 0
    episode_reward = 0.0
    max_ep_rew = -10000
    valid = None

    angle_bounds = np.array([2.617, 1.571, 1.571, 1.745])
    pos_bounds = np.array([0.37, 0.37, 0.51])
    nb_epochs = 100
    with tf.Session(config=tf_config) as sess:
        sess_start = time.time()
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        # data_log = open('logs/'+datetime_str+'_log.txt', 'w+')

        verbose = False
        # generic data gathering

        X_data = []
        y_data = []

        x = tf.placeholder(dtype=tf.float32, shape=([None, 4]))
        real_y = tf.placeholder(dtype=tf.float32, shape=([None, 3]))
        y = fk_learner(x, 0.5)*pos_bounds
        loss = losses.loss_p(real_y, y)
        train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()


        if load is None:
            for d in range(1,10):
                if verbose:
                    print('Loading Data...')
                data_file = 'data/real_widowx_train_10hz_100K_default_processed.pkl'
                train_data = pickle.load(open(data_file, 'rb+'))[:d*1000]
                print(str(len(train_data))+' Datapoints: ')
                for episode in train_data:
                    X_data.append(episode[0][:-3])
                    y_data.append(episode[0][-3:])
                x_batches = batch(X_data, 32)
                y_batches = batch(y_data, 32)

                if verbose:
                    print('Training...')
                epoch_losses = []
                epochs = []
                for epoch in range(nb_epochs):
                    start = time.time()
                    losses = []
                    for i in range(len(x_batches)):
                        x_batch = np.array(x_batches[i])
                        y_batch = np.array(y_batches[i])
                        l, _ = sess.run([loss, train_step], feed_dict={x: x_batch, real_y: y_batch})
                        losses.append(l)

                    duration = time.time()-start

                    if verbose:
                        print('Epoch '+str(epoch)+' with loss '+str(np.mean(losses))+' Completed in '+str(duration)+'s')
                    epoch_losses.append(np.mean(losses))
                    epochs.append(epoch)
                    if epoch%10 == 9:
                        save_path = saver.save(sess, 'models/' + str(datetime_str) + '.ckpt')
                        if verbose:
                            print('Model Saved to: models/' + str(datetime_str) + '.ckpt')
                if verbose:
                    plt.plot(epochs, epoch_losses)
                    plt.show()

                print('Testing...')

                drifts = []
                reals = []
                preds = []
                for i in range(10000):
                    obs = env.reset()
                    angles = np.array([obs[:-3]])
                    pos = obs[-3:]
                    pred = sess.run([y], feed_dict={x: angles})[0][0]
                    if pos[2] > 0.01:
                        drift = np.linalg.norm(pos-pred)
                        reals.append(pos)
                        preds.append(pred)
                        drifts.append(drift)
                    else:
                        i-=1
                print(np.median(drifts))
                print(np.std(drifts))
                print('')


        else:
            saver.restore(sess, load)
            print('Model: ' + load + ' Restored')
            drifts = []
            reals = []
            preds = []
            from mpl_toolkits.mplot3d import Axes3D
            for i in range(10000):
                obs = env.reset()
                angles = np.array([obs[:-3]])
                pos = obs[-3:]
                pred = sess.run([y], feed_dict={x: angles})[0][0]
                if pos[2] > 0.01:
                    drift = np.linalg.norm(pos-pred)
                    reals.append(pos)
                    preds.append(pred)
                    drifts.append(drift)
                else:
                    i-=1
            print(np.median(drifts))
            print(np.std(drifts))
            print('')
            plt.clf()
            drift_max = np.max(drifts)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(len(drifts)):
                c = min(max(drifts[i]/drift_max, 0), 1)
                ax.plot(xs=[reals[i][0], preds[i][0]],
                        ys=[reals[i][1], preds[i][1]],
                        zs=[reals[i][2], preds[i][2]],
                        color=(0+c,1-c,0))
            plt.show()
            plt.clf()
            plt.hist(drifts, bins=100)
            plt.show()
