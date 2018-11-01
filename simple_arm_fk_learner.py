import tensorflow as tf
import numpy as np
import time, datetime, pickle
from misc.models import simple_fk_learner
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import os, sys

def batch(data, batch_size):
    batches = []
    while len(data) >= batch_size:
        batches.append(data[:batch_size])
        data = data[batch_size:]
    return batches


def train_and_save(x_batches, y_batches, nb_epochs):
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    verbose = True
    from misc import losses
    with tf.Session(config=tf_config) as sess:

        x = tf.placeholder(dtype=tf.float32, shape=([None, 2]))
        real_y = tf.placeholder(dtype=tf.float32, shape=([None, 2]))
        y = simple_fk_learner(x, 0.5)*2
        loss = losses.loss_p(real_y, y)
        train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
        sess.run(tf.global_variables_initializer())

        name_str = 'simple_'+str(datetime_str)
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

            # if verbose:
            #     print('Epoch '+str(epoch)+' with loss '+str(np.mean(losses))+' Completed in '+str(duration)+'s')
            epoch_losses.append(np.mean(losses))
            epochs.append(epoch)

        # print('Testing...')

        drifts = []
        reals = []
        preds = []
        for i in range(10000):
            obs = env.reset()
            angles = np.array([obs[:-2]])
            pos = obs[-2:]
            pred = sess.run([y], feed_dict={x: angles})[0][0]
            drift = np.linalg.norm(pos-pred)
            reals.append(pos)
            preds.append(pred)
            drifts.append(drift)

        print(name_str+' Finished '+str(nb_epochs)+' epochs: Median: '+str(np.median(drifts))+' Stdev: '+str(np.std(drifts))+'\n')
        minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["output"])
        tf.train.write_graph(minimal_graph, '.', str(nb_epochs)+'/'+name_str+'.proto', as_text=False)
        tf.train.write_graph(minimal_graph, '.', str(nb_epochs)+'/'+name_str+'.txt', as_text=True)

if __name__ == '__main__':
    load = None
    from envs.simple_arm import SimpleArm
    env = SimpleArm()
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

    tf_config.gpu_options.allow_growth = True
    episode_duration = -1
    nb_valid_episodes = 50
    episode_step = 0
    episode_reward = 0.0
    max_ep_rew = -10000
    valid = None


    train_episodes = 1000
    i = 0
    X_data = []
    y_data = []
    max_action = env.action_space.high
    while i < train_episodes:
        action = np.random.uniform(-1, 1, env.action_space.shape[0])
        new_obs, r, done, info = env.step(max_action * action)
        if episode_duration > 0:
            done = (done or (episode_step >= episode_duration))
        X_data.append(new_obs[:2])
        y_data.append(new_obs[-2:])
        episode_step += 1

        episode_reward += r
        if done:
            episode_step = 0.0
            obs = env.reset()
            max_ep_rew = max(max_ep_rew, episode_reward)
            episode_reward = 0.0
            i += 1
    x_batches = batch(X_data, 32)
    y_batches = batch(y_data, 32)

    for i in range(1, 5):
        exited = []
        active = []
        children = 0

        max_epochs = 20*i

        while len(exited)+len(active) < 100 or len(active) > 0:
            if len(active) < 5 and len(exited)+len(active) < 100:
                child_pid = os.fork()
                if child_pid == 0:
                    print 'in child'
                    train_and_save(x_batches, y_batches, nb_epochs=max_epochs)
                    os._exit(os.EX_OK)
                    print('Error: Child failed to terminate')
                else:
                    print('Forked child '+str(children)+' to PID: '+str(child_pid))
                    active.append(child_pid)
                    children += 1
                    time.sleep(1)
            else:
                exit_codes = [os.waitpid(p, os.WNOHANG) for p in active]
                for code in exit_codes:
                    if code[0] != 0:
                        active.remove(code[0])
                        exited.append(code[0])

