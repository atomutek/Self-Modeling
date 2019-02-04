import datetime
import math
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import pickle

def run_tests(test_episodes, env, data_log, env_learner, max_action, loop, verbose=True):
    # return run_tests_a_star(test_episodes, env, data_log, env_learner, max_action, loop, verbose=True)
    return run_tests_hill_climb(test_episodes, env, data_log, env_learner, max_action, loop, verbose=True)

def run_tests_hill_climb(test_episodes, env, data_log, env_learner, max_action, loop, verbose=True):
    episode_step = 0
    # last_d = env.d
    # d = last_d
    acts = []
    all_final_drifts = []
    all_final_lens = []
    all_final_real_ds = []
    all_final_pred_ds = []

    fail_final_drifts = []
    fail_final_lens = []
    fail_final_real_ds = []
    fail_final_pred_ds = []


    test = []
    failures = 0
    for i in range(test_episodes):
        start_time = time.time()
        done = False
        obs = env.reset()
        env_learner.reset(obs)
        init_d = np.linalg.norm(env.target - obs[-3:])
        data_log.write('Episode: ' + str(i) + ' Start\n')
        data_log.write('Target: ' + str(env.target) + '\n')
        data_log.write('Pred State: ' + str(obs) + '\n')
        data_log.write('Pred D: ' + str(init_d) + '\n')
        data_log.write('Real State: ' + str(obs) + '\n')
        data_log.write('Real D: ' + str(init_d) + '\n')
        data_log.write('Drift: ' + str(0.0) + '\n')
        start_pos = [obs[-3], obs[-2], obs[-1]]

        pred_ds = []
        real_ds = []
        drifts = []

        pred_ds.append(init_d)
        real_ds.append(init_d)
        drifts.append(0)

        while not done:
            action = find_next_move_test(env, env_learner, obs, max_action, episode_step, dof=4)
            # new_obs = env_learner.step(obs_in=obs, action_in=max_action * action, episode_step=episode_step, save=True)
            new_obs = env_learner.step(action_in=max_action * action)
            real_obs, r, real_done, _ = env.step(max_action * action)
            # print(real_obs/new_obs)
            d = np.linalg.norm(env.target - new_obs[-3:])
            real_d = np.linalg.norm(env.target - real_obs[-3:])
            test.append([obs, max_action * action, 0.0, new_obs, done, episode_step])
            acts.append(action)
            drift = np.linalg.norm(real_obs[-3:] - new_obs[-3:])
            episode_step += 1
            data_log.write('Action ' + str(episode_step) + ': ' + str(action) + '\n')
            data_log.write('Real Reward: ' + str(r) + '\n')
            data_log.write('Pred State: ' + str(new_obs) + '\n')
            data_log.write('Pred D: ' + str(d) + '\n')
            data_log.write('Real State: ' + str(real_obs) + '\n')
            data_log.write('Real D: ' + str(real_d) + '\n')
            data_log.write('Drift: ' + str(drift) + '\n')

            # print(drift)
            pred_ds.append(d)
            real_ds.append(real_d)
            drifts.append(drift)

            if loop == 'open':
                obs = new_obs
            elif loop == 'closed':
                obs = real_obs
            else:
                obs = new_obs

            done = episode_step > env.max_iter

            if d < 0.01:
                done = True

            if done:
                data_log.write('Episode: ' + str(episode_step) + ' done\n\n')
                if verbose:
                    print('Episode: ' + str(i) + ' in ' + str(time.time() - start_time) + ' seconds')
                    print(str(episode_step) + '\nPred D: ' + str(d) + '\nReal D: ' + str(real_d))
                    print('Drift: ' + str(drift))
                if d < 0.01:
                    all_final_drifts.append(drift)
                    all_final_lens.append(episode_step)
                    all_final_pred_ds.append(d)
                    all_final_real_ds.append(real_d)
                else:
                    failures += 1
                    fail_final_drifts.append(drift)
                    fail_final_lens.append(episode_step)
                    fail_final_pred_ds.append(d)
                    fail_final_real_ds.append(real_d)

                episode_step = 0
    if len(all_final_drifts) > 10:
        return failures, all_final_drifts, all_final_lens, all_final_pred_ds, all_final_real_ds
    else:
        return failures, fail_final_drifts, fail_final_lens, fail_final_pred_ds, fail_final_real_ds

def run_tests_a_star(test_episodes, env, data_log, env_learner, max_action, loop, verbose=True):
    episode_step = 0
    # last_d = env.d
    # d = last_d
    acts = []
    all_final_drifts = []
    all_final_lens = []
    all_final_real_ds = []
    all_final_pred_ds = []

    fail_final_drifts = []
    fail_final_lens = []
    fail_final_real_ds = []
    fail_final_pred_ds = []


    test = []
    failures = 0
    for i in range(test_episodes):
        start_time = time.time()
        done = False
        obs = env.reset()
        init_d = np.linalg.norm(env.target - obs[-3:])
        data_log.write('Episode: ' + str(i) + ' Start\n')
        data_log.write('Target: ' + str(env.target) + '\n')
        data_log.write('Pred State: ' + str(obs) + '\n')
        data_log.write('Pred D: ' + str(init_d) + '\n')
        data_log.write('Real State: ' + str(obs) + '\n')
        data_log.write('Real D: ' + str(init_d) + '\n')
        data_log.write('Drift: ' + str(0.0) + '\n')
        start_pos = [obs[-3], obs[-2], obs[-1]]

        pred_ds = []
        real_ds = []
        drifts = []

        pred_ds.append(init_d)
        real_ds.append(init_d)
        drifts.append(0)

        n = 2

        d = np.linalg.norm(env.target - obs[-3:])
        real_d = d
        drift = 0
        while d > 0.01:
            states, actions, buffers = a_star(env.action_space.shape[0], env, env_learner, obs, max_action, episode_step, is_test=False, step_size=math.pi/n)
            n*=2
        if actions is not None:
            for i in range(len(actions)):
                action = actions[i]
                # new_obs = env_learner.step(obs, max_action * action, episode_step, save=True)
                # new_obs = env_learner.step(obs_in=obs, action_in=max_action * action, episode_step=episode_step, save=True)
                new_obs = env_learner.step(max_action * action)
                real_obs, r, real_done, _ = env.step(max_action * action)

                d = np.linalg.norm(env.target - new_obs[-3:])
                real_d = np.linalg.norm(env.target - real_obs[-3:])
                test.append([obs, max_action * action, 0.0, new_obs, done, episode_step])
                acts.append(action)
                drift = np.linalg.norm(real_obs[-3:] - new_obs[-3:])
                episode_step += 1
                data_log.write('Action ' + str(episode_step) + ': ' + str(action) + '\n')
                data_log.write('Real Reward: ' + str(r) + '\n')
                data_log.write('Pred State: ' + str(new_obs) + '\n')
                data_log.write('Pred D: ' + str(d) + '\n')
                data_log.write('Real State: ' + str(real_obs) + '\n')
                data_log.write('Real D: ' + str(real_d) + '\n')
                data_log.write('Drift: ' + str(drift) + '\n')

                pred_ds.append(d)
                real_ds.append(real_d)
                drifts.append(drift)

                obs = new_obs
                # print(obs-states[i])
                # if loop == 'open':
                #     obs = new_obs
                # elif loop == 'closed':
                #     obs = real_obs
                # else:
                #     obs = new_obs

                # done = episode_step > env.max_iter
                #
                # if d < 0.1:
                #     done = True

                # if done:
            data_log.write('Episode: ' + str(episode_step) + ' done\n\n')
            if verbose:
                print('Episode: ' + str(i) + ' in ' + str(time.time() - start_time) + ' seconds')
                print(str(episode_step) + '\nPred D: ' + str(d) + '\nReal D: ' + str(real_d))
                print('Drift: ' + str(drift))
            if d < 0.01:
                all_final_drifts.append(drift)
                all_final_lens.append(episode_step)
                all_final_pred_ds.append(d)
                all_final_real_ds.append(real_d)
            else:
                failures += 1
                fail_final_drifts.append(drift)
                fail_final_lens.append(episode_step)
                fail_final_pred_ds.append(d)
                fail_final_real_ds.append(real_d)

            episode_step = 0

            print('n='+str(n)+' Found Path with final d='+str(d))
    if len(all_final_drifts) > 10:
        return failures, all_final_drifts, all_final_lens, all_final_pred_ds, all_final_real_ds
    else:
        return failures, fail_final_drifts, fail_final_lens, fail_final_pred_ds, fail_final_real_ds

def find_next_move_train(env, env_learner, obs, max_action, episode_step, dof, bottom=-1, top=1):
    # return find_next_move(env, env_learner, obs, max_action, episode_step, dof, bottom=bottom, top=top, is_test=False)
    return hill_climb(env.action_space.shape[0], env, env_learner, obs, max_action, episode_step, is_test=False, rand=False)

def find_next_move_test(env, env_learner, obs, max_action, episode_step, dof, bottom=-1, top=1):
    # return find_next_move(env, env_learner, obs, max_action, episode_step, dof, bottom=bottom, top=top, is_test=True)
    return hill_climb(env.action_space.shape[0], env, env_learner, obs, max_action, episode_step, is_test=True, rand=False)
    # return a_star(env.action_space.shape[0], env, env_learner, obs, max_action, episode_step, is_test=True)

def evaluate(action, env_learner, obs, max_action, env, episode_step, test=True):
    if test == False: new_obs = env.step(max_action * action, save=False)[0]
    # else: new_obs = env_learner.step(obs, max_action * action, episode_step, save=False)

    else: new_obs = env_learner.step(obs_in=obs, action_in=max_action * action, episode_step=episode_step, save=False)
    d = np.linalg.norm(env.target - new_obs[-3:])
    #if new_obs[-1] < 0.04: d -= 10000
    return -d

# taken from wikipedia
def hill_climb(act_dim, env, env_learner, obs, max_action, episode_step, is_test, rand=False):
    epsilon = 0.001
    if rand:
        current_point = np.random.uniform(-1, 1, act_dim)
    else:
        current_point = np.zeros(act_dim) # the 0 magnitude vector is common
    step_size = 100*epsilon*np.ones(act_dim) # a vector of all 1's is common
    acc = 1.2 # a value of 1.2 is common
    candidate = np.array([-acc, -1.0/acc, 0.0, 1.0/acc, acc])
    while True:
        before = evaluate(current_point, env_learner, obs, max_action, env, episode_step, test=is_test)
        for i in range(act_dim):
            best = -1
            best_score = -10000
            for j in range(5):
                last_pt = current_point[i]
                current_point[i] = current_point[i] + step_size[i] * candidate[j]
                current_point[i] = max(current_point[i], -0.9)
                current_point[i] = min(current_point[i], 0.9)

                temp = evaluate(current_point, env_learner, obs, max_action, env, episode_step, test=is_test)
                current_point[i] = last_pt
                if temp > best_score:
                    best_score = temp
                    best = j
            if candidate[best] == 0:
                step_size[i] = step_size[i] / acc
            else:
                current_point[i] = current_point[i] + step_size[i] * candidate[best]
                step_size[i] = step_size[i] * candidate[best] # accelerate
        if (evaluate(current_point, env_learner, obs, max_action, env, episode_step, test=is_test) - before) < epsilon:
            return current_point

def a_star(act_dim, env, env_learner, obs, max_action, episode_step, is_test, step_size=math.pi/16):
    from collections import defaultdict, deque
    import copy, sys
    reals = dict()
    closedSet = set()
    openSet = set()
    openScores = dict()
    start_hash = (0,0,0,0)
    openSet.add(start_hash)
    cameFrom = {}
    cameFromActions = {}
    buffers = {}
    gScore = {}
    gScore = defaultdict(lambda:1000, gScore)
    fScore = {}
    fScore = defaultdict(lambda:1000, fScore)


    pos_lim = np.ceil(np.array([2*math.pi]*7)[:act_dim]/step_size)
    possibilities = 1
    for entry in pos_lim:
        possibilities *= entry
    # print(possibilities)

    goal = env.target
    def reconstruct_path(last_hash):
        total_path = [reals[last_hash]]
        actions = []
        b = []
        while last_hash in cameFrom:
            actions.append(cameFromActions[last_hash])
            b.append(buffers[last_hash])
            last_hash = cameFrom[last_hash]
            total_path.append(reals[last_hash])
        total_path.reverse()
        actions.reverse()
        b.reverse()
        return total_path, actions, b
    def heuristic(tmp, goal):
        return np.linalg.norm(goal - tmp[act_dim:])
    def get_next():

        v=list(openScores.values())
        k=list(openScores.keys())
        choice = k[v.index(min(v))]
        return reals[choice], choice, buffers[choice]
    def get_neighbors(tmp, tmp_hash, buffer):
        n = []
        hashes = []
        b = []
        actions = [
            np.array([1, 0, 0, 0]),
            np.array([-1, 0, 0, 0]),
            np.array([0, 1, 0, 0]),
            np.array([0, -1, 0, 0]),
            np.array([0, 0, 1, 0]),
            np.array([0, 0, -1, 0]),
            np.array([0, 0, 0, 1]),
            np.array([0, 0, 0, -1]),
        ]
        for action in actions:
            tmp_buffer = copy.copy(buffer)
            tmp_state = current/env_learner.state_mul_const
            tmp_action = max_action * step_size * action/env_learner.act_mul_const
            tmp_buffer.append(np.array([np.concatenate([tmp_state, tmp_action])]).flatten())

            if not is_test: new_obs = env.step(max_action * step_size * action, save=False, obs_in=tmp)[0]
            else: new_obs = env_learner.step(obs_in=obs, action_in=max_action * action, episode_step=episode_step, save=False)
            n.append(new_obs)
            b.append(tmp_buffer)
            # TODO: Need better maxing here w.r.t. the bounds and the correct model
            # neg_lim = np.ceil((-env.bounds[:-3]-obs[:-3])/step_size)
            # pos_lim = np.ceil((env.bounds[:-3]-obs[:-3])/step_size)
            # bounds = env.bounds
            bounds = np.array([2*math.pi]*7)
            neg_lim = np.ceil(-bounds[:act_dim]/step_size)
            pos_lim = np.ceil(bounds[:act_dim]/step_size)
            # lim = 2*math.pi*np.ones(4)/step_size
            # else: new_obs = env_learner.step(tmp, max_action * step_size * action, 0, save=False, buff=tmp_buffer)
            hash = np.array(tmp_hash)+action
            # Ensuring greater than -360
            hash = np.maximum(neg_lim, hash)
            # Ensuring less than 360
            hash = np.minimum(pos_lim, hash)
            hashes.append(tuple(hash))
        actions = [action*step_size for action in actions]
        return n, hashes, actions, b

    reals[start_hash] = obs
    gScore[start_hash] = 0
    fScore[start_hash] = heuristic(obs, goal)
    openScores[start_hash] = heuristic(obs, goal)
    buffers[start_hash] = deque(env_learner.buff_init * env_learner.buff_len, maxlen=env_learner.buff_len)
    counter = 0
    min_dist = heuristic(obs, goal)
    min_hash = start_hash
    # print('Iteration: '+str(counter)+' With openSet size: '+str(len(openSet)))
    # print('Min Dist so Far: '+str(min_dist))
    while len(openSet) > 0:
        counter += 1
        current, current_hash, buffer = get_next()
        if counter%1024 == 1023:
            sys.stdout.write(str(counter+1)+'/'+str(possibilities)+' with min dist: '+str(min_dist)+'\r')
            sys.stdout.flush()
            # print('Iteration: '+str(counter+1)+' With openSet size: '+str(len(openSet)))
            # print('Min Dist so Far: '+str(min_dist))
        if heuristic(current, goal) < 2*math.sin(step_size):
        # if heuristic(current, goal) < 0.1:
        #     print('Path found with dist: '+str(heuristic(current, goal)))
            return reconstruct_path(current_hash)
        elif heuristic(current, goal) < min_dist:
            min_dist = heuristic(current, goal)
            min_hash = current_hash
        # openSet.remove(current)
        openSet.remove(current_hash)
        openScores.pop(current_hash)
        closedSet.add(current_hash)
        neighbors, neighbor_hashes, actions, tmp_buffers = get_neighbors(current, current_hash, buffer)
        for i in range(len(neighbors)):
        # for neighbor in neighbors:
            neighbor = neighbors[i]
            neighbor_hash = neighbor_hashes[i]
            if neighbor_hash in closedSet:
                continue
            tentative_gScore = gScore[current_hash] + np.linalg.norm(np.array(current) - np.array(neighbor))

            if neighbor_hash not in openSet:	# Discover a new node
                reals[neighbor_hash] = neighbor
                openSet.add(neighbor_hash)


            elif tentative_gScore >= gScore[neighbor_hash]:
                continue		# This is not a better path.

            cameFrom[neighbor_hash] = current_hash
            cameFromActions[neighbor_hash] = actions[i]
            gScore[neighbor_hash] = tentative_gScore
            # fScore[neighbor_hash] = heuristic(neighbor, goal)
            fScore[neighbor_hash] = gScore[neighbor_hash] + heuristic(neighbor, goal)
            openScores[neighbor_hash] = gScore[neighbor_hash] + heuristic(neighbor, goal)

            buffers[neighbor_hash] = tmp_buffers[i]

            # # This path is the best until now. Record it!
            # cameFrom[neighbor_hash] = current
            # cameFromActions[neighbor_hash] = actions[i]
            # gScore[neighbor_hash] = tentative_gScore
            # fScore[neighbor_hash] = gScore[neighbor_hash] + heuristic(neighbor, goal)
    print('No path found')
    print('Returning Closest Approximation...')
    path, actions, b = reconstruct_path(min_hash)
    print('Closest Approximation Found...')

    return path, actions, b

def test(env, env_learner, epochs=100, train_episodes=10, test_episodes=100, loop='open', show_model=False, load=None):
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    print('scaling actions by {} before executing in env'.format(max_action))
    print('Env Learner')

    print('Done Env Learner')
    print('Using agent with the following configuration:')
    try:
        saver = tf.train.Saver()
    except:
        saver=None
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    # gpu_options = None
    num_cpu = 1
    if gpu_options is None:
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu)
    else:
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_cpu,
            gpu_options=gpu_options)

    tf_config.gpu_options.allow_growth = True
    episode_duration = -1
    nb_valid_episodes = 50
    episode_step = 0
    episode_reward = 0.0
    max_ep_rew = -10000
    valid = None

    with tf.Session(config=tf_config) as sess:
        sess_start = time.time()
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        data_log = open('logs/'+datetime_str+'_log.txt', 'w+')

        if load is not None:
            saver.restore(sess, load)
            print('Model: ' + load + ' Restored')
            env_learner.initialize(sess, load=True)


        # generic data gathering

        if load is None or train_episodes > 0:
            print('Loading Data...')
            if load is None:
                env_learner.initialize(sess)

            # data_files = ['widowx_train_100K_1e-2.pkl',
            #               'widowx_train_100K_5e-2.pkl',
            #               'widowx_train_100K_1e-1.pkl',
            #               'widowx_train_100K_5e-1.pkl',
            #               'widowx_train_10hz_100K.pkl']
            # data_files = ['widowx_train_10hz_100K.pkl']
            # data_files = ['widowx_train_10hz_100K_deformed.pkl']
            data_files = ['real_widowx_train_10hz_100K_default_processed.pkl']
            env.max_iter = 100
            for data_file in data_files:
                train_data = pickle.load(open(data_file, 'rb+'))
                train = train_data[:(train_episodes*95)]
                valid = train_data[(train_episodes*95):]
                # Training self model
                print('Training on ' + str(len(train)) + ' data points')
                print('Validating on ' + str(len(valid)) + ' data points')
                # env_learner.get_loss(valid)
                env_learner.train(train, epochs, valid, saver=saver, save_str=datetime_str, verbose=True)
                print('Trained Self Model on data: '+str(data_file))

                # Progressive Learning
                # for i in range(len(train)/100,len(train),len(train)/100):
                #     batch_train = train[i-len(train)/100:i]
                #     print('Training on ' + str(len(batch_train)) + ' data points')
                #     env_learner.train(batch_train, epochs, valid, saver=saver, save_str=datetime_str)
                #
                #     data_log = open('logs/'+datetime_str+'_log.txt', 'w+')
                #     failures, all_final_drifts, all_final_lens, all_final_pred_ds, all_final_real_ds = \
                #         run_tests(test_episodes, env, data_log, env_learner, max_action, loop, verbose=False)
                #     print('Model '+str(i)+':')
                #     print('Percent Failed: ' + str(100.0 * float(failures) / float(test_episodes)) + '%')
                #     print('Mean Final Drift: '+str(np.mean(all_final_drifts)))
                #     print('Median Final Drift: '+str(np.median(all_final_drifts)))
                #     print('Stdev Final Drift: '+str(np.std(all_final_drifts)))
                #     print('')
                #     print('Trained Self Model on data: '+str(data_file))

        # Testing in this env
        print('Testing...')
        failures, all_final_drifts, all_final_lens, all_final_pred_ds, all_final_real_ds = run_tests(test_episodes, env, data_log, env_learner, max_action, loop, verbose=True)

        import statistics
        num_bins = 10
        print('\nModel: \'models/' + str(datetime_str) + '.ckpt\'')
        print('Percent Failed: '+str(100.0*float(failures)/float(test_episodes))+'%')

        print('Mean Final Drift: '+str(statistics.mean(all_final_drifts)))
        print('Median Final Drift: '+str(statistics.median(all_final_drifts)))
        print('Stdev Final Drift: '+str(statistics.stdev(all_final_drifts)))

        print('Mean Episode Len: '+str(statistics.mean(all_final_lens)))
        print('Median Episode Len: '+str(statistics.median(all_final_lens)))
        print('Stdev Episode Len: '+str(statistics.stdev(all_final_lens)))

        print('Mean Final Pred D: '+str(statistics.mean(all_final_pred_ds)))
        print('Median Final Pred D: '+str(statistics.median(all_final_pred_ds)))
        print('Stdev Final Pred D: '+str(statistics.stdev(all_final_pred_ds)))

        print('Mean Final Real D: '+str(statistics.mean(all_final_real_ds)))
        print('Median Final Real D: '+str(statistics.median(all_final_real_ds)))
        print('Stdev Final Real D: '+str(statistics.stdev(all_final_real_ds)))

        print('\nCompleted In: '+str(time.time()-sess_start)+' s')

        # _, _, _ = plt.hist(all_final_drifts, num_bins, facecolor='blue', alpha=0.5)
        # plt.title('Final Drifts')
        # # plt.savefig(datetime_str+'_final_drift')
        # plt.clf()
        # _, _, _ = plt.hist(all_final_lens, num_bins, facecolor='blue', alpha=0.5)
        # plt.title('Episode Lengths')
        # # plt.savefig(datetime_str+'_final_lens')
        # plt.clf()
        # _, _, _ = plt.hist(all_final_pred_ds, num_bins, facecolor='blue', alpha=0.5)
        # plt.title('Final Predicted Distances')
        # # plt.savefig(datetime_str+'_final_pred_ds')
        # plt.clf()
        # _, _, _ = plt.hist(all_final_real_ds, num_bins, facecolor='blue', alpha=0.5)
        # plt.title('Final Real Distances')
        # # plt.savefig(datetime_str+'_final_real_ds')
        # plt.clf()


#     args = parse_args()
