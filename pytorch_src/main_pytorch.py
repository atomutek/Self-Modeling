import argparse
import time
import gym
import torch


def run(**kwargs):
    # if kwargs['env'] == 'AntBulletEnv-v0':
    print('Environment \'AntBulletEnv-v0\' chosen')
    from pytorch_src.test_planners import test_plan_walker as testing
    import pybullet, pybullet_envs
    pybullet.connect(pybullet.DIRECT)
    # env = testing.AntWrapper(gym.make("AntBulletEnv-v0"))
    env = testing.RealerAntWrapper(gym.make("AntBulletEnv-v0"))
    # elif kwargs['env'] == 'AntBulletEnv-v0_active':


    if kwargs['arch'] == 'seq_encoding':
        print('Seq Encoding architecture chosen')
        from pytorch_src.env_learners.seq_env_learner import SeqEnvLearner
        env_learner = SeqEnvLearner(env)
    elif kwargs['arch'] == 'preco_gen':
        print('Seq Encoding architecture chosen')
        from pytorch_src.env_learners.preco_gen_env_learner import PrecoGenEnvLearner
        env_learner = PrecoGenEnvLearner(env)
    else:
        print('No valid architecture chosen')
        print('Defaulting to \'seq_encoding\'')
        from tensorflow_src.env_learners import SeqEnvLearner
        env_learner = SeqEnvLearner(env)

    # Disable logging for rank != 0 to avoid noise.
    start_time = time.time()
    testing.test(env=env, env_learner=env_learner, epochs=kwargs['nb_epochs'], train_episodes=kwargs['nb_train_episodes'], load=kwargs['load'],
                 test_episodes=kwargs['nb_test_episodes'], loop=kwargs['loop'], show_model=kwargs['show_model'])
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default='AntBulletEnv-v0')
    parser.add_argument('--arch', type=str, default='preco_gen')
    parser.add_argument('--loop', type=str, default='open')
    parser.add_argument('--nb-epochs', type=int, default=100)
    parser.add_argument('--nb-train-episodes', type=int, default=5)
    parser.add_argument('--nb-test-episodes', type=int, default=100)
    parser.add_argument('--show-model', dest='show_model', action='store_true')
    parser.add_argument('--load', type=str, default=None)
    # 'models/2018-09-16-11:32:53.ckpt' for precoVAE
    parser.set_defaults(show_model=False)
    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    # Run actual script.
    run(**args)
