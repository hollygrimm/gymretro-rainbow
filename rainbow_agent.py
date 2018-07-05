#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf
import os
import time
import argparse

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_util import AllowBacktracking, make_env

REWARD_HISTORY = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', '-restore', action='store_true')
    args = parser.parse_args()

    """Run DQN until the environment throws an exception."""
    env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

    checkpoint_dir = os.path.join(os.getcwd(), 'results')
    results_dir = os.path.join(os.getcwd(), 'results', time.strftime("%d-%m-%Y_%H-%M-%S"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    summary_writer = tf.summary.FileWriter(results_dir)

    save_iters = 1024
    saver = tf.train.Saver()

    # TODO
    # env = wrappers.Monitor(env, results_dir, force=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        if args.restore:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
                saver.restore(sess, latest_checkpoint)
            else:
                print("Checkpoint not found")

        dqn = DQN(*rainbow_models(sess,
                env.action_space.n,
                gym_space_vectorizer(env.observation_space),
                min_val=-200,
                max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 3)
        optimize = dqn.optimize(learning_rate=1e-4)
        sess.run(tf.global_variables_initializer())

        reward_hist = []
        total_steps = 0

        def _handle_ep(steps, rew):
            nonlocal total_steps
            total_steps += steps
            reward_hist.append(rew)
            if len(reward_hist) == REWARD_HISTORY:
                print('%d steps: mean=%f' % (total_steps, sum(reward_hist) / len(reward_hist)))
                summary_meanreward = tf.Summary()
                summary_meanreward.value.add(tag='global/mean_reward', simple_value=sum(reward_hist) / len(reward_hist))
                summary_writer.add_summary(summary_meanreward, global_step=total_steps)
                reward_hist.clear()
            if total_steps % save_iters == 0:
                print('save model')
                saver.save(sess=sess, save_path=checkpoint_dir, global_step=total_steps)

        dqn.train(num_steps=2000000, # Make sure an exception arrives before we stop.
                player=player,
                replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                optimize_op=optimize,
                train_interval=1,
                target_interval=8192,
                batch_size=32,
                min_buffer_size=20000,
                handle_ep=_handle_ep)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
