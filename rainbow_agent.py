#!/usr/bin/env python

"""
Train an agent on Sonic using an open source Rainbow DQN
implementation.
"""

import tensorflow as tf
import os
import time

from anyrl.algos import DQN
from anyrl.envs import BatchedGymEnv
from anyrl.envs.wrappers import BatchedFrameStack
from anyrl.models import rainbow_models
from anyrl.rollouts import BatchedPlayer, PrioritizedReplayBuffer, NStepPlayer
from anyrl.spaces import gym_space_vectorizer
import gym_remote.exceptions as gre

from sonic_util import AllowBacktracking, make_env

REWARD_HISTORY = 10

# TODO: Add DownsampleEnv, GrayscaleEnv

def main():
    """Run DQN until the environment throws an exception."""
    env = AllowBacktracking(make_env(stack=False, scale_rew=False))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)

    results_dir = os.path.join(os.getcwd(), 'results', time.strftime("%d-%m-%Y_%H-%M-%S"))
    summary_writer = tf.summary.FileWriter(results_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
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
