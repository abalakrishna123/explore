#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from optimizer_env import *
from evaluator import Evaluator
from ddpg import DDPG
from util import *
from scipy.io import savemat

def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):

    print(debug)
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    episode_loss = 0.
    observation = None
    episode_rewards = []
    episode_losses = []

    while step < num_iterations:

        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
        # agent pick action ...
        if step <= args.warmup:
            # action = agent.random_action()
            # For now when exploring, follow what SGD would do:
            action = agent.SGD_action(env.get_theta(), env.get_data())
        else:
            action = agent.select_action(observation)
        
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, loss = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            agent.update_policy()
        
        # [optional] evaluate
        # if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
        #     # policy = lambda x: agent.select_action(x, decay_epsilon=False)
        #     policy = lambda x: agent.SGD_action(env.get_theta(), env.get_data())
        #     validate_reward, validate_loss = evaluate(env, policy, debug=False, visualize=False)
        #     if debug: 
        #         prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
        #         prYellow('[Evaluate] Step_{:07d}: mean_loss:{}'.format(step, validate_loss))

        # [optional] save intermediate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        if step % len(env.get_data()) == 0:
            print("Reward: " + str(reward) )
            print("Loss: " + str(loss))
            print("Done: " + str(done) )

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        episode_loss += loss
        observation = deepcopy(observation2)

        if done: # end of episode

            episode_rewards.append( episode_reward/float(episode_steps)  )
            episode_losses.append(  episode_loss/float(episode_steps)  )

            if episode % 10 == 0:                 
                x = np.arange(len(episode_rewards))
                y = np.array(episode_rewards)
                plt.xlabel('Episode')
                plt.ylabel(' Average Reward')
                plt.plot(x, y)
                plt.savefig('{}/episode_reward'.format(output) +'.png')
                savemat('{}/episode_reward'.format(output) +'.mat', {'reward':episode_rewards})
                plt.clf()
                plt.cla()
                plt.close()

                x = np.arange(len(episode_losses))
                y = np.array(episode_losses)
                plt.xlabel('Episode')
                plt.ylabel(' Average Loss')
                plt.plot(x, y)
                plt.savefig('{}/episode_loss'.format(output) +'.png')
                savemat('{}/episode_loss'.format(output) +'.mat', {'loss':episode_losses})
                plt.clf()
                plt.cla()
                plt.close()

            if debug: 
                prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward/float(episode_steps),step))
                prGreen('#{}: episode_loss:{} steps:{}'.format(episode,episode_loss/float(episode_steps),step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode_loss = 0.
            episode += 1

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='LearnedOptimizationEnv', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=3000000, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=30000, type=int, help='')
    parser.add_argument('--validate_steps', default=30000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=25000000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    # env = NormalizedEnv(gym.make(args.env))
    env = LearnedOptimizationEnv(1000, 50, 1, 0.001, 30, 100, 100)

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.get_state_dim()
    nb_actions = env.get_action_dim()


    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))