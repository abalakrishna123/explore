
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from util import *

class Evaluator(object):

    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.reward_results = np.array([]).reshape(num_episodes,0)
        self.loss_results = np.array([]).reshape(num_episodes,0)

    def __call__(self, env, policy, debug=False, visualize=False, save=True):

        self.is_training = False
        observation = None
        reward_result = []
        loss_result = []

        import pdb; pdb.set_trace()

        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation = env.reset()
            episode_steps = 0
            episode_reward = 0.
            episode_loss = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)

                observation, reward, done, loss = env.step(action)
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    done = True
                
                if visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_loss += loss
                episode_steps += 1

            episode_reward = episode_reward/float(episode_steps)
            episode_loss = episode_loss/float(episode_steps)
            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            reward_result.append(episode_reward)
            loss_result.append(episode_loss)

        reward_result = np.array(reward_result).reshape(-1,1)
        loss_result = np.array(loss_result).reshape(-1,1)
        self.reward_results = np.hstack([self.reward_results, reward_result])
        self.loss_results = np.hstack([self.loss_results, loss_result])

        if save:
            self.save_reward_results('{}/validate_reward'.format(self.save_path))
            self.save_loss_results('{}/validate_loss'.format(self.save_path))
        return [np.mean(reward_result), np.mean(loss_result)]

    def save_reward_results(self, fn):

        y = np.mean(self.reward_results, axis=0)
        error=np.std(self.reward_results, axis=0)
                    
        x = range(0,self.reward_results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.reward_results})

    def save_loss_results(self, fn):

        y = np.mean(self.loss_results, axis=0)
        error=np.std(self.loss_results, axis=0)
                    
        x = range(0,self.loss_results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Loss')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'loss':self.loss_results})