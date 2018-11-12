"""Number of different environments and scenarios.

Data and losses are organized into the following order:
- linear
- MNIST
- nonconvex easy
- nonconvex medium
- nonconvex hard
- CIFAR-10 (coming soon)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import pickle


#################
# FAKE DATASETS #
#################

def get_data(dataset, num_points, dim, noise_std):
    """Main data-loading abstraction"""
    if dataset.lower() == 'simple':
        return generate_linear_data(num_points, dim, noise_std)
    elif dataset.lower() == 'mnist':
        return get_mnist_data(num_points, dim, noise_std)
    elif dataset.lower() in ('nonconvex_easy', 'nonconvex_medium', 'nonconvex_hard'):
        return get_nonconvex_easy_data(num_points, dim, noise_std)


# First logical test would be to make sure that LearnedOptimizationEnv
# works when just use SGD as action, then test DDPG, should be easy

def generate_linear_data(num_points, dim, noise_std):
    """Generate roughly linear synthetic data with noise"""
    theta = 10 * np.ones(dim + 1)
    X = np.random.random((num_points, dim + 1))
    X[:, 0] = np.ones(num_points)
    y = np.array([np.dot(X, theta) + noise_std * np.random.randn(num_points)])
    return np.concatenate((X, y.T), axis=1)


def get_mnist_data(_, dim, __):
    """Ignore all of the arguments lulz."""
    assert dim == 28*28, "MNIST dimensions are 28x28"
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
           transform=transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
           ])),
        batch_size=1, shuffle=True).cpu()  # add cuda kwargs
    for sample in train_loader:
        yield sample.cpu()


def get_nonconvex_easy_data(num_points, dim, noise_std):
    return generate_linear_data(num_points, dim, noise_std)


########
# LOSS #
########


def get_loss(dataset, theta, data):
    if dataset.lower() == 'simple':
        return get_linear_loss(theta, data)
    elif dataset.lower() == 'mnist':
        return get_mnist_loss(theta, data)
    elif dataset.lower() == 'nonconvex_easy':
        return get_nonconvex_easy_loss(theta, data)
    elif dataset.lower() == 'nonconvex_medium':
        return get_nonconvex_medium_loss(theta, data)
    elif dataset.lower() == 'nonconvex_hard':
        return get_nonconvex_hard_loss(theta, data)


def get_linear_loss(theta, data):
    """Get total loss function for linear regression on dataset"""
    X = data[:, :-1]
    y = data[:, -1]
    return np.linalg.norm(X.dot(theta) - y) / float(len(X))


def get_mnist_loss(theta, data):
    return get_linear_loss(theta, data)


def get_nonconvex_easy_loss(theta, data):
    X = data[:, :-1]
    y = data[:, -1]
    return np.linalg.norm(np.minimum(X.dot(theta) - y, X.dot(theta) - 2*y)) / float(len(X))


def get_nonconvex_medium_loss(theta, data):
    X = data[:, :-1]
    y = data[:, -1]
    return np.linalg.norm(np.minimum(X.dot(theta) - y, X.dot(theta) - 2*y)) / float(len(X))


def get_nonconvex_hard_loss(theta, data):
    X = data[:, :-1]
    y = data[:, -1]
    return np.linalg.norm(np.sin(X.dot(theta) - y)) / float(len(X))


############
# GRADIENT #
############


def get_stoch_gradient(dataset, theta, data, batch_size, eta=1):
    X, Y = data[:, :-1], data[:, -1]

    # select batch
    indices = np.random.randint(len(X), size=batch_size)
    x, y = X[indices], Y[indices]

    if dataset.lower() == 'simple':
        gradient = get_stoch_linear_gradient(theta, x, y)
    elif dataset.lower() == 'mnist':
        gradient = get_stoch_linear_gradient(theta, x, y)
    elif dataset.lower() == 'nonconvex_easy':
        gradient = get_nonconvex_easy_gradient(theta, x, y)
    elif dataset.lower() == 'nonconvex_medium':
        gradient = get_nonconvex_medium_gradient(theta, x, y)
    elif dataset.lower() == 'nonconvex_hard':
        gradient = get_nonconvex_hard_gradient(theta, x, y)
    else:
        raise UserWarning('Invalid dataset: {}'.format(dataset))

    # normalize gradient
    gradient = -eta * gradient / float(batch_size)
    gradient = gradient.tolist()

    return gradient

def run_adam_optimizer(env, alpha=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):	
    state = env.reset()
    data = env.get_data()
    episode_done = False

    rewards = []
    losses = []
    m_t = np.zeros(len(env.theta))
    v_t = np.zeros(len(env.theta))
    t = 0
    while episode_done is False:
        t += 1
        g_t = np.array(get_stoch_gradient(env.dataset, env.theta, env.data, env.grad_batch_size))
        m_t = beta_1*m_t + (1-beta_1)*g_t
        v_t = beta_2 * v_t + (1-beta_2)*g_t * g_t 
        m_cap = m_t/(1 - (beta_1**t))
        v_cap = v_t/(1 - (beta_2**t))
        action = -(alpha * m_cap) / (np.sqrt(v_cap) + epsilon)
        next_state, reward, done, loss = env.step(action)
        episode_done = done 
        if t % len(data) == 0:
            print("Reward: " + str(reward))
            print("Done: " + str(done))
            print("Loss: " + str(loss))
        rewards.append(reward)
        losses.append(loss)

    print(t)
    print(episode_done)

    return t, np.sum(rewards), losses[-1]

    # # Only plot results if dim = 1
    # plot_results(env.get_theta(), data)
    # plt.plot(losses)
    # plt.show()
    # plt.plot(rewards)
    # plt.show()

def run_SGD(env, step_size = 0.01):
    state = env.reset()
    data = env.get_data()
    print(env.get_state_dim())
    episode_done = False

    i = 0
    rewards = []
    losses = []
    while episode_done is False:
        action = linear_gradient(env.get_theta(), step_size, data[np.random.randint(len(data))])
        next_state, reward, done, loss = env.step(action)
        episode_done = done

        if i % len(data) == 0:
            print("Reward: " + str(reward))
            print("Done: " + str(done))
            print("Loss: " + str(loss))
            # print("losses: ")
            # print(env.losses.get_list())
        i += 1
        rewards.append(reward)
        losses.append(loss)

    print(i)
    print(episode_done)
    return i, np.sum(rewards), losses[-1]

    # Only plot results if dim = 1
    # plot_results(env.get_theta(), data)
    # plt.plot(losses)
    # plt.show()
    # plt.plot(rewards)
    # plt.show()

def get_stoch_linear_gradient(theta, x, y):
    return 2 * x.T.dot(x.dot(theta) - y)


def get_nonconvex_easy_gradient(theta, x, y):
    return get_stoch_linear_gradient(theta, x, y)


def get_nonconvex_medium_gradient(theta, x, y):
    indicator = np.linalg.norm(x.dot(theta) - y) < np.linalg.norm(x.dot(theta) - 2 * y)
    gradient1 = get_stoch_linear_gradient(theta, x, y)
    gradient2 = get_stoch_linear_gradient(theta, x, 2*y)
    return gradient1 * indicator + gradient2 * (1 - indicator)


def get_nonconvex_hard_gradient(theta, x, y):
    pass


# Some other gradients?

def linear_batch_gradient(theta, eta, data):
    X = data[:, :-1]
    Y = data[:, -1]
    grad = 2 * X.T.dot(X.dot(theta) - Y)
    return -eta * grad / np.linalg.norm(grad)


# Perform SGD
def linear_gradient(theta, eta, data_i):
    X_i = data_i[:-1]
    y_i = data_i[-1]
    grad = (np.dot(X_i, theta) - y_i) * X_i
    return -eta * (grad/np.linalg.norm(grad))


# Run SGD Optimization
def optimize_linear_SGD(data, eta, loss_thresh, max_epochs):
    theta = np.random.random(len(data[0]) - 1)
    epochs = 0
    loss = loss_thresh + 1

    while loss > loss_thresh and epochs < max_epochs:
        np.random.shuffle(data)
        for i in range(len(data)):
            theta = theta + linear_gradient(theta, eta, data[i])
        loss = get_linear_loss(theta, data)
        if epochs % 100 == 0:
            print("Loss: " + str(loss))
        epochs += 1
    return [theta, epochs, loss]


#############
# UTILITIES #
#############


def plot_results(learned_theta, data):
    X = data[:, :-1]
    y = data[:, -1]
    preds_y = np.dot(X, learned_theta)
    X_plot = X[:, -1].flatten()
    plt.plot(X_plot, y, '*')
    plt.plot(X_plot, preds_y)
    plt.show()


class Buffer(object):
    def __init__(self, buff_size, init_value):
        self.buff_size = buff_size
        self.buffer = deque([init_value] * buff_size)

    def push(self, val):
        """Saves a new value in buffer."""
        self.buffer.popleft()
        self.buffer.append(val)

    def get_list(self):
        return list(self.buffer)


###############
# ENVIRONMENT #
###############


# TODO: Add something about exploration policies, this should probably be in the DDPG part tbh since can just step
#       using this. Can add in exploration policies once basic DDPG works
# TODO: Currently operates off linear data with linear loss, will need to make this more general when optimizing
#       random functions

class LearnedOptimizationEnv:
    def __init__(self, num_points, grad_batch_size, dim, loss_thresh, max_steps, losses_hist_length,
                 grads_hist_length, skip=0, dataset='simple'):
        # Get data + initialize optimization parameters
        self.dataset = dataset
        self.dim = dim
        self.num_points = num_points
        self.data = get_data(dataset, self.num_points, self.dim, 0.05)
        self.loss_thresh = loss_thresh
        self.max_steps = max_steps
        self.grad_batch_size = grad_batch_size
        self.skip = skip
        self.num_steps = 0

        # Initialize State
        self.losses_hist_length = losses_hist_length
        self.grads_hist_length = grads_hist_length
        self.theta = np.random.random(self.dim + 1)
        self.losses = Buffer(self.losses_hist_length, get_loss(dataset, self.theta, self.data))
        self.gradients = Buffer(self.grads_hist_length,
                                get_stoch_gradient(self.dataset, self.theta, self.data, self.grad_batch_size))
        self.state = np.array(
            self.losses.get_list() + [grad_elem for grad in self.gradients.get_list() for grad_elem in grad])
        # self.state = np.array(self.losses.get_list())

    def reset(self):
        """Reset environment and get initial state"""
        self.p_coor = 1
        self.theta = np.random.random(self.dim + 1)
        self.losses = Buffer(self.losses_hist_length, get_loss(self.dataset, self.theta, self.data))
        self.gradients = Buffer(self.grads_hist_length,
                                get_stoch_gradient(self.dataset, self.theta, self.data, self.grad_batch_size))
        self.state = np.array(
            self.losses.get_list() + [grad_elem for grad in self.gradients.get_list() for grad_elem in grad])
        self.num_steps = 0
        # self.state = np.array(self.losses.get_list())
        return self.state

    def step(self, action):
        """
        Step environment when an action is performed and return [next_state, reward, done]

        State representation is change in objective value at current location relative to ith
        most recent location and gradient of objective function evaluated at ith most recent location
        for i in {2, ... H + 1}. If there are less than H losses/gradients, the most recent ones
        are repated

        Reward is the percent reduction in loss between average of previous losses in loss buffer and new loss after executed action
        TODO: Think about using different size loss history for computing reward vs. storing state

        Done is set to True if max_epochs epcohs are reached or the loss < loss_thresh since
        these signify the end of the episode

        An action is a vector of the same dimension as theta
        """
        # --- Update theta based on action ---
        self.theta = self.theta + action
        # --- Determine whether the episode is done ---
        done = True if (self.losses.get_list()[-1] < self.loss_thresh or self.num_steps >= self.max_steps) else False
        # --- Get new state ---
        # Update losses and gradients variables based on current loss and gradient
        self.losses.push(get_loss(self.dataset, self.theta, self.data))
        self.gradients.push(get_stoch_gradient(self.dataset, self.theta, self.data, self.grad_batch_size))
        # Get new state
        next_state = np.array(
            self.losses.get_list()[::self.skip + 1] + \
            [grad_elem for grad in self.gradients.get_list()[::self.skip + 1] for grad_elem in grad])
        # next_state = np.array(self.losses.get_list())
        # Get reward, will be positive if average loss of first losses_hist_length - 1 values is more than new loss
        losses = self.losses.get_list()
        reward = np.mean(losses[:-1]) - losses[-1]
        loss = losses[-1]
        # Set current state to the next state
        self.state = next_state
        self.num_steps += 1

        return [next_state, reward, done, loss]

    def get_losses(self):
        return self.losses.get_list()

    def get_theta(self):
        return self.theta

    def get_gradients(self):
        return self.gradients.get_list()

    def get_data(self):
        return self.data

    def get_state_dim(self):
        return len(self.state)

    def get_action_dim(self):
        return len(self.theta)


# Here we can compare SGD, Adam against learned optimizer
if __name__ == "__main__":
    env = LearnedOptimizationEnv(1000, 50, 10, 1, 20000, 32, 32, 0, 'nonconvex_medium')
    run_adam_optimizer(env)

    # episode_rewards = []
    # episode_losses = []
    # episode_steps_list = []

    # env = LearnedOptimizationEnv(1000, 50, 10, 1, 20000, 32, 32, 0, 'nonconvex_medium')
    # num_episodes = 1000

    # for i in range(num_episodes):
    # 	print("Episode :" + str(i))
    # 	num_steps, ep_reward, ep_final_loss = run_SGD(env)
    # 	episode_rewards.append(ep_reward)
    # 	episodes_losses.append(ep_losses)
    # 	episode_steps_list.append(num_steps)

    # 	if i % 100 == 0:
	   #  	pickle.dump( {'episode_steps' : episode_steps_list, 'episode_rewards' : episode_rewards, 
	   #  		'episode_losses' : episode_losses}, open( "SGD_stats.p", "wb" ) )
