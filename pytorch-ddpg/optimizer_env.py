import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch


# First logical test would be to make sure that LearnedOptimizationEnv
# works when just use SGD as action, then test DDPG, should be easy

# Generate roughly linear synthetic data with noise
def generate_linear_data(num_points, dim, noise_std):
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


def get_data(dataset, num_points, dim, noise_std):
    if dataset.lower() == 'mnist':
        return get_mnist_data(num_points, dim, noise_std)
    else:
        return generate_linear_data(num_points, dim, noise_std)


# Get total loss function for linear regression on dataset
def get_linear_loss(theta, data):
    loss = 0
    X = data[:, :-1]
    y = data[:, -1]
    for i in range(len(X)):
        loss += (np.dot(X[i], theta) - y[i]) ** 2
    return loss / float(len(X))


def get_loss(_, theta, data):
    return get_linear_loss(theta, data)


def get_stoch_linear_gradient(theta, data, batch_size):
    X = data[:, :-1]
    y = data[:, -1]
    grad = np.zeros(len(X[0]))
    indices = np.random.randint(len(X), size=batch_size)
    for i in indices:
        grad += 2 * (np.dot(X[i], theta) - y[i]) * X[i]
    return (grad / float(batch_size)).tolist()


# Return SGD update step
def SGD_linear_loss(theta, eta, data_i):
    X_i = data_i[:-1]
    y_i = data_i[-1]
    return -eta * (np.dot(X_i, theta) - y_i) * X_i


# Run SGD Optimization
def optimize_linear_SGD(data, eta, loss_thresh, max_epochs):
    theta = np.random.random(len(data[0]) - 1)
    epochs = 0
    loss = loss_thresh + 1

    while loss > loss_thresh and epochs < max_epochs:
        np.random.shuffle(data)
        for i in range(len(data)):
            theta = theta + SGD_linear_loss(theta, eta, data[i])
        loss = get_linear_loss(theta, data)
        if epochs % 100 == 0:
            print("Loss: " + str(loss))
        epochs += 1
    return [theta, epochs, loss]


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


# TODO: Add something about exploration policies, this should probably be in the DDPG part tbh since can just step
#       using this. Can add in exploration policies once basic DDPG works
# TODO: Currently operates off linear data with linear loss, will need to make this more general when optimizing
#       random functions

class LearnedOptimizationEnv:
    def __init__(self, num_points, grad_batch_size, dim, loss_thresh, max_epochs, losses_hist_length,
                 grads_hist_length, skip=0, dataset='simple'):
        # Get data + initialize optimization parameters
        self.dim = dim
        self.num_points = num_points
        self.data = get_data(dataset, self.num_points, self.dim, 0.05)
        self.loss_thresh = loss_thresh
        self.max_epochs = max_epochs
        self.grad_batch_size = grad_batch_size
        self.skip = skip

        # Initialize State
        self.losses_hist_length = losses_hist_length
        self.grads_hist_length = grads_hist_length
        self.theta = np.random.random(self.dim + 1)
        self.losses = Buffer(self.losses_hist_length, get_loss(dataset, self.theta, self.data))
        self.gradients = Buffer(self.grads_hist_length,
                                get_stoch_linear_gradient(self.theta, self.data, self.grad_batch_size))
        self.state = np.array(
            self.losses.get_list() + [grad_elem for grad in self.gradients.get_list() for grad_elem in grad])
        # self.state = np.array(self.losses.get_list())

    def reset(self):
        """Reset environment and get initial state"""
        self.p_coor = 1
        self.theta = np.random.random(self.dim + 1)
        self.losses = Buffer(self.losses_hist_length, get_linear_loss(self.theta, self.data))
        self.gradients = Buffer(self.grads_hist_length,
                                get_stoch_linear_gradient(self.theta, self.data, self.grad_batch_size))
        self.state = np.array(
            self.losses.get_list() + [grad_elem for grad in self.gradients.get_list() for grad_elem in grad])
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
        done = True if (self.losses.get_list()[-1] < self.loss_thresh) else False
        # --- Get new state ---
        # Update losses and gradients variables based on current loss and gradient
        self.losses.push(get_linear_loss(self.theta, self.data))
        self.gradients.push(get_stoch_linear_gradient(self.theta, self.data, self.grad_batch_size))
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


# Analyze performance of SGD when plugged into this framework, note that
# SGD makes no use of the state at all, but each action is a perturbation
# of the current parameters
if __name__ == "__main__":
    env = LearnedOptimizationEnv(1000, 50, 1, 0.005, 30, 100, 100)
    state = env.reset()
    data = env.get_data()
    episode_done = False

    i = 0
    rewards = []
    losses = []
    while episode_done is False:
        action = SGD_linear_loss(env.get_theta(), 0.01, data[np.random.randint(len(data))])
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

    # Only plot results if dim = 1
    plot_results(env.get_theta(), data)
    plt.plot(losses)
    plt.show()
    plt.plot(rewards)
    plt.show()
