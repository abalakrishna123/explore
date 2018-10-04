import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# First logical test would be to make sure that LearnedOptimizationEnv
# works when just use SGD as action, then test DDPG, should be easy

# Generate roughly linear synthetic data with noise
def generate_linear_data(num_points, dim, noise_std):
    theta = 10*np.ones(dim + 1)
    X = np.random.random((num_points, dim + 1))
    X[:, 0] = np.ones(num_points)
    y = np.array([np.dot(X, theta) + noise_std * np.random.randn(num_points)])
    return np.concatenate((X, y.T), axis=1)

# Get total loss function for linear regression on dataset
def get_linear_loss(theta, data):
    loss = 0
    X = data[:, :-1]
    y = data[:, -1]
    for i in range(len(X)):
        loss += ( np.dot(X[i], theta) - y[i] ) ** 2
    return loss/float(len(X))

def get_stoch_linear_gradient(theta, data, batch_size):
    grad = np.zeros(len(X[0]))
    X = data[:, :-1]
    y = data[:, -1]
    indices = np.random.randint(len(X), size=batch_size)
    for i in indices:
        grad += 2 * ( np.dot(X[i], theta) - y[i] ) * X[i]
    return (grad/float(batch_size)).tolist()

# Perform SGD
def SGD_linear_loss(theta, eta, data_i):
    X_i = data_i[:-1]
    y_i = data_i[-1]
    return theta - eta * ( np.dot(X_i, theta) - y_i ) * X_i

# Run SGD Optimization
def optimize_linear_SGD(data, eta, loss_thresh, max_epochs):
    theta = np.random.random(len(data[0]) - 1)
    epochs = 0
    loss = loss_thresh + 1

    while loss > loss_thresh and epochs < max_epochs:
        np.random.shuffle(data)
        for i in range(len(data)):
            theta = SGD_linear_loss(theta, eta, data[i])
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

    def get_list():
        return list(self.buffer)

class LearnedOptimizationEnv:
    def __init__(self, num_points, grad_batch_size, dim, init_exp_step_size, loss_thresh, max_epochs, num_exp_policies, losses_hist_length, grads_hist_length, max_epochs):
        # Get data + initialize optimization parameters
        self.data = generate_linear_data(num_points, dim, 0.05)
        self.loss_thresh = loss_thresh
        self.max_epochs = max_epochs
        self.grad_batch_size = grad_batch_size
        self.epoch_count = 0

        # Initialize exploration policy parameters
        self.exp_step_size = init_exp_step_size # step size for exploration policies
        self.p_coor = 1 # Probability of exploration policy sticking with its assigned coordinate
        self.num_exp_policies = num_exp_policies # Number of exploration policies

        # Initialize State
        self.losses_hist_length = losses_hist_length
        self.grads_hist_length = grads_hist_length
        self.theta = np.random.random(dim + 1)
        self.losses = Buffer(self.losses_hist_length, get_linear_loss(self.theta, self.data))
        self.gradients = Buffer(self.grads_hist_length, get_stoch_linear_gradient(self.theta, self.data, self.grad_batch_size))
        self.state = np.array(self.losses.get_list() + self.gradients.get_list())

    '''
        Reset environment and get initial state
    '''
    def reset():
        self.epoch_count = 0
        self.p_coor = 1
        self.theta = np.random.random(dim + 1)
        self.losses = Buffer(self.losses_hist_length, get_linear_loss(self.theta, self.data))
        self.gradients = Buffer(self.grads_hist_length, get_stoch_linear_gradient(self.theta, self.data, self.grad_batch_size))
        self.state = np.array(self.losses.get_list() + self.gradients.get_list())
        return self.state
    '''
        Step environment when an action is performed and return [next_state, reward, done]

        State representation is change in objective value at current location relative to ith
        most recent location and gradient of objective function evaluated at ith most recent location
        for i \in {2, ... H + 1}. If there are less than H losses/gradients, the most recent ones
        are repated

        Reward is the reduction in loss between the current step and the next step

        Done is set to True if max_epochs epcohs are reached or the loss < loss_thresh since
        these signify the end of the episode

        An action is a vector of the same dimension as theta
    '''
    def step(action):
        # --- Update theta based on action ---
        self.theta = self.theta + action
        # --- Determine whether the episode is done ---
        done = self.epoch_count > self.max_epochs or self.losses[-1] < self.loss_thresh
        # --- Get new state ---
        # Update losses and gradients variables based on current loss and gradient
        self.losses.push(get_linear_loss(self.theta, self.data))
        self.gradients.push(get_stoch_linear_gradient(self.theta, self.data, self.grad_batch_size))
        # Get new state
        next_state = np.array(self.losses.get_list() + self.gradients.get_list())
        # Get reward, will be positive if last loss was higher than new loss
        reward = self.losses.get_list()[-2] - self.losses.get_list()[-1]
        # Set current state to the next state
        self.state = next_state

        return [next_state, reward, done]

# data = generate_linear_data(1000, 50, 0.05)
# theta, epochs, loss = optimize_linear_SGD(data, 0.001, 0.001, 1000)
# print("Loss: " + str(loss))
# print("Epochs: " + str(epochs))

# Only plot results if dim = 1
# plot_results(theta, data)
