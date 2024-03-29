"""Number of different environments and scenarios.

Data and losses are organized into the following order:
- linear
- MNIST
- nonconvex easy
- nonconvex medium
- nonconvex hard
- CIFAR-10 (coming soon)
"""

import autograd.numpy as np
from autograd import elementwise_grad
import matplotlib.pyplot as plt
from collections import deque
import torch
import pickle
import math
import os


##################
# NONCONVEX ENVS #
##################


class Optimization:

    def __init__(self, f, optimum, xrange, yrange):
        self.f  = f
        self.fdx1 = elementwise_grad(self.f, argnum=0)
        self.fdx2 = elementwise_grad(self.f, argnum=1)

        xmin, xmax, xstep = xrange
        ymin, ymax, ystep = yrange
        self.x, self.y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
        self.z = self.f(self.x, self.y)
        self.fdx1_solved = self.fdx1(self.x, self.y)
        self.fdx2_solved = self.fdx2(self.x, self.y)

        self.min_x = np.array(optimum)  # global minimum
        self.min_y = self.f(*self.min_x)

    def get_loss(self, x1, x2):
        return (self.f(x1, x2) - self.min_y) ** 2

    def get_gradient(self, x1, x2):
        return np.array([self.fdx1(x1, x2), self.fdx2(x1, x2)])


class Beale(Optimization):

    def __init__(self):
        super(Beale, self).__init__(
            f=lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2,
            optimum=[3., .5],
            xrange=(-4.5, 4.5, .2),
            yrange=(-4.5, 4.5, .2),
        )

class GoldsteinPrice(Optimization):

    def __init__(self):
        super(GoldsteinPrice, self).__init__(
            f=lambda x, y: (1 + (x + y + 1) ** 2. * (19 - 14*x + 3*x**2. - 14*y + 6*x*y + 3*y**2.)) * \
            (30 + (2*x - 3*y)**2. * (18 - 32*x + 12*x**2. + 48*y - 36*x*y + 27*y**2.)),
            optimum=[0, -1.],
            xrange=(-2, 2, .1),
            yrange=(-2, 2, .1),
        )

class Booth(Optimization):

    def __init__(self):
        super(Booth, self).__init__(
            f=lambda x, y: (x + 2*y - 7) ** 2. + (2*x + y - 5) ** 2.,
            optimum=[1., 3.],
            xrange=(-10, 10, .5),
            yrange=(-10, 10, .5),
        )

class SchafferN2(Optimization):

    def __init__(self):
        super(SchafferN2, self).__init__(
            f=lambda x, y: 0.5 + ((np.sin(x**2. - y**2.)**2. - 0.5) / (1 + 0.001 * (x**2. + y**2.))**2.),
            optimum=[0., 0.],
            xrange=(-100, 100, 5),
            yrange=(-100, 100, 5),
        )

class Ackley(Optimization):

    def __init__(self):
        super(Ackley, self).__init__(
            f=lambda x, y: -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2. + y ** 2.))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20,
            optimum=[0., 0.],
            xrange=(-5, 5, 0.2),
            yrange=(-5, 5, 0.2)
        )


class OptimizationNDimensional:

    def __init__(self, f, optimum, xirange, n):
        """
        :param xirange: (start, end, step)
        :param n: n dimensional problem
        """
        self.f  = f
        self.fdxis = [elementwise_grad(self.f, argnum=i) for i in range(n)]

        self.min_xis = np.array(optimum)  # global minimum
        self.min_z = self.f(*self.min_xis)

    def get_loss(self, *xis):
        return (self.f(*xis) - self.min_z) ** 2

    def get_gradient(self, *xis):
        return np.array([f(*xis) for f in self.fdxis])


class Rastrigin(OptimizationNDimensional):

    def __init__(self, n=2, a=10):
        self.n = n
        self.a = a
        super(Rastrigin, self).__init__(
            f=self.f,
            optimum=[0.] * n,
            xirange=(-5.12, 5.12, 0.25),
            n=n
        )

    def f(self, *xis):
        assert len(xis) == self.n, len(xis)

        X = np.array(xis)
        return self.a * self.n + np.sum(X**2. - self.a * np.cos(2 * np.pi * X))


envs = {
    'beale': Beale(),
    'goldstein-price': GoldsteinPrice(),
    'booth': Booth(),
    'schaffer-n2': SchafferN2(),
    'ackley': Ackley(),
    'rastrigin': Rastrigin  # TODO: lulz, ugly hack
}

class UCB1():
  def __init__(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]

  def select_arm(self):
    n_arms = len(self.counts)
    for arm in range(n_arms):
      if self.counts[arm] == 0:
        return arm

    ucb_values = [0.0 for arm in range(n_arms)]
    total_counts = sum(self.counts)
    for arm in range(n_arms):
      bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
      ucb_values[arm] = self.values[arm] + bonus
    return ind_max(ucb_values)

  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]

    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value
    return

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
    elif dataset in envs:
        return get_custom_env_data(num_points, dim, noise_std, dataset)

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



def get_custom_env_data(num_points, dim, noise_std, dataset):
    """return nxd"""
    return []


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
    elif dataset in envs:
        return get_custom_env_loss(dataset, theta, data)
    else:
        raise NotImplementedError


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


def get_custom_env_loss(dataset, theta, data):
    # assert len(theta) == 2
    env = envs[dataset]
    return env.get_loss(*theta)


############
# GRADIENT #
############


def get_stoch_gradient(dataset, theta, data, batch_size, eta=1):
    if dataset not in envs.keys():
        X, Y = data[:, :-1], data[:, -1]
        # select batch
        indices = np.random.randint(len(X), size=batch_size)
        x, y = X[indices], Y[indices]
    else:
        x, y = None, None

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
    elif dataset in envs:
        gradient = get_custom_env_gradient(dataset, theta, x, y)
    else:
        raise UserWarning('Invalid dataset: {}'.format(dataset))

    # normalize gradient
    gradient = -eta * gradient / (np.linalg.norm(gradient) * float(batch_size))
    gradient = gradient.tolist()

    return gradient

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
    X = data[:, :-1]
    Y = data[:, -1]


def get_custom_env_gradient(dataset, theta, x, y):
    # assert len(theta) == 2
    env = envs[dataset]
    return env.get_gradient(*theta)


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


# For SGD with mom
def linear_gradient_mom(theta, eta, data_i):
    X_i = data_i[:-1]
    y_i = data_i[-1]
    grad = (np.dot(X_i, theta) - y_i) * X_i
    return eta * grad


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
# BASELINES #
#############

def run_rand_sample_action(env, step_size_choices):
    state = env.reset()
    data = env.get_data()
    # print(env.get_state_dim())
    episode_done = False

    i = 0
    rewards = []
    losses = []

    while episode_done is False:
        step_size_decision = np.random.choice(step_size_choices)
        action = np.array(get_stoch_gradient(env.dataset, env.get_theta(), data, batch_size=1, eta=step_size_decision))
        # action = linear_gradient(env.get_theta(), step_size_decision, data[np.random.randint(len(data))])
        next_state, reward, done, loss = env.step(action)
        episode_done = done

        if i % (len(data) or 100) == 0:
            print("Reward: " + str(reward), end="\r")
            print("Done: " + str(done), end="\r")
            print("Loss: " + str(loss), end="\r")
            # print("losses: ")
            # print(env.losses.get_list())
        i += 1
        rewards.append(reward)
        losses.append(loss)
    print("\t".join(map(str, env.theta)))

    return np.array([i, np.sum(rewards), losses[-1]])

def run_SGD(env, step_size):
    state = env.reset()
    data = env.get_data()
    # print(env.get_state_dim())
    episode_done = False

    i = 0
    rewards = []
    losses = []
    while episode_done is False:
        action = np.array(get_stoch_gradient(env.dataset, env.get_theta(), data, batch_size=1, eta=step_size))
        # action = linear_gradient(env.get_theta(), step_size, data[np.random.randint(len(data))])
        next_state, reward, done, loss = env.step(action)
        episode_done = done

        if i % (len(data) or 100) == 0:
            print("Reward: " + str(reward), end="\r")
            print("Done: " + str(done), end="\r")
            print("Loss: " + str(loss), end="\r")
            # print("losses: ")
            # print(env.losses.get_list())
        i += 1
        rewards.append(reward)
        losses.append(loss)

    return np.array([i, np.sum(rewards), losses[-1]])

    # Only plot results if dim = 1
    # plot_results(env.get_theta(), data)
    # plt.plot(losses)
    # plt.show()
    # plt.plot(rewards)
    # plt.show()

def run_SGD_mom(env, eta, gamma):
    state = env.reset()
    data = env.get_data()
    # print(env.get_state_dim())
    episode_done = False

    v = 0
    i = 0
    rewards = []
    losses = []
    while episode_done is False:
        scaled_grad = -np.array(get_stoch_gradient(env.dataset, env.get_theta(), data, batch_size=1, eta=eta))
        # scaled_grad = linear_gradient_mom(env.get_theta(), eta, data[np.random.randint(len(data))])
        v = gamma * v + scaled_grad
        action = -v
        next_state, reward, done, loss = env.step(action)
        episode_done = done

        if i % (len(data) or 100) == 0:
            print("Reward: " + str(reward))
            print("Done: " + str(done))
            print("Loss: " + str(loss))
            # print("losses: ")
            # print(env.losses.get_list())
        i += 1
        rewards.append(reward)
        losses.append(loss)

    return np.array([i, np.sum(rewards), losses[-1]])

    # Only plot results if dim = 1
    # plot_results(env.get_theta(), data)
    # plt.plot(losses)
    # plt.show()
    # plt.plot(rewards)
    # plt.show()


def run_FTL(env, step_size_choices):
    rewards_choices_total = np.zeros(len(step_size_choices))
    state = env.reset()
    data = env.get_data()
    episode_done = False

    i = 0
    rewards = []
    losses = []

    while episode_done is False:
        reward_choices = env.get_rewards(step_size_choices)
        rewards_choices_total += reward_choices
        if i > 50:
            step_size_decision = step_size_choices[np.argmax(rewards_choices_total)]
        else:
            step_size_decision = np.random.choice(step_size_choices)
        action = np.array(get_stoch_gradient(env.dataset, env.get_theta(), data, batch_size=1, eta=step_size_decision))
        # action = linear_gradient(env.get_theta(), step_size_decision, data[np.random.randint(len(data))])
        next_state, reward, done, loss = env.step(action)
        episode_done = done

        if i % (len(data) or 100) == 0:
            print("Reward: " + str(reward), end="\r")
            print("Done: " + str(done), end="\r")
            print("Loss: " + str(loss), end="\r")
            # print("losses: ")
            # print(env.losses.get_list())
        i += 1
        rewards.append(reward)
        losses.append(loss)

    os.makedirs('./tmp-ftl/', exist_ok=True)
    np.save('./tmp-ftl/theta.npy', env.get_theta())
    return np.array([i, np.sum(rewards), losses[-1]])

def m_weights(reward_choices, eta):
    weights = np.exp(eta*reward_choices)
    return weights/sum(weights)

def m_weights_sample(probs, step_size_choices):
    return step_size_choices[np.random.choice(len(probs), p=probs)]

def run_multiplicative_weights(env, step_size_choices):
    rewards_choices_total = np.zeros(len(step_size_choices))
    state = env.reset()
    data = env.get_data()
    # print(env.get_state_dim())
    episode_done = False

    i = 0
    rewards = []
    losses = []
    T = 10
    eta = np.sqrt(np.log(len(step_size_choices)))/T

    while episode_done is False:
        if i >= T:
            T = T*2
            eta = np.sqrt(np.log(len(step_size_choices)))/T

        probs = m_weights(rewards_choices_total, eta)
        reward_choices = env.get_rewards(step_size_choices)
        rewards_choices_total += reward_choices
        step_size_decision = m_weights_sample(probs, step_size_choices)
        action = np.array(get_stoch_gradient(env.dataset, env.get_theta(), data, batch_size=1, eta=step_size_decision))
        # action = linear_gradient(env.get_theta(), step_size_decision, data[np.random.randint(len(data))])
        next_state, reward, done, loss = env.step(action)
        episode_done = done

        if i % (len(data) or 100) == 0:
            print("Reward: " + str(reward), end="\r")
            print("Done: " + str(done), end="\r")
            print("Loss: " + str(loss), end="\r")
            # print("losses: ")
            # print(env.losses.get_list())
        i += 1
        rewards.append(reward)
        losses.append(loss)

    os.makedirs('./tmp-mw/', exist_ok=True)
    np.save('./tmp-mw/theta.npy', env.get_theta())
    return np.array([i, np.sum(rewards), losses[-1]])

def ind_max(x):
  m = max(x)
  return x.index(m)

def run_UCB(env, step_size_choices):
    state = env.reset()
    data = env.get_data()
    episode_done = False
    agent = UCB1(len(step_size_choices))

    i = 0
    rewards = []
    losses = []
    while episode_done is False:
        step_size_decision = agent.select_arm()
        temp_r = env.get_rewards(step_size_choices)[step_size_decision]
        action = np.array(get_stoch_gradient(env.dataset, env.get_theta(), data, batch_size=1, eta=step_size_decision))
        # action = linear_gradient(env.get_theta(), step_size_decision, data[np.random.randint(len(data))])
        next_state, reward, done, loss = env.step(action)
        agent.update(step_size_decision, reward)
        episode_done = done

        if i % (len(data) or 100) == 0:
            print("Reward: " + str(reward), end="\r")
            print("Done: " + str(done), end="\r")
            print("Loss: " + str(loss), end="\r")
        i += 1
        rewards.append(reward)
        losses.append(loss)

    os.makedirs('./tmp-ucb/', exist_ok=True)
    np.save('./tmp-ucb/values.npy', agent.values)
    np.save('./tmp-ucb/counts.npy', agent.counts)
    return np.array([i, np.sum(rewards), losses[-1]])
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
        if dataset.lower() in envs.keys():
            assert dim == 1 or dataset.lower() in ('rastrigin',)

        if dataset.lower() in ('rastrigin',):  # TODO: really bad hack
            envs[dataset.lower()] = envs[dataset.lower()](n=dim + 1)

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

    def reset(self, dataset=None):
        """Reset environment and get initial state"""
        self.p_coor = 1
        self.theta = np.random.random(self.dim + 1)
        self.losses = Buffer(self.losses_hist_length, get_loss(self.dataset, self.theta, self.data))
        self.gradients = Buffer(self.grads_hist_length,
                                get_stoch_gradient(self.dataset, self.theta, self.data, self.grad_batch_size))
        self.state = np.array(
            self.losses.get_list() + [grad_elem for grad in self.gradients.get_list() for grad_elem in grad])
        self.num_steps = 0
        if dataset is not None:
            self.dataset = dataset
        # self.state = np.array(self.losses.get_list())
        return self.state

    # Reward 1 for the best choice, reward 0 for everything else
    def get_rewards(self, actions):
        rewards = np.zeros(len(actions))
        max_reward_ind = 0
        max_reward = -np.inf
        data = self.get_data()
        for i in range(len(actions)):
            temp_theta = self.theta + get_stoch_gradient(self.dataset, self.get_theta(), data, batch_size=1, eta=1)
            # temp_theta = self.theta + linear_gradient(self.get_theta(), actions[i], data[np.random.randint(len(data))])
            losses = self.losses.get_list()
            r = np.mean(losses) - get_loss(self.dataset, temp_theta, self.data)
            if r > max_reward:
                max_reward_ind = i
                max_reward = r

        rewards[max_reward_ind] = 1
        return rewards

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
        orig_theta = self.theta
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
    env = LearnedOptimizationEnv(1000, 50, 2, 1, 20000, 32, 32, 0, 'nonconvex_medium')
    num_episodes = 1000
    for i in range(num_episodes):
        run_adam_optimizer(env)

    print("Random Learning Rate")
    run_rand_sample_action(env, np.array([0.001, 0.01, 0.1, 1, 10, 100]))
    print("Multiplicative Weights")
    run_multiplicative_weights(env, np.array([0.001, 0.01, 0.1, 1, 10, 100]))
    print("UCB")
    run_UCB(env, np.array([0.001, 0.01, 0.1, 1, 10, 100]))
    print("FTL")
    run_FTL(env, np.array([0.001, 0.01, 0.1, 1, 10, 100]))
    print("SGD")
    run_SGD_mom(env, 0.01, 0.7)

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
