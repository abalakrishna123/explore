import argparse
import scipy.io as sio
import os
import pickle 
import numpy as np

parser = argparse.ArgumentParser(description='Analyze Results')
parser.add_argument('--model_results_path', default='output/LearnedOptimizationEnv-run239', type=str, help='path to model results')
args = parser.parse_args()

baseline_results = pickle.load( open('baseline_results.p', 'rb'))
model_results_loss = sio.loadmat(os.path.join(args.model_results_path, 'episode_loss_test.mat'))
model_results_steps = sio.loadmat(os.path.join(args.model_results_path, 'episode_steps_test.mat'))

print("Mean test loss: ", np.mean(model_results_loss["loss"]) )
print("Mean test steps: ", np.mean(model_results_steps["steps"]) )
print("Baseline results")
print(baseline_results)
