import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')

from main import loadPolicy, forwardPolicy
from utils import *


## 12 participants assembled a 4-leg chair with 3 test conditions
total_users = 12
total_legs = len(LEGS)

## convert chair leg states: joints space to cartesian space 
Panda = TrajectoryClient()
Legs_q = np.array(LEGS)
Legs_xyz = np.array(list(map(lambda q: Panda.joint2pose(q)[0].tolist(), Legs_q)))

## data structure:  (leg number)
initial_errors = {
				"implicit": np.zeros((total_legs)),
				"gui": np.zeros((total_legs)),
				"ar+haptic":np.zeros((total_legs))
				}

## load pre-trained models with expert dems
pre_trained_models_path = '../data/pre-training/models'
pre_trained_policies_implicit = loadPolicy(num_models, pre_trained_models_path, 'implicit')
pre_trained_policies_gui = loadPolicy(num_models, pre_trained_models_path, 'gui')
pre_trained_policies_arhaptic = loadPolicy(num_models, pre_trained_models_path, 'ar+haptic')

## compute error: pre-trained models
for leg_idx, (leg_q, hole_q) in enumerate(zip(LEGS, HOLES)):
		
		initial_mean_waypoints_implicit, initial_std_wapoints_implicit = forwardPolicy(pre_trained_policies_implicit, leg_q, hole_q)
		initial_mean_waypoints_gui, initial_std_wapoints_gui = forwardPolicy(pre_trained_policies_gui, leg_q, hole_q)
		initial_mean_waypoints_arhaptic, initial_std_wapoints_arhaptic = forwardPolicy(pre_trained_policies_arhaptic, leg_q, hole_q)

		initial_xyz_implicit = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=initial_mean_waypoints_implicit)
		initial_xyz_gui = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=initial_mean_waypoints_gui)
		initial_xyz_arhaptic = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=initial_mean_waypoints_arhaptic)

		initial_error_implicit = np.linalg.norm(Legs_xyz[leg_idx] - initial_xyz_implicit, axis=1)
		initial_error_gui = np.linalg.norm(Legs_xyz[leg_idx] - initial_xyz_gui, axis=1)
		initial_error_arhaptic = np.linalg.norm(Legs_xyz[leg_idx] - initial_xyz_arhaptic, axis=1)

		initial_errors["implicit"][leg_idx] = np.mean(initial_error_implicit[[0,-1]])
		initial_errors["gui"][leg_idx] = np.mean(initial_error_gui[[0,-1]])
		initial_errors["ar+haptic"][leg_idx] = np.mean(initial_error_arhaptic[[0,-1]])



## data structure:  (number of users) X (leg number)
errors = {
		"implicit": np.zeros((total_users, total_legs)),
		"gui": np.zeros((total_users, total_legs)),
		"ar+haptic":np.zeros((total_users, total_legs))
		}

## load models trained with user feedback
for user in range(total_users):

	models_path = '../data/{}/models'.format('user'+str(user+1))
	policies_implicit = loadPolicy(num_models, models_path, 'implicit')
	policies_gui = loadPolicy(num_models, models_path, 'gui')
	policies_arhaptic = loadPolicy(num_models, models_path, 'ar+haptic')

	## compute error: user trained models
	for leg_idx, (leg_q, hole_q) in enumerate(zip(LEGS, HOLES)):

		mean_waypoints_implicit, std_wapoints_implicit = forwardPolicy(policies_implicit, leg_q, hole_q)
		mean_waypoints_gui, std_wapoints_gui = forwardPolicy(policies_gui, leg_q, hole_q)
		mean_waypoints_arhaptic, std_wapoints_arhaptic = forwardPolicy(policies_arhaptic, leg_q, hole_q)

		xyz_implicit = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=mean_waypoints_implicit)
		xyz_gui = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=mean_waypoints_gui)
		xyz_arhaptic = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=mean_waypoints_arhaptic)

		error_implicit = np.linalg.norm(Legs_xyz[leg_idx] - xyz_implicit, axis=1)
		error_gui = np.linalg.norm(Legs_xyz[leg_idx] - xyz_gui, axis=1)
		error_arhaptic = np.linalg.norm(Legs_xyz[leg_idx] - xyz_arhaptic, axis=1)

		errors["implicit"][user, leg_idx] = np.mean(error_implicit[[0,-1]])
		errors["gui"][user, leg_idx] = np.mean(error_gui[[0,-1]])
		errors["ar+haptic"][user, leg_idx] = np.mean(error_arhaptic[[0,-1]])


## plot
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
categories = ['leg1', 'leg2', 'leg3', 'leg4', 'mean']
conditions = ['implicit', 'gui', 'ar+haptic']

## mean errors across all users
implicit_data = np.mean(errors["implicit"], axis=0)
gui_data = np.mean(errors["gui"], axis=0)
arhaptic_data = np.mean(errors["ar+haptic"], axis=0)
implicit_stacked_data = np.concatenate((implicit_data, np.mean(implicit_data).reshape(1,)))
gui_stacked_data = np.concatenate((gui_data, np.mean(gui_data).reshape(1,)))
arhaptic_stacked_data = np.concatenate((arhaptic_data, np.mean(arhaptic_data).reshape(1,)))

bar_width = 0.2
index = np.arange(len(categories))
ax.bar(index, implicit_stacked_data, bar_width, label='implicit', color=[179/255, 179/255, 179/255])
ax.bar(index + bar_width, gui_stacked_data, bar_width, label='gui', color=[160/255, 212/255, 164/255])
ax.bar(index + 2*bar_width, arhaptic_stacked_data, bar_width, label='ar+haptic', color=[34/255, 139/255, 69/255])
ax.set_ylabel('Error')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(categories)
ax.legend()
plt.savefig("error.png")









