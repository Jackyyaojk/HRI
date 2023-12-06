import numpy as np
import pickle
import sys
sys.path.append('../')

from utils import *




## 12 participants assembled a 4-leg chair with 3 test conditions
total_users = 12
total_legs = len(LEGS)
dem_nums = ['1', '2', '3', '4']
interfaces = ['implicit', 'gui', 'ar+haptic']

## convert chair leg states: joints space to cartesian space 
Panda = TrajectoryClient()
Legs_q = np.array(LEGS)
Legs_xyz = np.array(list(map(lambda q: Panda.joint2pose(q)[0].tolist(), Legs_q)))

## data structure: (number of users) X (leg number) 
dists = {
        "implicit": np.zeros((total_users, total_legs)),
        "gui": np.zeros((total_users, total_legs)),
        "ar+haptic":np.zeros((total_users, total_legs))
        }

## load user demonstrations
for idx in range(0,12):
    user = 'user' + str(idx+1)
    data_path = '../data/{}/dems'.format(user)
    for interface in interfaces:
        with open('{}/{}_dems.pkl'.format(data_path, interface), 'rb') as file:
            user_dems = pickle.load(file)
        user_pick_dems_q = np.array(
                            [user_dems["leg1_traj"][0],
                            user_dems["leg2_traj"][0],
                            user_dems["leg3_traj"][0],
                            user_dems["leg4_traj"][0]])
        user_pick_dems_xyz = np.array(list(map(lambda q: Panda.joint2pose(q)[0].tolist(), user_pick_dems_q)))
        dists[interface][idx, :] = np.linalg.norm(Legs_xyz - user_pick_dems_xyz, axis=1)

## count users correct predictions
dist_thresh = 0.12
correct_guess_implicit = np.count_nonzero(dists["implicit"] < dist_thresh, axis=1) / 4 * 100
correct_guess_gui = np.count_nonzero(dists["gui"] < dist_thresh, axis=1) / 4 * 100
correct_guess_arhaptic = np.count_nonzero(dists["ar+haptic"] < dist_thresh, axis=1) / 4 * 100
## means
correct_guess_implicit_mean = np.mean(correct_guess_implicit)
correct_guess_gui_mean = np.mean(correct_guess_gui)
correct_guess_arhaptic_mean = np.mean(correct_guess_arhaptic)
## standard deviations
correct_guess_implicit_var = np.std(correct_guess_implicit)
correct_guess_gui_var = np.std(correct_guess_gui)
correct_guess_arhaptic_var = np.std(correct_guess_arhaptic)


## plot
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
categories = ['Conditions']
conditions = ['implicit', 'gui', 'ar+haptic']
index = np.arange(len(categories))
bar_width = 0.01
ax.bar(index, correct_guess_implicit_mean, yerr=correct_guess_implicit_var, width=bar_width, label='implicit', color=[179/255, 179/255, 179/255])
ax.bar(index + bar_width, correct_guess_gui_mean, yerr=correct_guess_gui_var, width=bar_width, label='gui', color=[160/255, 212/255, 164/255])
ax.bar(index + 2*bar_width, correct_guess_arhaptic_mean, yerr=correct_guess_arhaptic_var, width=bar_width, label='ar+haptic', color=[34/255, 139/255, 69/255])
ax.set_ylabel('Correct Prediction [%]')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(categories)
ax.legend()
plt.savefig("prediction.png")