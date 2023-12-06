import numpy as np
import pickle

from utils import *



dem_nums = ['1', '2', '3', '4']
interfaces = ['none', 'gui', 'ar+haptic']

total_users = 12
total_legs = len(LEGS)

Panda = TrajectoryClient()
Legs_q = np.array(LEGS)
Legs_xyz = np.array(list(map(lambda q: Panda.joint2pose(q)[0].tolist(), Legs_q)))

## data structure: (number of users) X (leg number) 
dists = {
        "none": np.zeros((total_users, total_legs)),
        "gui": np.zeros((total_users, total_legs)),
        "ar+haptic":np.zeros((total_users, total_legs))
        }

for idx in range(0,12):
    ## user dems
    user = 'user' + str(idx+1)
    data_path = 'data/{}/dems'.format(user)

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


## count how many times users misunderstood the robot
dist_thresh = 0.12
total_times = total_users * total_legs

correct_guess_none = np.count_nonzero(dists["none"] < dist_thresh, axis=1) / 4
correct_guess_gui = np.count_nonzero(dists["gui"] < dist_thresh, axis=1) / 4
correct_guess_arhaptic = np.count_nonzero(dists["ar+haptic"] < dist_thresh, axis=1) / 4

correct_guess_none_mean = np.mean(correct_guess_none)
correct_guess_gui_mean = np.mean(correct_guess_gui)
correct_guess_arhaptic_mean = np.mean(correct_guess_arhaptic)

correct_guess_none_var = np.std(correct_guess_none)
correct_guess_gui_var = np.std(correct_guess_gui)
correct_guess_arhaptic_var = np.std(correct_guess_arhaptic)

print(1 - correct_guess_none_mean, correct_guess_gui_mean, correct_guess_arhaptic_mean)
exit()

## plotting
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
categories = ['Correct Perceived Leg']
conditions = ['none', 'gui', 'ar+haptic']


# Create bar plots
index = np.arange(len(categories))
bar_width = 0.01
ax.bar(index, correct_guess_none_mean, yerr=correct_guess_none_var, width=bar_width, label='none')
ax.bar(index + bar_width, correct_guess_gui_mean, yerr=correct_guess_gui_var, width=bar_width, label='gui')
ax.bar(index + 2*bar_width, correct_guess_arhaptic_mean, yerr=correct_guess_arhaptic_var, width=bar_width, label='ar+haptic')

# Customize the plot
ax.set_ylabel('Percent Times')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
# plt.show()
plt.savefig("count.svg")