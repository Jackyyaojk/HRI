import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from main import loadPolicy, forwardPolicy
from utils import *



total_users = 12
total_legs = len(LEGS)
Panda = TrajectoryClient()
Legs_q = np.array(LEGS)
Holes_q = np.array(HOLES)
Legs_xyz = np.array(list(map(lambda q: Panda.joint2pose(q)[0].tolist(), Legs_q)))
Holes_xyz = np.array(list(map(lambda q: Panda.joint2pose(q)[0].tolist(), Holes_q)))
Target_xyz = np.concatenate((Legs_xyz, Holes_xyz), axis=1)


## computing uncertainty for pre-trained models
## data structure:   (leg number)
initial_model_uncertainties = {
					"none": np.zeros((total_legs)),
			  		"gui": np.zeros((total_legs)),
			   		"ar+haptic":np.zeros((total_legs))
					}

initial_dists = {
				"none": np.zeros((total_legs)),
				"gui": np.zeros((total_legs)),
				"ar+haptic":np.zeros((total_legs))
				}

pre_trained_models_path = 'data/pre-training/models'
pre_trained_policies_none = loadPolicy(num_models, pre_trained_models_path, 'none')
pre_trained_policies_gui = loadPolicy(num_models, pre_trained_models_path, 'gui')
pre_trained_policies_arhaptic = loadPolicy(num_models, pre_trained_models_path, 'ar+haptic')

for leg_idx, (leg_q, hole_q) in enumerate(zip(LEGS, HOLES)):
		
		initial_mean_waypoints_none, initial_std_wapoints_none = forwardPolicy(pre_trained_policies_none, leg_q, hole_q)
		initial_mean_waypoints_gui, initial_std_wapoints_gui = forwardPolicy(pre_trained_policies_gui, leg_q, hole_q)
		initial_mean_waypoints_arhaptic, initial_std_wapoints_arhaptic = forwardPolicy(pre_trained_policies_arhaptic, leg_q, hole_q)

		initial_xyz_none = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=initial_mean_waypoints_none)
		initial_xyz_gui = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=initial_mean_waypoints_gui)
		initial_xyz_arhaptic = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=initial_mean_waypoints_arhaptic)

		initial_dists["none"][leg_idx] = np.linalg.norm(Target_xyz[leg_idx] - initial_xyz_none[[0,-1]].flatten())
		initial_dists["gui"][leg_idx] = np.linalg.norm(Target_xyz[leg_idx] - initial_xyz_gui[[0,-1]].flatten())
		initial_dists["ar+haptic"][leg_idx] = np.linalg.norm(Target_xyz[leg_idx] - initial_xyz_arhaptic[[0,-1]].flatten())

		initial_model_uncertainties["none"][leg_idx] = np.mean(initial_std_wapoints_none[[0, -1]])
		initial_model_uncertainties["gui"][leg_idx] = np.mean(initial_std_wapoints_gui[[0, -1]])
		initial_model_uncertainties["ar+haptic"][leg_idx] = np.mean(initial_std_wapoints_arhaptic[[0, -1]])

## taking average across 4 legs
initial_mean_uncertainty_none = np.mean(initial_model_uncertainties["none"], axis=0)
initial_mean_uncertainty_gui = np.mean(initial_model_uncertainties["gui"], axis=0)
initial_mean_uncertainty_arhaptic = np.mean(initial_model_uncertainties["ar+haptic"], axis=0)

## data structure:    (number of users) X (leg number)
model_uncertainties = {
					"none": np.zeros((total_users, total_legs)),
			  		"gui": np.zeros((total_users, total_legs)),
			   		"ar+haptic":np.zeros((total_users, total_legs))
					}

dists = {
		"none": np.zeros((total_users, total_legs)),
		"gui": np.zeros((total_users, total_legs)),
		"ar+haptic":np.zeros((total_users, total_legs))
		}

for user in range(total_users):

	## paths to saved dems and models
	models_path = 'data/{}/models'.format('user'+str(user+1))

	## load trained policies
	policies_none = loadPolicy(num_models, models_path, 'none')
	policies_gui = loadPolicy(num_models, models_path, 'gui')
	policies_arhaptic = loadPolicy(num_models, models_path, 'ar+haptic')

	for leg_idx, (leg_q, hole_q) in enumerate(zip(LEGS, HOLES)):

		mean_waypoints_none, std_wapoints_none = forwardPolicy(policies_none, leg_q, hole_q)
		mean_waypoints_gui, std_wapoints_gui = forwardPolicy(policies_gui, leg_q, hole_q)
		mean_waypoints_arhaptic, std_wapoints_arhaptic = forwardPolicy(policies_arhaptic, leg_q, hole_q)

		xyz_none = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=mean_waypoints_none)
		xyz_gui = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=mean_waypoints_gui)
		xyz_arhaptic = np.apply_along_axis(lambda q: Panda.joint2pose(q)[0], axis=1, arr=mean_waypoints_arhaptic)

		dists["none"][user, leg_idx] = np.linalg.norm(Target_xyz[leg_idx] - xyz_none[[0,-1]].flatten())
		dists["gui"][user, leg_idx] = np.linalg.norm(Target_xyz[leg_idx] - xyz_gui[[0,-1]].flatten())
		dists["ar+haptic"][user, leg_idx] = np.linalg.norm(Target_xyz[leg_idx] - xyz_arhaptic[[0,-1]].flatten())

		model_uncertainties["none"][user, leg_idx] = np.mean(std_wapoints_none[[0, -1]])
		model_uncertainties["gui"][user, leg_idx] = np.mean(std_wapoints_gui[[0, -1]])
		model_uncertainties["ar+haptic"][user, leg_idx] = np.mean(std_wapoints_arhaptic[[0, -1]])


## taking average across 4 legs
mean_uncertainty_users_none = np.mean(model_uncertainties["none"], axis=1)
mean_uncertainty_users_gui = np.mean(model_uncertainties["gui"], axis=1)
mean_uncertainty_users_arhaptic = np.mean(model_uncertainties["ar+haptic"], axis=1)

## uncertainty difference after users feedback (avg all legs)
diff_uncertainty_none = np.mean((initial_mean_uncertainty_none - mean_uncertainty_users_none) / initial_mean_uncertainty_none)
diff_uncertainty_gui = np.mean((initial_mean_uncertainty_gui - mean_uncertainty_users_gui) / initial_mean_uncertainty_gui)
diff_uncertainty_arhaptic = np.mean((initial_mean_uncertainty_arhaptic - mean_uncertainty_users_arhaptic) / initial_mean_uncertainty_arhaptic)






## plotting distances to legs after training
fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
categories = ['leg1', 'leg2', 'leg3', 'leg4', 'mean']
conditions = ['none', 'gui', 'ar+haptic']

# none_data = np.mean((initial_dists["none"] - dists["none"]) / initial_dists["none"] * 100, axis=0)
# gui_data = np.mean((initial_dists["gui"] - dists["gui"]) / initial_dists["gui"] * 100, axis=0)
# arhaptic_data = np.mean((initial_dists["ar+haptic"] - dists["ar+haptic"]) / initial_dists["ar+haptic"] * 100, axis=0)

none_data_single = np.mean(dists["none"], axis=0)
gui_data_single = np.mean(dists["gui"], axis=0)
arhaptic_data_single = np.mean(dists["ar+haptic"], axis=0)

none_data_all = np.mean(dists["none"], axis=1)
gui_data_all = np.mean(dists["gui"], axis=1)
arhaptic_data_all = np.mean(dists["ar+haptic"], axis=1)

none_data_all_mean = np.mean(none_data_all).reshape(1,)
gui_data_all_mean = np.mean(gui_data_all).reshape(1,)
arhaptic_data_all_mean = np.mean(arhaptic_data_all).reshape(1,)

none_data_all_var = np.std(none_data_all)
gui_data_all_var = np.std(gui_data_all)
arhaptic_data_all_var = np.std(arhaptic_data_all)

none_stacked_mean = np.concatenate((none_data_single, none_data_all_mean))
gui_stacked_mean = np.concatenate((gui_data_single, gui_data_all_mean))
arhaptic_stacked_mean = np.concatenate((arhaptic_data_single, arhaptic_data_all_mean))

none_stacked_var = np.array([0, 0, 0, 0, none_data_all_var])
gui_stacked_var = np.array([0, 0, 0, 0, gui_data_all_var])
arhaptic_stacked_var = np.array([0, 0, 0, 0, arhaptic_data_all_var])


# Create bar plots
bar_width = 0.2
index = np.arange(len(categories))
ax.bar(index, none_stacked_mean, yerr=none_stacked_var, width=bar_width, label='none')
ax.bar(index + bar_width, gui_stacked_mean, yerr=gui_stacked_var, width=bar_width, label='gui')
ax.bar(index + 2*bar_width, arhaptic_stacked_mean, yerr=arhaptic_stacked_var, width=bar_width, label='ar+haptic')

# Customize the plot
ax.set_ylabel('Distance to Pick-n-Place Positions')
ax.set_title('Final distances')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(categories)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

# Show the plot
# plt.show()
plt.savefig("distances.svg")

## saving results to excel
export_dist_data = {"none": np.column_stack((dists["none"], np.mean(dists["none"], axis=1, keepdims=True))),
					"gui": np.column_stack((dists["gui"], np.mean(dists["gui"], axis=1, keepdims=True))),
					"ar+haptic": np.column_stack((dists["ar+haptic"], np.mean(dists["ar+haptic"], axis=1, keepdims=True)))}

excel_file_path = 'dists.xlsx'
with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    for category, array_data in export_dist_data.items():
        df = pd.DataFrame(array_data, columns=['leg1', 'leg2', 'leg3', 'leg4', 'mean'])
        df.to_excel(writer, sheet_name=category, index=False)
print(f'Data saved to {excel_file_path}')






# ## plotting uncertainty
# fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
# categories = ['initial', 'change']
# conditions = ['none', 'gui', 'ar+haptic']

# none_data = np.hstack((initial_mean_uncertainty_none,
# 					   diff_uncertainty_none))

# gui_data = np.hstack((initial_mean_uncertainty_gui,
# 					  diff_uncertainty_gui))

# arhaptic_data = np.stack((initial_mean_uncertainty_arhaptic,
# 						  diff_uncertainty_arhaptic))


# index = np.arange(len(categories))

# # Create bar plots
# bar_width = 0.2
# ax.bar(index, none_data, bar_width, label='none')
# ax.bar(index + bar_width, gui_data, bar_width, label='gui')
# ax.bar(index + 2*bar_width, arhaptic_data, bar_width, label='ar+haptic')

# # Customize the plot
# ax.set_ylabel('Robot Uncertainty')
# ax.set_xticks(index + bar_width)
# ax.set_xticklabels(categories)
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.tight_layout()

# # Show the plot
# plt.show()







