import sys
import time
import numpy as np
import serial
import copy
import pickle
import time


## check arduino's serial name
comm_arduino = serial.Serial('/dev/ttyACM0', baudrate=9600)
# comm_arduino = serial.Serial(port='COM0', baudrate=9600)


def send_pressure_arduino(user_input):

	# uncertainties = _std / np.linalg.norm(_std)

	# high_uncertainty_threshold = 0.66
	# low_uncertainty_threshold = 0.33
	# p_string = []

	# print(f'uncertainty values: {uncertainties}\n')
	# # print(f'mean uncertainty: {mean_unc}\n')
    

	# for i in uncertainties:

	# 	if i > high_uncertainty_threshold:
	# 		p_string.append("0.0; 0.0; 9.0")
	# 	elif i < low_uncertainty_threshold:
	# 		p_string.append("0.0; 0.0; 0.0")
	# 	else: # medium uncertainty threshold
	# 		p_string.append("0.0; 0.0; 3.0")

	string = '<' + user_input + '>'
	comm_arduino.write(str.encode(string))
	# print(f'String: {p_string}')
	# return(p_string)

while True:
	# Type the following as the terminal input:
	# LOW PRESSURE: 0.0;0.0;0.0 
	# HIGH PRESSURE: 0.0;0.0;9.0
	user_input = input('Pressure: ')  

	send_pressure_arduino(user_input)	

# print(f'\nStd devs: {_std}\n')

# string = send_pressure_arduino(_std)

# # Here, depending on which waypoint we are close to, 
# #  we select the strings in position 0, 1, or 2.
# comm_arduino.write(str.encode(string))
