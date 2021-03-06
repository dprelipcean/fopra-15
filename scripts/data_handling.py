import numpy as np
import pandas as pd
from statistics import mean, stdev


def read_data(filename, data_format="voltage"):
	if data_format == "voltage":
		data = pd.read_csv(filename, skiprows=13, sep=';')
		voltage = np.asarray(data.iloc[:, 0])
		frequency = np.asarray(data.iloc[:, 1])
		intensity = np.asarray(data.iloc[:, 2])
		ret_vals = voltage, frequency, intensity
	elif data_format == "time":
		data = pd.read_csv(filename, skiprows=14, sep=';')
		time = np.asarray(data.iloc[:, 0])
		intensity = np.asarray(data.iloc[:, 1])
		ret_vals = time, intensity

	return ret_vals


def trim_data(values, y_values, x_min, x_max):
	values_x_filtered = list()
	values_y_filtered = list()
	for index in range(len(values)):
		value = values[index]
		if x_min < value < x_max:
			values_x_filtered.append(value)
			values_y_filtered.append(y_values[index])
	return np.asarray(values_x_filtered), np.asarray(values_y_filtered)


def adjust_intensity(intensity_values, ground_level, reverse=True):
	if reverse:
		values_refined = [ground_level - value for value in intensity_values]
	else:
		values_refined = [value - ground_level for value in intensity_values]
	return values_refined


def compute_background_intensity(intensity_values, frequency_values, frequency_limit, request_minimum=False):
	background_intensity_values = list()
	for index in range(len(intensity_values)):
		frequency = frequency_values[index]
		if frequency_limit[0] <= frequency <= frequency_limit[1]:
			background_intensity_values.append(intensity_values[index])

	background = mean(background_intensity_values)
	if request_minimum:
		background = min(background_intensity_values)
	print(background)
	return background


def average_data(x_values, y_values):
	x_values_average = list()
	y_values_average = list()
	y_values_average_std = list()

	y_buffer = list()
	for index in range(len(x_values)):
		x_value = x_values[index]
		y_value = y_values[index]

		if len(x_values_average) == 0:
			x_values_average.append(x_value)
			y_buffer.append(y_value)
		elif x_value == x_values_average[-1]:
			y_buffer.append(y_value)
		elif x_value != x_values_average[-1]:
			mean, std = compute_mean_and_std(y_buffer)
			y_values_average.append(mean)
			y_values_average_std.append(std)

			x_values_average.append(x_value)
			y_buffer = list()
			y_buffer.append(y_value)

		if index == len(x_values)-1:
			mean, std = compute_mean_and_std(y_buffer)
			y_values_average.append(mean)
			y_values_average_std.append(std)

	number_of_measurements_per_data_point = len(y_buffer)
	print(number_of_measurements_per_data_point, x_values_average, y_values_average, y_values_average_std)
	return number_of_measurements_per_data_point, np.asarray(x_values_average), np.asarray(y_values_average), np.asarray(y_values_average_std)


def compute_mean_and_std(values):
	average = mean(values)
	std = stdev(values, average)
	return average, std
