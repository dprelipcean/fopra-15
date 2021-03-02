
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit


def read_data(filename):
	data = pd.read_csv(filename, skiprows=13, sep=';')
	voltage = data.iloc[:, 0]
	frequency = data.iloc[:, 1]
	intensity = data.iloc[:, 2]
	return np.asarray(voltage), np.asarray(frequency), np.asarray(intensity)


def gaussian(x, mu, sig, amplitude):
	return amplitude * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def trim_data(values, y_values, x_min, x_max):
	values_x_filtered = list()
	values_y_filtered = list()
	for index in range(len(values)):
		value = values[index]
		if x_min < value < x_max:
			values_x_filtered.append(value)
			values_y_filtered.append(y_values[index])
	return np.asarray(values_x_filtered), np.asarray(values_y_filtered)


def adjust_intensity(intensity_values, ground_level):
	values_refined = [ground_level - value for value in intensity_values]
	return values_refined


def compute_background_intensity(intensity_values, frequency_values, frequency_limit):
	background_intensity_values = list()
	for index in range(len(intensity_values)):
		frequency = frequency_values[index]
		if frequency < frequency_limit:
			background_intensity_values.append(intensity_values[index])

	average = sum(background_intensity_values) / len(background_intensity_values)
	return average


def plot_data(x_values, y_values):
	plt.plot(x_values, y_values, 'o-', label='Measured Data')

	background = compute_background_intensity(intensity_values=y_values, frequency_values=x_values, frequency_limit=2750)
	y_values = adjust_intensity(intensity_values=y_values, ground_level=background)

	values_x_filtered, values_y_filtered = trim_data(x_values, y_values, 2700, 2865)

	popt = curve_fit(gaussian, values_x_filtered, values_y_filtered, p0=[2825, 50, 4000000])

	mu, sigma, amplitude = popt[0]
	print(mu)
	y_values = gaussian(values_x_filtered, mu, sigma, amplitude)

	y_values = adjust_intensity(intensity_values=y_values, ground_level=background)

	plt.plot(values_x_filtered, y_values, '*:', label='Gaussian Fit')

	plt.legend()
	plt.grid()

	plt.xlabel("Frequency [MHz]")
	plt.ylabel("Intensity [counts]")

	plt.show()


def main():
	voltage, frequency, intensity = read_data("data/odmr2_118")

	plot_data(frequency, intensity)


if __name__ == "__main__":
	main()
