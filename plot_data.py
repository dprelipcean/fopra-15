import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit


from scripts.math_functions import gaussian, exponential, exponentially_decaying_gaussian, exponentially_decaying_sinusoid, scale_values_to_unity
from scripts.data_handling import compute_background_intensity, adjust_intensity, trim_data, read_data, average_data


def plot_data(x_values, y_values, fit_function):
	plt.plot(x_values, y_values, 'o-', label='Measured Data')

	try:
		background = compute_background_intensity(intensity_values=y_values, frequency_values=x_values, frequency_limit=2750)
	except ZeroDivisionError:
		background = 4718883.78

	y_values = adjust_intensity(intensity_values=y_values, ground_level=background)

	values_x_filtered, values_y_filtered = trim_data(x_values, y_values, 2700, 2865)

	popt = curve_fit(fit_function, values_x_filtered, values_y_filtered, p0=[2825, 50, 4000000])

	function_parameters = popt[0]

	y_values = fit_function(values_x_filtered, *function_parameters)

	y_values = adjust_intensity(intensity_values=y_values, ground_level=background)

	plt.plot(values_x_filtered, y_values, '*:', label='Gaussian Fit')

	plt.legend()
	plt.grid()

	plt.xlabel("Frequency [MHz]")
	plt.ylabel("Intensity [counts]")

	plt.show()


def plot_data_rabi(x_values, y_values):
	number_of_measurements_per_data_point, x_values_average, y_values_average, y_values_average_std \
		= average_data(x_values, y_values)

	background = compute_background_intensity(intensity_values=y_values_average, frequency_values=x_values_average, frequency_limit=[125, 200], request_minimum=True)

	y_values_average_background_subtracted = adjust_intensity(intensity_values=y_values_average, ground_level=background, reverse=False)

	# print(x_values_average, y_values_average_background_subtracted)

	popt_measured_exponentical_decay = curve_fit(exponential, x_values_average, y_values_average_background_subtracted)

	decay_length, amplitude = popt_measured_exponentical_decay[0]
	print(decay_length, amplitude)
	y_values_fit = exponential(x_values_average, decay_length, amplitude)

	y_values_fit = adjust_intensity(intensity_values=y_values_fit, ground_level=-background, reverse=False)

	# Compute Peaks

	y_values_measured_minus_fit = y_values_average - y_values_fit

	# Disregard first point as statistical fluctuation
	x_values_average_peaks = x_values_average[1:]
	y_values_measured_minus_fit_peaks = y_values_measured_minus_fit[1:]

	popt_decaying_sinusoid = curve_fit(exponentially_decaying_sinusoid, x_values_average_peaks, y_values_measured_minus_fit_peaks, p0=[60, 200, 0.015])
	omega, decay_time, amplitude = popt_decaying_sinusoid[0]
	print(omega, decay_time, amplitude)

	# Add x value at 0
	# x_values_average_peaks = np.insert(x_values_average_peaks, 0, 0.05)
	x_values_average_peaks = np.linspace(0, 200, 200)

	# print(x_values_average_peaks)
	y_values_measured_minus_fit_peaks = exponentially_decaying_sinusoid(x_values_average_peaks, omega, decay_time, amplitude)
	# print(y_values_measured_minus_fit_peaks)

	normalization = scale_values_to_unity(y_values_measured_minus_fit_peaks)

	y_values_measured_minus_fit *= normalization
	y_values_measured_minus_fit_peaks *= normalization

	plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

	plt.plot(x_values, y_values, 'o',
			 label=f'Measured Data with {number_of_measurements_per_data_point} measurements per data point', alpha=0.2)
	plt.errorbar(x_values_average, y_values_average, yerr=y_values_average_std, marker="*", color="navy", capsize=5,
				 label='Average Measured Data with 1 Standard Deviation Errors')
	plt.plot(x_values_average, y_values_fit, '*', linestyle="dashed", color="darkred",
				label=f'Exponential Fit with decay time {decay_length:.2f} s')

	handles, labels = plt.gca().get_legend_handles_labels()
	print(labels)
	order = [0, 2, 1]
	plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

	plt.grid()

	plt.xlabel("Pulse length $T_p$ [s]")
	plt.ylabel("Intensity [counts]")

	plt.show()

	plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

	plt.plot(x_values_average, y_values_measured_minus_fit, "o", linestyle="dashed", color="navy",
				label=f"Measured data minus exponential Fit with decay time {decay_length:.2f}")
	plt.plot(x_values_average_peaks, y_values_measured_minus_fit_peaks, color="darkred",
				label=f"Sinusoidal function with time period {omega:.2f} s and decaying exponential {decay_time:.2f} s")

	plt.hlines(0, x_values_average_peaks[0], x_values_average_peaks[-1], linestyles=":")

	plt.vlines(omega/2, -1, 1, linestyles=":")
	plt.text(omega/2, 1, "$T_{\pi}$", fontsize=14, verticalalignment='top')
	plt.vlines(omega, -1, 1, linestyles=":")
	plt.text(omega, 1, "$T_{2\pi}$", fontsize=14, verticalalignment='top')
	plt.vlines(omega*3/2, -1, 1, linestyles=":")
	plt.text(omega*3/2, 1, "$T_{3\pi}$", fontsize=14, verticalalignment='top')
	plt.vlines(omega*2, -1, 1, linestyles=":")
	plt.text(omega*2, 1, "$T_{4\pi}$", fontsize=14, verticalalignment='top')

	plt.legend()
	# plt.grid()
	plt.ylabel("Spin z-component")
	plt.xlabel("Pulse length $T_p$ [s]")
	plt.show()


def main():
	# voltage, frequency, intensity = read_data("data/odmr_118")
	# plot_data(frequency, intensity, fit_function=gaussian)
	#
	# voltage, frequency, intensity = read_data("data/odmr2_118")
	# plot_data(frequency, intensity, fit_function=gaussian)
	#
	# voltage, frequency, intensity = read_data("data/odmr_zoom_118")
	# plot_data(frequency, intensity, fit_function=gaussian)

	time, intensity = read_data("data/Rabi_118", data_format="time")
	plot_data_rabi(time, intensity, fit_function=exponential)


if __name__ == "__main__":
	main()
