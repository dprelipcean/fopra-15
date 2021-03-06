import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import linregress
from scipy.optimize import curve_fit


from scripts.math_functions import gaussian, exponential, exponentially_decaying_gaussian, exponentially_decaying_cosine, scale_values_to_unity, exponentially_decaying_sine, linear_function
from scripts.data_handling import compute_background_intensity, adjust_intensity, trim_data, read_data, average_data


def plot_data_voltage_frequency_correlation(x_values, y_values):
	result = linregress(x_values, y_values)
	print(result.slope, result.intercept, result.rvalue)
	y_values_fit = linear_function(x_values, result.slope, result.intercept)

	plt.plot(x_values, y_values, 'o-', label='Measured Data', color="navy")
	plt.plot(x_values, y_values_fit, '*',
			 label=f'Linear Fit with\n'
				   f'y = {result.slope:.2f} $\cdot$ x [MHz/V] + {result.intercept:.2f} [MHz]\n'
				   f'and $R^2=${result.rvalue:.5f}', color="darkred")

	plt.grid()

	plt.xlabel("Voltage [V]")
	plt.ylabel("Frequency [MHz]")

	plt.legend()

	plt.show()


def plot_data(x_values, y_values, fit_function):
	plt.plot(x_values, y_values, 'o-', label='Measured Data', color="navy")

	try:
		background = compute_background_intensity(intensity_values=y_values, frequency_values=x_values, frequency_limit=[0, 2750])
	except Exception:
		background = 4718883.78

	y_values = adjust_intensity(intensity_values=y_values, ground_level=background)

	values_x_filtered, values_y_filtered = trim_data(x_values, y_values, 2700, 2865)

	popt = curve_fit(fit_function, values_x_filtered, values_y_filtered, p0=[2825, 50, 4000000])

	function_parameters = popt[0]

	y_values = fit_function(values_x_filtered, *function_parameters)

	y_values = adjust_intensity(intensity_values=y_values, ground_level=background)

	plt.plot(values_x_filtered, y_values, '*:', label='Gaussian Fit', color="darkred")

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

	popt_decaying_sinusoid = curve_fit(exponentially_decaying_cosine, x_values_average_peaks, y_values_measured_minus_fit_peaks, p0=[60, 200, 0.015])
	omega, decay_time, amplitude = popt_decaying_sinusoid[0]
	print(omega, decay_time, amplitude)

	# Add x value at 0
	# x_values_average_peaks = np.insert(x_values_average_peaks, 0, 0.05)
	x_values_average_peaks = np.linspace(0, 200, 200)

	# print(x_values_average_peaks)
	y_values_measured_minus_fit_peaks = exponentially_decaying_cosine(x_values_average_peaks, omega, decay_time, amplitude)
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
				label=f"Cosine function with time period {omega:.2f} s and decaying exponential {decay_time:.2f} s")

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


def plot_data_ramsey(x_values, y_values):
	number_of_measurements_per_data_point, x_values_average, y_values_average, y_values_average_std \
		= average_data(x_values, y_values)

	background = compute_background_intensity(intensity_values=y_values_average, frequency_values=x_values_average, frequency_limit=[125, 200], request_minimum=False)

	y_values_average_background_subtracted = adjust_intensity(intensity_values=y_values_average, ground_level=background, reverse=False)

	# Compute Peaks
	popt_decaying_sinusoid = curve_fit(exponentially_decaying_sine, x_values_average, y_values_average_background_subtracted, p0=[60, 200, 0.015, 10])
	omega, decay_time, amplitude, phase_shift = popt_decaying_sinusoid[0]
	print(omega, decay_time, amplitude, phase_shift)

	# Add x value at 0
	# x_values_average_peaks = np.insert(x_values_average_peaks, 0, 0.05)
	x_values_fit = np.linspace(0, 200, 200)

	y_values_fit = exponentially_decaying_sine(x_values_fit, omega, decay_time, amplitude, phase_shift)
	# y_values_fit = adjust_intensity(intensity_values=y_values_fit, ground_level=-background, reverse=False)

	normalization = scale_values_to_unity(y_values_fit, request_max=True)
	y_values_fit *= normalization

	fig, ax1 = plt.subplots(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
	ax2 = ax1.twinx()

	ax1.plot(x_values, y_values, 'o',
			 label=f'Measured Data with {number_of_measurements_per_data_point} measurements per data point', alpha=0.2)
	ax1.errorbar(x_values_average, y_values_average, yerr=y_values_average_std, marker="*", color="navy", capsize=5,
				 label='Average Measured Data with 1 Standard Deviation Errors')
	ax2.plot(x_values_fit, y_values_fit, '*', linestyle="dashed", color="darkred",
				label=f"Cosine function with time period {omega:.2f} s and decaying exponential {decay_time:.2f} s")

	handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
	handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
	handles = handles_ax1 + handles_ax2
	labels = labels_ax1 + labels_ax2
	order = [0, 1, 2]
	plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

	ax1.set_xlabel("Free evolution time $\\tau$ [s]")
	ax1.set_ylabel("Intensity [counts]", color="navy")
	# ax1.legend()

	ax2.set_ylim(-1.05, 1.05)
	ax2.set_ylabel("Spin z-component", color="darkred")
	# ax2.legend()

	ax2.hlines(0, x_values_fit[0], x_values_fit[-1], linestyles=":")
	#
	plt.vlines(phase_shift, -1, 1, linestyles=":")
	plt.text(phase_shift, -1, "Phase shift $\phi$", fontsize=14, verticalalignment='top')
	plt.vlines(phase_shift+omega/2, -1, 1, linestyles=":")
	plt.text(phase_shift+omega/2, -1, "$\\tau_{\pi}$", fontsize=14, verticalalignment='top')
	plt.vlines(phase_shift+omega, -1, 1, linestyles=":")
	plt.text(phase_shift+omega, -1, "$\\tau_{2\pi}$", fontsize=14, verticalalignment='top')
	plt.vlines(phase_shift+omega*3/2, -1, 1, linestyles=":")
	plt.text(phase_shift+omega*3/2, -1, "$\\tau_{3\pi}$", fontsize=14, verticalalignment='top')
	plt.vlines(phase_shift+omega*2, -1, 1, linestyles=":")
	plt.text(phase_shift+omega*2, -1, "$\\tau_{4\pi}$", fontsize=14, verticalalignment='top')

	plt.show()


def plot_data_echo(x_values, y_values):
	number_of_measurements_per_data_point, x_values_average, y_values_average, y_values_average_std \
		= average_data(x_values, y_values)

	background = compute_background_intensity(intensity_values=y_values_average, frequency_values=x_values_average, frequency_limit=[50, 200], request_minimum=True)

	y_values_average_background_subtracted = adjust_intensity(intensity_values=y_values_average, ground_level=background, reverse=False)
	print(y_values_average_background_subtracted)

	border_points_x = [x_values_average[0], x_values_average[-3],  x_values_average[-2], x_values_average[-1]]
	border_points_y = [y_values_average_background_subtracted[0], y_values_average_background_subtracted[-3],  y_values_average_background_subtracted[-2], y_values_average_background_subtracted[-1]]
	border_points_y_error = [y_values_average_std[0], y_values_average_std[-3],  y_values_average_std[-2], y_values_average_std[-1]]

	print(border_points_x, border_points_y)
	popt_exponential_decay = curve_fit(exponential, border_points_x, border_points_y,
									   p0=[100, 0.005, 60], sigma=border_points_y_error)
	decay_length, amplitude, shift = popt_exponential_decay[0]
	print(decay_length, amplitude, shift)
	y_values_fit = exponential(x_values_average, decay_length, amplitude, shift)
	print(y_values_fit)

	y_values_fit = adjust_intensity(intensity_values=y_values_fit, ground_level=-background, reverse=False)

	y_values_data_minus_fit = y_values_average - y_values_fit

	popt_gaussian = curve_fit(gaussian, x_values_average, y_values_data_minus_fit,
									   p0=[100, 30, 1], sigma=y_values_average_std)
	mu, sig, amplitude = popt_gaussian[0]
	print(mu, sig, amplitude)

	x_values_fit = np.linspace(49, 151, 102)
	y_values_gaussian_fit = gaussian(x_values_fit, mu, sig, amplitude)
	print(y_values_gaussian_fit)

	normalization = scale_values_to_unity(y_values_gaussian_fit, request_max=True)
	y_values_gaussian_fit *= normalization


	# Compute Peaks
	# popt_decaying_sinusoid = curve_fit(exponentially_decaying_gaussian, x_values_average, y_values_average_background_subtracted,
	# 								   p0=[100, 20, 0.005, 60, 0.2, 20])
	# mu, sig, amplitude_gaussian, decay_length, amplitude_exponential, shift = popt_decaying_sinusoid[0]
	# print(mu, sig, amplitude_gaussian, decay_length, amplitude_exponential, shift)

	# x_values_fit = x_values_average
	# x_values_fit = np.linspace(0, 200, 200)

	# y_values_fit = exponentially_decaying_gaussian(x_values_fit, mu, sig, amplitude_gaussian, decay_length, amplitude_exponential, shift)
	# y_values_fit = adjust_intensity(intensity_values=y_values_fit, ground_level=-background, reverse=False)

	# normalization = scale_values_to_unity(y_values_fit, request_max=True)
	# y_values_fit *= normalization

	fig, ax1 = plt.subplots(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
	ax2 = ax1.twinx()

	ax1.plot(x_values, y_values, 'o',
			 label=f'Measured Data with {number_of_measurements_per_data_point} measurements per data point', alpha=0.2)
	ax1.errorbar(x_values_average, y_values_average, yerr=y_values_average_std, marker="*", color="navy", capsize=5,
				 label='Average Measured Data with 1 Standard Deviation Errors')
	ax1.plot(x_values_average, y_values_fit, '*', linestyle="dashed", color="darkred",
				label=f"Exponential with decay time {decay_length:.2f} s")

	handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
	handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
	handles = handles_ax1 + handles_ax2
	labels = labels_ax1 + labels_ax2
	order = [0, 2, 1]
	plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

	ax1.set_xlabel("Free evolution time $\\tau$ [s]")
	ax1.set_ylabel("Intensity [counts]", color="navy")
	# ax1.legend()

	# ax2.set_ylim(-1.05, 1.05)
	ax2.set_ylabel("Spin z-component [a.u]", color="darkred")
	# ax2.legend()

	# ax2.hlines(0, x_values_fit[0], x_values_fit[-1], linestyles=":")
	# #
	# plt.vlines(phase_shift, -1, 1, linestyles=":")
	# plt.text(phase_shift, -1, "Phase shift $\phi$", fontsize=14, verticalalignment='top')
	# plt.vlines(phase_shift+omega/2, -1, 1, linestyles=":")
	# plt.text(phase_shift+omega/2, -1, "$\\tau_{\pi}$", fontsize=14, verticalalignment='top')
	# plt.vlines(phase_shift+omega, -1, 1, linestyles=":")
	# plt.text(phase_shift+omega, -1, "$\\tau_{2\pi}$", fontsize=14, verticalalignment='top')
	# plt.vlines(phase_shift+omega*3/2, -1, 1, linestyles=":")
	# plt.text(phase_shift+omega*3/2, -1, "$\\tau_{3\pi}$", fontsize=14, verticalalignment='top')
	# plt.vlines(phase_shift+omega*2, -1, 1, linestyles=":")
	# plt.text(phase_shift+omega*2, -1, "$\\tau_{4\pi}$", fontsize=14, verticalalignment='top')

	plt.show()


	fig, ax1 = plt.subplots(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
	ax2 = ax1.twinx()

	ax1.errorbar(x_values_average, y_values_average-y_values_fit, yerr=y_values_average_std, marker="*", color="navy", capsize=5,
				 label='Average Measured Data with 1 Standard Deviation Errors')
	ax2.plot(x_values_fit, y_values_gaussian_fit, '*', linestyle="dashed", color="darkred",
				label=f"Gaussian function  peaked at {mu:.3f} s with std {sig:.3f} s")

	ylim = [-0.3, 1.5]
	ax1.set_ylim(ylim[0]/normalization, ylim[1]/normalization)
	ax1.set_xlabel("Free evolution time $\\tau_{2}$ [s]")
	ax1.set_ylabel("Intensity [counts]", color="navy")
	# ax1.legend()

	ax2.set_ylim(ylim[0], ylim[1])
	ax2.set_ylabel("Spin z-component [a.u.]", color="darkred")

	# ax2.hlines(0, x_values_fit[0], x_values_fit[-1], linestyles=":")

	ax2.vlines(mu, 0, 1, linestyles=":")
	plt.text(mu, 0, f"$\\tau_{{1}}$={mu:.3f} s", fontsize=10, verticalalignment='top')

	handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
	handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
	handles = handles_ax1 + handles_ax2
	labels = labels_ax1 + labels_ax2
	order = [0, 1]

	plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

	plt.show()




def plot_data_echo_50ns(x_values, y_values):
	number_of_measurements_per_data_point, x_values_average, y_values_average, y_values_average_std \
		= average_data(x_values, y_values)

	background = compute_background_intensity(intensity_values=y_values_average, frequency_values=x_values_average, frequency_limit=[24, 76], request_minimum=True)
	print(background)

	y_values_average_background_subtracted = adjust_intensity(intensity_values=y_values_average, ground_level=background, reverse=False)
	# print(y_values_average_background_subtracted)

	popt_gaussian = curve_fit(gaussian, x_values_average, y_values_average_background_subtracted,
									   p0=[50, 30, 1], sigma=y_values_average_std)
	mu, sig, amplitude = popt_gaussian[0]
	print(mu, sig, amplitude)

	x_values_fit = np.linspace(24, 76, 52)
	y_values_gaussian_fit = gaussian(x_values_fit, mu, sig, amplitude)
	# print(y_values_gaussian_fit)

	y_values_gaussian_fit = adjust_intensity(intensity_values=y_values_gaussian_fit, ground_level=-background, reverse=False)

	# normalization = scale_values_to_unity(y_values_gaussian_fit, request_max=True)
	# y_values_gaussian_fit *= normalization


	# Compute Peaks
	# popt_decaying_sinusoid = curve_fit(exponentially_decaying_gaussian, x_values_average, y_values_average_background_subtracted,
	# 								   p0=[100, 20, 0.005, 60, 0.2, 20])
	# mu, sig, amplitude_gaussian, decay_length, amplitude_exponential, shift = popt_decaying_sinusoid[0]
	# print(mu, sig, amplitude_gaussian, decay_length, amplitude_exponential, shift)

	# x_values_fit = x_values_average
	# x_values_fit = np.linspace(0, 200, 200)

	# y_values_fit = exponentially_decaying_gaussian(x_values_fit, mu, sig, amplitude_gaussian, decay_length, amplitude_exponential, shift)
	# y_values_fit = adjust_intensity(intensity_values=y_values_fit, ground_level=-background, reverse=False)

	# normalization = scale_values_to_unity(y_values_fit, request_max=True)
	# y_values_fit *= normalization

	fig, ax1 = plt.subplots(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
	ax2 = ax1.twinx()

	ax1.plot(x_values, y_values, 'o',
			 label=f'Measured Data with {number_of_measurements_per_data_point} measurements per data point', alpha=0.2)
	ax1.errorbar(x_values_average, y_values_average, yerr=y_values_average_std, marker="*", color="navy", capsize=5,
				 label='Average Measured Data with 1 Standard Deviation Errors')
	ax1.plot(x_values_fit, y_values_gaussian_fit, '*', linestyle="dashed", color="darkred",
				label=f"Gaussian function  peaked at {mu:.3f} s with std {sig:.3f} s")

	handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
	handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
	handles = handles_ax1 + handles_ax2
	labels = labels_ax1 + labels_ax2
	order = [0, 2, 1]

	plt.legend([handles[idx] for idx in order],
			   [labels[idx] for idx in order],
			   loc=4)

	ax1.set_xlabel("Free evolution time $\\tau$ [s]")
	ax1.set_ylabel("Intensity [counts]", color="navy")
	# ax1.legend()

	# ax2.set_ylim(-1.05, 1.05)
	ax2.set_ylabel("Spin z-component [a.u]", color="darkred")
	# ax2.legend()

	ax2.vlines(mu, 0, 1, linestyles=":")
	plt.text(mu, 0.3, f"$\\tau_{{1}}$={mu:.3f} s", fontsize=10, verticalalignment='top')

	plt.show()


def main():
	voltage, frequency, intensity = read_data("data/odmr_118")
	# plot_data(frequency, intensity, fit_function=gaussian)
	plot_data_voltage_frequency_correlation(voltage, frequency)

	# voltage, frequency, intensity = read_data("data/odmr2_118")
	# plot_data(frequency, intensity, fit_function=gaussian)

	# voltage, frequency, intensity = read_data("data/odmr_zoom_118")
	# plot_data(frequency, intensity, fit_function=gaussian)

	# time, intensity = read_data("data/Rabi_118", data_format="time")
	# plot_data_rabi(time, intensity)

	# time, intensity = read_data("data/Ramsey_118", data_format="time")
	# plot_data_ramsey(time, intensity)

	# plot_data_echo(time, intensity)
	# time, intensity = read_data("data/Echo_118", data_format="time")

	# time, intensity = read_data("data/Echo_50ns_118", data_format="time")
	# plot_data_echo_50ns(time, intensity)


if __name__ == "__main__":
	main()
