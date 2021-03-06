import numpy as np


def gaussian(x, mu, sig, amplitude):
	return amplitude * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def exponential(x, decay_length, amplitude, shift):
	return amplitude * np.exp(-1 * (x-shift) / decay_length)


def exponentially_decaying_gaussian(x, mu, sig, amplitude_gaussian, decay_length, amplitude_exponential, shift):
	return gaussian(x + shift, mu, sig, amplitude_gaussian) * exponential(x, decay_length, amplitude_exponential)


def exponentially_decaying_cosine(x, omega, decay_time, amplitude, x_shift):
	return amplitude * np.cos(x * 2 * np.pi / omega) * exponential(x, decay_time, 1, x_shift)


def exponentially_decaying_sine(x, omega, decay_time, amplitude, phase_shift, x_shift):
	return amplitude * np.cos(x * 2 * np.pi / omega + phase_shift) * exponential(x, decay_time, 1, x_shift)


def scale_values_to_unity(y_values, request_max=False):
	reference_value = y_values[0]
	if request_max:
		reference_value = max(y_values)
	normalization = 1 / reference_value
	return normalization


def linear_function(x, slope, interceipt):
	return x * slope + interceipt

