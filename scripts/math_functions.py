import numpy as np


def gaussian(x, mu, sig, amplitude):
	return amplitude * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def exponential(x, decay_length, amplitude):
	return amplitude * np.exp(-x / decay_length)


def exponentially_decaying_gaussian(x, mu, sig, amplitude_gaussian, decay_length, amplitude_exponential):
	return gaussian(x, mu, sig, amplitude_gaussian) * exponential(x, decay_length, amplitude_exponential)


def exponentially_decaying_sinusoid(x, omega, decay_time, amplitude):
	return amplitude * np.cos(x * 2 * np.pi / omega) * exponential(x, decay_time, 1)


def scale_values_to_unity(y_values):
	initial_value = y_values[0]
	normalization = 1 / initial_value
	return normalization
