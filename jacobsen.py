"""
DFT peak interpolation using the Jacobsen method with bias correction.
"""

from __future__ import print_function
import sys
import numpy as np

I = complex(0, 1)
PI2I = 2 * np.pi * I


def jacobsen(dft_values, frequency_window):
    k, n, r = _get_dft_peak(dft_values, frequency_window)
    delta = np.tan(np.pi / n) / (np.pi / n)
    kp = (k + 1) % n
    km = (k - 1) % n
    delta = delta * np.real((r[km] - r[kp]) / (2 * r[k] - r[km] - r[kp]))
    return (k + delta) / n


def quinn(dft_values, frequency_window):
    k, n, r = _get_dft_peak(dft_values, frequency_window)
    alfa1 = np.real(r[k - 1] - r[k])
    alfa2 = np.real(r[k + 1] - r[k])
    delta1 = alfa1 / (1 - alfa1)
    delta2 = alfa2 / (1 - alfa2)
    delta = delta1
    if delta1 > 0 and delta2 > 0:
        delta = delta2
    return (k + delta) / n


def parabolic_fit(dft_values, frequency_window, k):
    n = len(dft_values)
    kp = (k + 1) % n
    km = (k - 1) % n
    real_k, real_kp, real_km = (np.real(dft_values[k]), np.real(dft_values[kp]), np.real(dft_values[km]))
    imag_k, imag_kp, imag_km = (np.imag(dft_values[k]), np.imag(dft_values[kp]), np.imag(dft_values[km]))
    real_peak = real_k - ((real_km - real_kp) ** 2. / (8. * (real_km + real_kp - 2. * real_k)))
    imag_peak = imag_k - ((imag_km - imag_kp) ** 2. / (8. * (imag_km + imag_kp - 2. * imag_k)))
    return np.complex128(complex(real_peak, imag_peak))


def grid_search(samples, dft_values, frequency_window, iterations):
    k, n, r = _get_dft_peak(dft_values, frequency_window)
    delta = 0.
    for i in range(iterations):
        current_coef_p = _compute_coef(samples, k + delta + 0.5)
        current_coef_n = _compute_coef(samples, k + delta - 0.5)
        delta += np.real((current_coef_p + current_coef_n) / (current_coef_p - current_coef_n)) / 2
        # delta += ((abs(current_coef_p) - abs(current_coef_n)) / (abs(current_coef_p) + abs(current_coef_n))) / 2
    return (k + delta) / n


def brute_force_search(samples, dft_values, frequency_window, grid_points_num):
    k, n, r = _get_dft_peak(dft_values, frequency_window)
    grid = np.linspace(-0.5, 0.5, grid_points_num)
    freqs = [(k + d) / n for d in grid]
    values = [np.abs(_compute_coef(samples, k + point)) for point in grid]
    return freqs[np.argmax(values) + 1]


def laskar_method(samples, num_harmonics):
    n = len(samples)
    coefficients = []
    frequencies = []
    uniform_freqs = np.arange(n, dtype=np.float64) / n
    dft_data = _fft(samples)
    for i in range(num_harmonics):
        # Compute this harmonic frequency and coefficient.
        frequency = jacobsen(dft_data, (0, len(dft_data)))
        coefficient = _compute_coef_freq_space(dft_data, uniform_freqs, frequency, n)

        # Store frequency and amplitude
        coefficients.append(coefficient)
        frequencies.append(frequency)

        # Subtract the found pure tune from the signal
        new_signal_dft =  _sum_formula(coefficient, frequency, uniform_freqs, n)
        dft_data = dft_data - new_signal_dft

    coefficients, frequencies = zip(*sorted(zip(coefficients, frequencies),
                                            key=lambda tuple: np.abs(tuple[0]),
                                            reverse=True))
    return frequencies, coefficients


def resonance_search(frequencies, coefficients, tune_x, tune_y, tune_z, tune_tolerance, resonance_list):
    resonances = {}
    remaining_resonances = resonance_list[:]
    sorted_coefficients, sorted_frequencies = zip(*sorted(zip(coefficients, frequencies),
                                                          key=lambda tuple: np.abs(tuple[0]),
                                                          reverse=True))
    for index in range(len(sorted_frequencies)):
        coefficient = sorted_coefficients[index]
        frequency = sorted_frequencies[index]
        for resonance in remaining_resonances:
            resonance_h, resonance_v, resonance_l = resonance
            resonance_freq = (resonance_h * tune_x) + (resonance_v * tune_y) + (resonance_l * tune_z)
            if resonance_freq < 0:
                resonance_freq = 1. + resonance_freq  # [0, 1] domain
            min_freq = resonance_freq - tune_tolerance
            max_freq = resonance_freq + tune_tolerance
            if frequency >= min_freq and frequency <= max_freq:
                resonances[resonance] = (frequency, coefficient)
                remaining_resonances.remove(resonance)
                break
    return resonances


def _compute_coef_freq_space(dft_values, uniform_freqs, fprime, n):
    if fprime in uniform_freqs:
        return dft_values[fprime * n] / n
    a = np.exp(PI2I * (uniform_freqs - fprime) * n) - 1
    b = np.exp(PI2I * (uniform_freqs - fprime)) - 1
    return np.sum(dft_values / n * (a / b)) / n


def _get_dft_peak(dft_values, frequency_window):
    r = dft_values
    min_value, max_value = frequency_window
    n = len(dft_values)
    min_k = min_value * n
    max_k = max_value * n
    k = np.argmax(np.abs(dft_values[min_k:max_k + 1])) + min_k
    return k, n, r


def _sum_formula(coefficient, fp, fk, n):
    """
    This is equivalent to the DFT of a pure tone of frequency fp.
    It is the formula for the sum of the series:
    sum(exp(2 pi i (fp - fk) n) with n from 0 to N - 1)
    fp is the signal real frequency, fk the frequency of each
    DFT bin and n (N in the formula) the size of the DFT.
    """
    if fp in fk:
        answer = np.zeros(n, dtype=np.complex128)
        answer[fp * n] = coefficient * n
        return answer
    a = np.exp(2 * np.pi * I * n * (fp - fk)) - 1
    b = np.exp(2 * np.pi * I * (fp - fk)) - 1
    return coefficient * (a / b)


# Conditional imports #
try:
    from scipy.fftpack import fft as scipy_fft
    _fft = scipy_fft
    print("Scipy found, using scipy FFT.")
except ImportError:
    from numpy.fft import fft as numpy_fft
    _fft = numpy_fft
    print("Scipy not found, using numpy FFT.")
######################


if __name__ == "__main__":
    print("This module is meant to be imported", file=sys.stderr)
