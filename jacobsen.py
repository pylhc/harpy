"""
DFT peak interpolation using the Jacobsen method with bias correction.
"""

from __future__ import print_function
import sys
import numpy as np

PI2I = 2 * np.pi * complex(0, 1)
CZERO = complex(0, 0)


def laskar_method_time_space(samples, num_harmonics):
    n = len(samples)
    coefficients = []
    frequencies = []
    uniform_samples = np.arange(n)
    for i in range(num_harmonics):
        # Compute this harmonic frequency and coefficient.
        dft_data = _fft(samples)
        frequency = jacobsen(dft_data, (0, len(dft_data)))
        coefficient = _compute_coef(samples, frequency * n) / n

        # Store frequency and amplitude
        coefficients.append(coefficient)
        frequencies.append(frequency)

        # Subtract the found pure tune from the signal
        new_signal = coefficient * np.exp(PI2I * frequency * uniform_samples)
        samples = samples - new_signal

    coefficients, frequencies = zip(*sorted(zip(coefficients, frequencies),
                                            key=lambda tuple: np.abs(tuple[0]),
                                            reverse=True))
    return frequencies, coefficients


def laskar_method_freq_space(samples, num_harmonics):
    n = len(samples)
    coefficients = []
    frequencies = []
    uniform_freqs = np.arange(n, dtype=np.float64) / n
    dft_data = _fft(samples)
    for i in range(num_harmonics):
        # Compute this harmonic frequency and coefficient.
        frequency = jacobsen(dft_data, (0, len(dft_data)))
        frequencies.append(frequency)

        # If the frequency found is in one of the bins just
        # remove it form the signal.
        if frequency in uniform_freqs:
            index = np.where(uniform_freqs == frequency)[0][0]
            coefficients.append(dft_data[index] / n)
            dft_data[index] = CZERO
            continue

        coefficient = _compute_coef_freq_space(dft_data, uniform_freqs, frequency, n)
        coefficients.append(coefficient)

        # Subtract the found pure tune from the signal
        new_signal_dft = _sum_formula(coefficient, frequency, uniform_freqs, n)
        dft_data = dft_data - new_signal_dft

    coefficients, frequencies = zip(*sorted(zip(coefficients, frequencies),
                                            key=lambda tuple: np.abs(tuple[0]),
                                            reverse=True))
    return frequencies, coefficients


def jacobsen(dft_values, frequency_window):
    """
    This method interpolates the real frequency of the
    signal using the three highest peaks in the FFT.
    """
    k, n, r = _get_dft_peak(dft_values, frequency_window)
    delta = np.tan(np.pi / n) / (np.pi / n)
    kp = (k + 1) % n
    km = (k - 1) % n
    delta = delta * np.real((r[km] - r[kp]) / (2 * r[k] - r[km] - r[kp]))
    return (k + delta) / n


def _get_dft_peak(dft_values, frequency_window):
    """
    Finds the index (k) of the highest peak in the FFT.
    """
    r = dft_values
    min_value, max_value = frequency_window
    n = len(dft_values)
    min_k = min_value * n
    max_k = max_value * n
    k = np.argmax(np.abs(dft_values[min_k:max_k + 1])) + min_k
    return k, n, r


def _compute_coef(samples, kprime):
    """
    Computes the coefficient of the Discrete Time Fourier
    Transform corresponding to the given frequency (kprime).
    """
    n = len(samples)
    freq = kprime / n
    exponents = np.exp(-PI2I * freq * np.arange(n))
    coef = np.sum(exponents * samples)
    return coef


def _compute_coef_goertzel(samples, kprime):
    """
    Computes the coefficient of the Discrete Time Fourier
    Transform corresponding to the given frequency (kprime).
    This function is faster than the previous one if compiled
    with Numba.
    """
    n = len(samples)
    a = 2 * np.pi * (kprime / n)
    b = 2 * np.cos(a)
    c = np.exp(-complex(0, 1) * a)
    d = np.exp(-complex(0, 1) * ((2 * np.pi * kprime) / n) * (n - 1))
    s0 = 0.
    s1 = 0.
    s2 = 0.
    for i in range(n - 1):
        s0 = samples[i] + b * s1 - s2
        s2 = s1
        s1 = s0
    s0 = samples[n - 1] + b * s1 - s2
    y = s0 - s1 * c
    return y * d


def _compute_coef_freq_space(dft_values, uniform_freqs, fprime, n):
    """
    Computes the coefficient of the Discrete Time Fourier
    Transform corresponding to the given frequency (kprime),
    directly in the frequency space.
    """
    a = np.exp(PI2I * (uniform_freqs - fprime) * n) - 1
    b = np.exp(PI2I * (uniform_freqs - fprime)) - 1
    return np.sum(dft_values / n * (a / b)) / n


def _compute_coef_freq_space_jit(dft_values, uniform_freqs, fprime, n):
    """
    Computes the coefficient of the Discrete Time Fourier
    Transform corresponding to the given frequency (kprime),
    directly in the frequency space. This function can be
    compiled with Numba to make it faster.
    """
    coefficient = complex(0.0, 0.0)
    for i in range(n):
        a = np.exp(PI2I * (uniform_freqs[i] - fprime) * n) - 1
        b = np.exp(PI2I * (uniform_freqs[i] - fprime)) - 1
        coefficient += dft_values[i] / n * (a / b)
    return coefficient / n


def _sum_formula(coefficient, fp, fk, n):
    """
    This is equivalent to the DFT of a pure tone of frequency fp.
    It is the formula for the sum of the series:
    sum(exp(2 pi i (fp - fk) n) with n from 0 to N - 1)
    fp is the signal real frequency, fk the frequency of each
    DFT bin and n (N in the formula) the size of the DFT.
    """
    a = np.exp(PI2I * n * (fp - fk)) - 1
    b = np.exp(PI2I * (fp - fk)) - 1
    return coefficient * (a / b)


def _sum_formula_jit(coefficient, fp, fk, n, result):
    """
    This is the same as _sum_formula, but it can be compiled
    with Numba to be way faster.
    """
    signal_dft = result
    for i in range(n):
        signal_dft[i] = coefficient * ((np.exp(PI2I * n * (fp - fk[i])) - 1) /
                                       (np.exp(PI2I * (fp - fk[i])) - 1))


def _sum_formula_jit_wrapper(coefficient, fp, fk, n):
    """
    One cannot create or return Numpy array in Numba.
    This wrapper makes the transformation transparent to laskar_method.
    """
    result = np.zeros(n, dtype=np.complex128)
    _sum_formula_jit(coefficient, fp, fk, n, result)
    return result


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


# Conditional imports #

try:
    from numba import jit
    _sum_formula_jit = jit(_sum_formula_jit, nopython=True)
    _sum_formula = _sum_formula_jit_wrapper  # noqa
    _compute_coef_freq_space = jit(_compute_coef_freq_space_jit, nopython=True)  # noqa
    _compute_coef = jit(_compute_coef_goertzel)  # noqa
    print("Using compiled Numba functions.")
except ImportError:
    print("Numba not found, using numpy functions.")

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
