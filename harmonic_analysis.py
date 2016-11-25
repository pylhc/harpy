import numpy as np

PI2I = 2 * np.pi * complex(0, 1)
CZERO = complex(0, 0)


class HarmonicAnalisys(object):

    def __init__(self, samples):
        self._samples = samples
        self._length = len(samples)
        self._int_range = np.arange(self._length)
        self._hann_window = np.hanning(self._length)

    def laskar_method(self, num_harmonics):
        samples = self._samples[:]  # Copy the samples array.
        n = self._length
        coefficients = []
        frequencies = []
        for _ in range(num_harmonics):
            # Compute this harmonic frequency and coefficient.
            dft_data = HarmonicAnalisys._fft(samples)
            frequency = self._jacobsen(dft_data, (0, len(dft_data)))
            coefficient = self._compute_coef(samples, frequency * n) / n

            # Store frequency and amplitude
            coefficients.append(coefficient)
            frequencies.append(frequency)

            # Subtract the found pure tune from the signal
            new_signal = coefficient * np.exp(PI2I * frequency * self._int_range)
            samples = samples - new_signal

        coefficients, frequencies = zip(*sorted(zip(coefficients, frequencies),
                                                key=lambda tuple: np.abs(tuple[0]),
                                                reverse=True))
        return frequencies, coefficients

    def _jacobsen(self, dft_values, frequency_window):
        """
        This method interpolates the real frequency of the
        signal using the three highest peaks in the FFT.
        """
        k = np.argmax(np.abs(dft_values))
        n = self._length
        r = dft_values
        delta = np.tan(np.pi / n) / (np.pi / n)
        kp = (k + 1) % n
        km = (k - 1) % n
        delta = delta * np.real((r[km] - r[kp]) / (2 * r[k] - r[km] - r[kp]))
        return (k + delta) / n

    def _compute_coef_simple(self, samples, kprime):
        """
        Computes the coefficient of the Discrete Time Fourier
        Transform corresponding to the given frequency (kprime).
        """
        n = self._length
        freq = kprime / n
        exponents = np.exp(-PI2I * freq * self._int_range)
        coef = np.sum(exponents * samples)
        return coef

    def _compute_coef_goertzel(self, samples, kprime):
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

    @staticmethod
    def _conditional_import_compute_coef():
        """
        Checks if Numba is installed.
        If it is, it sets the compiled goertzel algorithm as the
        coefficient function to use. If it isn't, it uses the
        normal Numpy one.
        """
        try:
            from numba import jit
            print("Using compiled Numba functions.")
            return jit(HarmonicAnalisys._compute_coef_goertzel)
        except ImportError:
            print("Numba not found, using numpy functions.")
            return HarmonicAnalisys._compute_coef_simple

    @staticmethod
    def _conditional_import_fft():
        """
        If SciPy is installed, it will set its fft as the one
        to use as it is slightly faster. Otherwise it will use
        the Numpy one.
        """
        try:
            from scipy.fftpack import fft as scipy_fft
            fft = staticmethod(scipy_fft)
            print("Scipy found, using scipy FFT.")
        except ImportError:
            from numpy.fft import fft as numpy_fft
            fft = staticmethod(numpy_fft)
            print("Scipy not found, using numpy FFT.")
        return fft

# Set up conditional functions on load ##############################################
HarmonicAnalisys._compute_coef = HarmonicAnalisys._conditional_import_compute_coef()
HarmonicAnalisys._fft = HarmonicAnalisys._conditional_import_fft()
#####################################################################################
