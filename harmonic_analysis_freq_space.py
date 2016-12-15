import numpy as np
import harmonic_analysis

PI2I = 2 * np.pi * complex(0, 1)
CZERO = complex(0, 0)

HANN_DEF = False


class HarmonicAnalysisFreqSpc(harmonic_analysis.HarmonicAnalysis):

    def __init__(self, samples, zero_pad=False, hann=HANN_DEF):
        self._samples = samples
        self._compute_orbit()
        self._length = len(self._samples)
        self._freq_range = np.arange(self._length) / self._length
        self._hann_window = None
        if hann:
            self._hann_window = np.hanning(self._length)

    def laskar_method(self, num_harmonics):
        n = self._length
        coefficients = []
        frequencies = []
        dft_data = HarmonicAnalysisFreqSpc._fft(self._samples)
        for i in range(num_harmonics):
            # Compute this harmonic frequency and coefficient.
            frequency = self._jacobsen(dft_data)
            frequencies.append(frequency)

            # If the frequency found is in one of the bins just
            # remove it form the signal.
            if frequency in self._freq_range:
                index = np.where(self._freq_range == frequency)[0][0]
                coefficients.append(dft_data[index] / n)
                dft_data[index] = CZERO
                continue

            coefficient = HarmonicAnalysisFreqSpc._compute_coef(
                    dft_data,
                    self._freq_range,
                    frequency,
                    n
            )
            coefficients.append(coefficient)

            # Subtract the found pure tune from the signal
            new_signal_dft = HarmonicAnalysisFreqSpc._sum_formula(
                coefficient,
                frequency,
                self._freq_range,
                n
            )
            dft_data -= new_signal_dft

        coefficients, frequencies = zip(
            *sorted(zip(coefficients, frequencies),
                    key=lambda tuple: np.abs(tuple[0]),
                    reverse=True)
        )
        return frequencies, coefficients

    def get_signal(self):
        if self._hann_window is not None:
            return self._samples * self._hann_window
        else:
            return self._samples

    @staticmethod
    def _compute_coef_simple(dft_values, uniform_freqs, fprime, n):
        """
        Computes the coefficient of the Discrete Time Fourier
        Transform corresponding to the given frequency (kprime),
        directly in the frequency space.
        """
        a = np.exp(PI2I * (uniform_freqs - fprime) * n) - 1
        b = np.exp(PI2I * (uniform_freqs - fprime)) - 1
        return np.sum(dft_values / n * (a / b)) / n

    @staticmethod
    def _compute_coef_jit(dft_values, uniform_freqs, fprime, n):
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

    @staticmethod
    def _sum_formula_simple(coefficient, fp, fk, n):
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

    @staticmethod
    def _sum_formula_jit(coefficient, fp, fk, n, result):
        """
        This is the same as _sum_formula, but it can be compiled
        with Numba to be way faster.
        """
        signal_dft = result
        for i in range(n):
            signal_dft[i] = coefficient * (
                (np.exp(PI2I * n * (fp - fk[i])) - 1) /
                (np.exp(PI2I * (fp - fk[i])) - 1)
            )

    @staticmethod
    def _sum_formula_jit_wrapper(coefficient, fp, fk, n):
        """
        One cannot create or return Numpy array in Numba.
        This wrapper makes the transformation transparent to laskar_method.
        """
        result = np.zeros(n, dtype=np.complex128)
        HarmonicAnalysisFreqSpc._sum_formula_jit(
            coefficient, fp, fk, n, result
        )
        return result

    @staticmethod
    def _conditional_import_compute_coef():
        try:
            from numba import jit
            print("Using compiled Numba coefficient.")
            return jit(HarmonicAnalysisFreqSpc._compute_coef_jit,
                       nopython=True, nogil=True)
        except ImportError:
            print("Numba not found, using numpy coefficient formula.")
            return HarmonicAnalysisFreqSpc._compute_coef_simple

    @staticmethod
    def _conditional_import_sum_formula():
        try:
            from numba import jit
            print("Using compiled Numba sum formula.")
            HarmonicAnalysisFreqSpc._sum_formula_jit = jit(
                HarmonicAnalysisFreqSpc._sum_formula_jit,
                nopython=True,
                nogil=True
            )
            return HarmonicAnalysisFreqSpc._sum_formula_jit_wrapper
        except ImportError:
            print("Numba not found, using numpy sum formula.")
            return HarmonicAnalysisFreqSpc._sum_formula_simple

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
HarmonicAnalysisFreqSpc._compute_coef = HarmonicAnalysisFreqSpc._conditional_import_compute_coef()
HarmonicAnalysisFreqSpc._sum_formula = HarmonicAnalysisFreqSpc._conditional_import_sum_formula()
HarmonicAnalysisFreqSpc._fft = HarmonicAnalysisFreqSpc._conditional_import_fft()
#####################################################################################
