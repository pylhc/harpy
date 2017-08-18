from __future__ import print_function
import os
import multiprocessing
import logging
import numpy as np
from harmonic_analysis import HarmonicAnalysis

import _python_path_manager
_python_path_manager.append_betabeat()

from Python_Classes4MAD import metaclass  # noqa
from Utilities import tfs_file_writer  # noqa
from Utilities import iotools  # noqa
from Utilities import outliers  # noqa

LOGGER = logging.getLogger(__name__)

PI2I = 2 * np.pi * complex(0, 1)

HEADERS = {"X": ["NAME", "S", "BINDEX", "SLABEL", "TUNEX", #"TUNEZ"
                 "NOISE", "PK2PK", "CO", "CORMS", "AMPX",
                 "MUX", "AVG_AMPX", "AVG_MUX", "BPM_RES"],
           "Y": ["NAME", "S", "BINDEX", "SLABEL", "TUNEY", #"TUNEZ"
                 "NOISE", "PK2PK", "CO", "CORMS",
                 "AMPY", "MUY", "AVG_AMPY", "AVG_MUY", "BPM_RES"]}

SPECTR_COLUMN_NAMES = ["FREQ", "AMP"]

RESONANCE_LISTS = {"X": ((1, 0, 0), (0, 1, 0), (-2, 0, 0), (0, 2, 0), (-3, 0, 0), (-1, -1, 0),
                         (2, -2, 0), (0, -2, 0), (1, -2, 0), (-1, 3, 0), (1, 2, 0), (-2, 1, 0),
                         (1, 1, 0), (2, 0, 0), (-1, -2, 0), (3, 0, 0), (0, 0, 1)),
                   "Y": ((0, 1, 0), (1, 0, 0), (-1, 1, 0), (-2, 0, 0), (1, -1, 0), (0, -2, 0),
                         (0, -3, 0), (2, 1, 0), (-1, 3, 0), (1, 1, 0), (-1, 2, 0), (0, 0, 1))}

MAIN_LINES = {"X": (1, 0, 0),
              "Y": (0, 1, 0)}

N_TO_P = {"0": "X",
          "1": "Y"}

DEFAULT_DFT_METHOD = "laskar"

DEF_TUNE_TOLERANCE = 0.001

NUM_HARMS = 300

PROCESSES = multiprocessing.cpu_count()


class DriveAbstract(object):
    def __init__(self,
                 tunes,
                 plane,
                 nattunes=None,
                 tolerance=DEF_TUNE_TOLERANCE,
                 start_turn=0,
                 end_turn=None,
                 sequential=False):
        # Inputs
        self._tunes = tunes
        self._plane = plane
        self._nattunes = nattunes
        self._tolerance = tolerance
        self._start_turn = start_turn
        self._end_turn = end_turn
        self._sequential = sequential
        self._compute_resonances_freqs()
        # Outputs
        self._measured_tune = None
        self._bpm_results = []

    @property
    def measured_tune(self):
        if self._measured_tune is None:
            raise ValueError(
                "Value not computed yet. Run start_analysis() first"
            )
        return self._measured_tune

    @property
    def bpm_results(self):
        if len(self._bpm_results) == 0:
            raise ValueError(
                "Value not computed yet. Run start_analysis() first"
            )
        return self._bpm_results

    # Public methods
    def start_analysis(self):
        self._bpm_processors = []
        self._do_analysis()
        self._gather_results()

    def write_full_results(self):
        LOGGER.debug("Writting results...")
        self._create_lin_files()
        iotools.create_dirs(self._spectr_outdir)
        lin_outfile = self._lin_outfile
        for bpm_results in self.bpm_results:
            self._write_single_bpm_results(
                lin_outfile,
                bpm_results
            )
        plane_number = "1" if self._plane == "X" else "2"
        tune, rms_tune = self._measured_tune
        lin_outfile.add_float_descriptor("Q" + plane_number, tune)
        lin_outfile.add_float_descriptor("Q" + plane_number + "RMS",
                                         rms_tune)
        lin_outfile.order_rows("S")
        lin_outfile.write_to_file()
        LOGGER.debug("Writting done.")
    ######

    # Methods to override in subclasses:
    def _do_analysis(self):
        raise NotImplementedError("Dont instantiate this abstract class!")

    def _get_outfile_name(self):
        raise NotImplementedError("Dont instantiate this abstract class!")
    ######

    # Private methods
    def _compute_resonances_freqs(self):
        """
        Computes the frequencies for all the resonances listed in the
        constante RESONANCE_LISTS, together with the natural tunes
        frequencies if given.
        """
        tune_x, tune_y, tune_z = self._tunes
        self._resonances_freqs = {}
        freqs = [(resonance_h * tune_x) +
                 (resonance_v * tune_y) +
                 (resonance_l * tune_z)
                 for (resonance_h,
                      resonance_v,
                      resonance_l) in RESONANCE_LISTS[self._plane]]
        # Move to [0, 1] domain.
        freqs = [freq + 1. if freq < 0. else freq for freq in freqs]
        self._resonances_freqs[self._plane] = dict(
            zip(RESONANCE_LISTS[self._plane], freqs)
        )
        if self._nattunes is not None:
            nattune_x, nattune_y, nattune_z = self._nattunes
            if self._plane == "X" and nattune_x is not None:
                self._resonances_freqs["X"]["NATX"] = nattune_x
            if self._plane == "Y" and nattune_y is not None:
                self._resonances_freqs["Y"]["NATY"] = nattune_y
            if nattune_z is not None:
                self._resonances_freqs[self._plane]["NATZ"] = nattune_z

    def _create_lin_files(self):
        file_name = self._get_outfile_name(self._plane)
        lin_outfile = tfs_file_writer.TfsFileWriter(
            os.path.join(self._output_dir, file_name)
        )
        headers = HEADERS[self._plane]
        for resonance in RESONANCE_LISTS[self._plane]:
            if resonance == MAIN_LINES[self._plane]:
                continue
            x, y, z = resonance
            if z == 0:
                resstr = (str(x) + str(y)).replace("-", "_")
            else:
                resstr = (str(x) + str(y) + str(z)).replace("-", "_")
            headers.extend(["AMP" + resstr, "PHASE" + resstr])
        headers.extend(["NATTUNE" + self._plane,
                        "NATAMP" + self._plane])
        lin_outfile.add_column_names(headers)
        lin_outfile.add_column_datatypes(
            ["%s"] + ["%le"] * (len(headers) - 1))
        self._lin_outfile = lin_outfile

    def _gather_results(self):
        LOGGER.debug("Gathering results...")
        self._measured_tune = self._compute_tune_stats()
        tune, _ = self._measured_tune
        for bpm_processor in self._bpm_processors:
            try:
                bpm_results = bpm_processor.bpm_results
            except AttributeError:
                continue
            (bpm_results.amp_from_avg,
             bpm_results.phase_from_avg) = self._compute_from_avg(
                tune,
                bpm_processor
            )
            self._bpm_results.append(bpm_results)

    def _write_single_bpm_results(self, lin_outfile, bpm_results):
        row = [bpm_results.name, bpm_results.position, 0, 0, bpm_results.tune,
               0, bpm_results.peak_to_peak, bpm_results.closed_orbit,
               bpm_results.closed_orbit_rms, bpm_results.amplitude,
               bpm_results.phase, bpm_results.amp_from_avg,
               bpm_results.phase_from_avg, bpm_results.bpm_resolution]
        resonance_list = RESONANCE_LISTS[self._plane]
        main_resonance = MAIN_LINES[self._plane]
        for resonance in resonance_list:
            if resonance != main_resonance:
                if resonance in bpm_results.resonances:
                    _, coefficient = bpm_results.resonances[resonance]
                    row.append(np.abs(coefficient) / bpm_results.amplitude)
                    row.append(np.angle(coefficient) / (2 * np.pi))
                else:
                    row.append(0.0)
                    row.append(0.0)

        col_name = "NAT" + self._plane.upper()
        try:
            natural_freq, natural_coef = bpm_results.resonances[col_name]
            row.append(natural_freq)
            row.append(np.abs(natural_coef) / bpm_results.amplitude)
        except KeyError:
            row.append(0.0)
            row.append(0.0)
        lin_outfile.add_table_row(row)

    def _compute_tune_stats(self):
        tune_list = []
        for bpm_processor in self._bpm_processors:
            try:
                bpm_results = bpm_processor.bpm_results
            except AttributeError:
                continue
            tune_list.append(bpm_results.tune)
        tune_array=np.array(tune_list)
        #tune_array = tune_array[outliers.get_filter_mask(tune_array, limit=1e-5)] #TODO propagate the limit from options
        return np.mean(tune_array), np.std(tune_array)

    def _compute_from_avg(self, tune, bpm_results):
        coef = bpm_results.get_coefficient_for_freq(tune)
        return np.abs(coef), np.angle(coef) / (2 * np.pi)


class DriveFile(DriveAbstract):

    def __init__(self,
                 input_file,
                 tunes,
                 plane,
                 nattunes=None,
                 tolerance=DEF_TUNE_TOLERANCE,
                 start_turn=0,
                 end_turn=None,
                 output_dir=None,
                 sequential=False):
        super(DriveFile, self).__init__(
            tunes,
            plane,
            nattunes,
            tolerance,
            start_turn,
            end_turn,
            sequential,
        )
        self._input_file = input_file
        if output_dir is None:
            self._output_dir = os.path.dirname(input_file)
        else:
            self._output_dir = output_dir
        self._spectr_outdir = os.path.join(
            self._output_dir, "BPM"
        )
        self._create_spectr_outdir()

    def _create_spectr_outdir(self):
        if not os.path.isdir(self._spectr_outdir):
            os.mkdir(self._spectr_outdir)

    def _get_outfile_name(self, plane):
        return os.path.basename(self._input_file) + "_lin" + plane.lower()

    def _do_analysis(self):
        lines = []
        with open(self._input_file, "r") as records:
            for line in records:
                bpm_data = line.split()
                try:
                    bpm_plane = N_TO_P[bpm_data.pop(0)]
                except KeyError:
                    continue  # Ignore comments
                if bpm_plane == self._plane:
                    lines.append(bpm_data)
                else:
                    continue
        pool = multiprocessing.Pool(PROCESSES)
        num_of_chunks = int(len(lines) / PROCESSES) + 1
        for bpm_datas in DriveFile.chunks(lines, num_of_chunks):
            self._launch_bpm_chunk_analysis(bpm_datas, pool)
        pool.close()
        pool.join()

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in xrange(0, len(l), n):  # noqa xrange doesn't exist in Python3
            yield l[i:i + n]

    def _launch_bpm_chunk_analysis(self, bpm_datas, pool):
        args = (self._plane, self._start_turn, self._end_turn, self._tolerance,
                self._resonances_freqs, self._spectr_outdir, bpm_datas)
        if self._sequential:
            LOGGER.info("Harpy in sequential mode")
            self._bpm_processors.extend(_analyze_bpm_chunk(*args))
        else:
            pool.apply_async(
                _analyze_bpm_chunk,
                args,
                callback=self._bpm_processors.extend
            )


# Global space ################################################
def _analyze_bpm_chunk(plane, start_turn, end_turn, tolerance,
                       resonances_freqs, spectr_outdir, bpm_datas):
    """
    This function triggers the per BPM data processing.
    It has to be outside of the classes to make it pickable for
    the multiprocessing module.
    """
    results = []
    LOGGER.debug("Staring process with chunksize" + len(bpm_datas))
    for bpm_data in bpm_datas:
        name = bpm_data.pop(0)
        position = bpm_data.pop(0)
        samples = _BpmProcessor._compute_bpm_samples(
            bpm_data, start_turn, end_turn
        )
        bpm_processor = _BpmProcessor(
            start_turn, end_turn, tolerance,
            resonances_freqs, spectr_outdir,
            plane, position, name, samples
        )
        bpm_processor.do_bpm_analysis()
        results.append(bpm_processor)
    return results
###############################################################


class DriveMatrix(DriveAbstract):
    def __init__(self,
                 bpm_names,
                 bpm_matrix,
                 tunes,
                 plane,
                 output_file,
                 model_path,
                 nattunes=None,
                 tolerance=DEF_TUNE_TOLERANCE,
                 start_turn=0,
                 end_turn=None,
                 sequential=False):
        super(DriveMatrix, self).__init__(
            tunes,
            plane,
            nattunes,
            tolerance,
            start_turn,
            end_turn,
            sequential,
        )
        self._bpm_names = bpm_names
        self._bpm_matrix = bpm_matrix
        self._model_path = model_path
        self._output_filename = os.path.basename(output_file)
        self._output_dir = os.path.dirname(output_file)
        self._spectr_outdir = os.path.join(
            self._output_dir, "BPM"
        )

    def _get_outfile_name(self, plane):
        return self._output_filename

    def _do_analysis(self):
        model = metaclass.twiss(self._model_path)
        pool = multiprocessing.Pool(PROCESSES)
        for bpm_index in range(len(self._bpm_names)):
            bpm_name = self._bpm_names[bpm_index]
            bpm_row = self._bpm_matrix[bpm_index]
            try:
                bpm_position = model.S[model.indx[bpm_name]]
            except KeyError:
                LOGGER.debug("Cannot find" + bpm_name + "in model.")
                continue
            self._launch_bpm_row_analysis(bpm_position, bpm_name,
                                          bpm_row, pool)
        pool.close()
        pool.join()

    def _launch_bpm_row_analysis(self, bpm_position,
                                 bpm_name, bpm_row, pool):
        args = (self._plane, bpm_name, bpm_row, bpm_position,
                self._start_turn, self._end_turn, self._tolerance,
                self._resonances_freqs, self._spectr_outdir)
        if self._sequential:
            self._bpm_processors.extend(_analyze_bpm_samples(*args))
        else:
            pool.apply_async(
                _analyze_bpm_samples,
                args,
                callback=self._bpm_processors.extend
            )


# Global space ################################################
def _analyze_bpm_samples(bpm_plane, bpm_name, bpm_samples, bpm_position,
                         start_turn, end_turn, tolerance,
                         resonances_freqs, spectr_outdir):
    """
    This function triggers the per BPM data processing.
    It has to be outside of the classes to make it pickable for
    the multiprocessing module.
    """
    results = []
    LOGGER.debug("Staring process for " + bpm_name)
    bpm_processor = _BpmProcessor(
        start_turn, end_turn, tolerance,
        resonances_freqs, spectr_outdir,
        bpm_plane, bpm_position, bpm_name, bpm_samples
    )
    bpm_processor.do_bpm_analysis()
    results.append(bpm_processor)
    return results
###############################################################

class DriveSvd(DriveAbstract):
    def __init__(self,
                 bpm_names,
                 bpm_matrix,
                 usv,
                 tunes,
                 plane,
                 output_file,
                 model_path,
                 nattunes=None,
                 tolerance=DEF_TUNE_TOLERANCE,
                 start_turn=0,
                 end_turn=None,
                 sequential=False,
                 fast=False):
        super(DriveSvd, self).__init__(
            tunes,
            plane,
            nattunes,
            tolerance,
            start_turn,
            end_turn,
            sequential,
        )
        self._bpm_names = bpm_names
        self._bpms_matrix = bpm_matrix  
        self._usv = usv
        self._model_path = model_path
        self._output_filename = os.path.basename(output_file)
        self._output_dir = os.path.dirname(output_file)
        self._spectr_outdir = os.path.join(
            self._output_dir, "BPM"
        )
        self._fast=fast


    def _get_outfile_name(self, plane):
        return self._output_filename

    def _do_analysis(self):
        USV = self._usv
        SV = np.dot(np.diag(USV[1]), USV[2])
        if self._fast:
            number_of_harmonics = 300
            freqs = _laskar_per_mode(np.mean(SV,axis=0), number_of_harmonics)
        else:
            number_of_harmonics = 100
            pool = multiprocessing.Pool(np.min([PROCESSES, SV.shape[0]]))
            freqs = []
            for i in range(SV.shape[0]):
                args = (SV[i, :], number_of_harmonics)
                if self._sequential:
                    freqs.extend(_laskar_per_mode(*args))
                else:
                    pool.apply_async(_laskar_per_mode, args, callback=freqs.extend)
            pool.close()
            pool.join()
        frequencies = np.array(freqs)
        svd_coefficients = self.compute_coefs_for_freqs(SV, frequencies)
        bpms_coefficients = np.dot(USV[0], svd_coefficients)

        model = metaclass.twiss(self._model_path)
        pool = multiprocessing.Pool(PROCESSES)
        for bpm_index in range(len(self._bpm_names)):
            bpm_name = self._bpm_names[bpm_index]
            bpm_coefficients = bpms_coefficients[bpm_index, :]
            bpm_samples = self._bpms_matrix[bpm_index, :]
            try:
                bpm_position = model.S[model.indx[bpm_name]]
            except KeyError:
                LOGGER.debug("Cannot find" + bpm_name + "in model.")
                continue
            args = (self._plane, bpm_name, bpm_coefficients, frequencies, bpm_samples, bpm_position,
                    self._start_turn, self._end_turn, self._tolerance,
                    self._resonances_freqs, self._spectr_outdir)
            if self._sequential:
                self._bpm_processors.append(_analyze_bpm_samples_svd(*args))
            else:
                pool.apply_async(
                    _analyze_bpm_samples_svd,
                    args,
                    callback=self._bpm_processors.append
                )
        pool.close()
        pool.join()

    def compute_coefs_for_freqs(self, samples, freqs):
        n = samples.shape[1]
        coefficients = np.dot(samples, np.exp(-PI2I * np.outer(np.arange(n), freqs))) / n
        return coefficients


 # Global space ################################################
def _analyze_bpm_samples_svd(bpm_plane, bpm_name, bpm_coefficients, frequencies, bpm_samples, bpm_position,
                             start_turn, end_turn, tolerance,
                             resonances_freqs, spectr_outdir):
    bpm_processor = _BpmProcessor(
        start_turn, end_turn, tolerance,
        resonances_freqs, spectr_outdir,
        bpm_plane, bpm_position, bpm_name, None
    )
    resonances = bpm_processor.resonance_search(frequencies, bpm_coefficients)
    bpm_processor.harmonic_analysis = HarmonicAnalysis(bpm_samples)
    bpm_processor.get_bpm_results(resonances, frequencies, bpm_coefficients)
    bpm_processor.write_bpm_spectrum(bpm_name, bpm_plane, np.abs(bpm_coefficients), frequencies)
    return bpm_processor


def _laskar_per_mode(sv, number_of_harmonics):
    har_analysis = HarmonicAnalysis(sv)
    freqs, _ = har_analysis.laskar_method(number_of_harmonics)
    return freqs
###########################################################


class _BpmProcessor(object):
    def __init__(self, start_turn, end_turn, tolerance,
                 resonances_freqs, spectr_outdir,
                 plane, position, name, samples):
        self._start_turn = start_turn
        self._end_turn = end_turn
        self._tolerance = tolerance
        self._spectr_outdir = spectr_outdir
        self._plane = plane
        self._name = name
        self._position = position
        self._main_resonance = MAIN_LINES[self._plane]
        self._resonances_freqs = resonances_freqs[self._plane]
        self._samples = samples
        self.harmonic_analysis = None
        self.bpm_results = None

    def do_bpm_analysis(self):
        self.harmonic_analysis = HarmonicAnalysis(self._samples)
        if DEFAULT_DFT_METHOD == "laskar":
            frequencies, coefficients = self.harmonic_analysis.laskar_method(
                NUM_HARMS
            )
        elif DEFAULT_DFT_METHOD == "fft":
            frequencies, coefficients = self.harmonic_analysis.fft_method(
                NUM_HARMS
            )
        resonances = self.resonance_search(
            frequencies, coefficients,
        )
        self.write_bpm_spectrum(self._name, self._plane,
                                 np.abs(coefficients), frequencies)
        LOGGER.debug("Done: " + self._name + ", plane:" + self._plane)
        self.get_bpm_results(resonances, frequencies, coefficients)

    def get_coefficient_for_freq(self, freq):
        return self.harmonic_analysis.get_coefficient_for_freq(freq)

    def get_bpm_results(self, resonances, frequencies, coefficients):
        try:
            tune, main_coefficient = resonances[self._main_resonance]
        except KeyError:
            LOGGER.debug("Cannot find main resonance for" + self._name +
                         "in plane" + self._plane)
            return None
        amplitude = np.abs(main_coefficient)
        phase = np.angle(main_coefficient)

        bpm_results = _BpmResults(self)
        bpm_results.tune = tune
        bpm_results.phase = phase / (2 * np.pi)
        bpm_results.amplitude = amplitude
        bpm_results.frequencies = frequencies
        bpm_results.coefficients = coefficients
        bpm_results.resonances = resonances
        bpm_results.peak_to_peak = self.harmonic_analysis.peak_to_peak
        bpm_results.closed_orbit = self.harmonic_analysis.closed_orbit
        bpm_results.closed_orbit_rms = self.harmonic_analysis.closed_orbit_rms
        self.bpm_results = bpm_results

    @staticmethod
    def _compute_bpm_samples(bpm_samples_str, start_turn, end_turn):
        data_length = len(bpm_samples_str)
        if end_turn is not None and end_turn < data_length:
            end_index = end_turn
        else:
            end_index = data_length
        return np.array(
            [float(sample) for sample in bpm_samples_str[start_turn:end_index]]
        )

    def write_bpm_spectrum(self, bpm_name, bpm_plane, amplitudes, freqs):
        file_name = bpm_name + "." + bpm_plane.lower()
        spectr_outfile = tfs_file_writer.TfsFileWriter(
            os.path.join(self._spectr_outdir, file_name)
        )
        spectr_outfile.add_column_names(SPECTR_COLUMN_NAMES)
        spectr_outfile.add_column_datatypes(["%le"] * (len(SPECTR_COLUMN_NAMES)))
        for index, amplitude in enumerate(amplitudes):
            spectr_outfile.add_table_row([freqs[index], amplitude])
        spectr_outfile.write_to_file()

    def resonance_search(self, frequencies, coefficients):
        np_frequencies = np.array(frequencies)
        np_coefficients = np.array(coefficients)
        found_resonances = {}
        bins = [((resonance_freq - self._tolerance,
                  resonance_freq + self._tolerance), resonance)
                for resonance, resonance_freq in self._resonances_freqs.iteritems()]
        for bin, resonance in bins:
            min, max = bin
            indices = np.where((np_frequencies >= min) & (np_frequencies <= max))[0]
            if len(indices) == 0:
                continue
            max_index = indices[np.argmax(np.abs(np_coefficients[indices]))]
            found_resonances[resonance] = (np_frequencies[max_index], np_coefficients[max_index])
            # TODO: Is it right to remove already used lines? I dont think so...:
            np_frequencies[max_index], np_coefficients[max_index] = -100000., 0.
        return found_resonances


class _BpmResults(object):

    def __init__(self, bpm_processor):
        self.name = bpm_processor._name
        self.position = bpm_processor._position
        self.tune = None
        self.phase = None
        self.avphase = None
        self.amplitude = None
        self.frequencies = None
        self.coefficients = None
        self.bpm_processor = None
        self.bpm_resolution = -1.
        self.resonances = None
        self.peak_to_peak = None
        self.closed_orbit = None
        self.closed_orbit_rms = None
