from __future__ import print_function
import sys
import os
import multiprocessing
import numpy as np
from harmonic_analysis import HarmonicAnalysis


if "win" not in sys.platform:
    sys.path.append("/afs/cern.ch/work/j/jcoellod/public/Beta-Beat.src")
else:
    sys.path.append("\\\\AFS\\cern.ch\\work\\j\\jcoellod\\public\\Beta-Beat.src")

from Python_Classes4MAD import metaclass  # noqa
from Utilities import tfs_file_writer  # noqa
from Utilities import iotools  # noqa

PI2I = 2 * np.pi * complex(0, 1)

HEADERS = {"X": ["NAME", "S", "BINDEX", "SLABEL", "TUNEX",
                 "NOISE", "PK2PK", "CO", "CORMS", "AMPX",
                 "MUX", "AVG_AMPX", "AVG_MUX"],
           "Y": ["NAME", "S", "BINDEX", "SLABEL", "TUNEY",
                 "NOISE", "PK2PK", "CO", "CORMS",
                 "AMPY", "MUY", "AVG_AMPY", "AVG_MUY"]}

SPECTR_COLUMN_NAMES = ["FREQ", "AMP"]

RESONANCE_LISTS = {"X": ((1, 0, 0), (0, 1, 0), (-2, 0, 0), (0, 2, 0), (-3, 0, 0), (-1, -1, 0),
                         (2, -2, 0), (0, -2, 0), (1, -2, 0), (-1, 3, 0), (1, 2, 0), (-2, 1, 0),
                         (1, 1, 0), (2, 0, 0), (-1, -2, 0), (3, 0, 0), (0, 0, 1)),
                   "Y": ((0, 1, 0), (1, 0, 0), (-1, 1, 0), (-2, 0, 0), (1, -1, 0), (0, -2, 0),
                         (0, -3, 0), (-1, 1, 0), (2, 1, 0), (-1, 3, 0), (1, 1, 0), (-1, 2, 0),
                         (0, 0, 1))}

MAIN_LINES = {"X": (1, 0, 0),
              "Y": (0, 1, 0)}

N_TO_P = {"0": "X",
          "1": "Y"}

DEF_TUNE_TOLERANCE = 0.001

NUM_HARMS = 300

PROCESSES = multiprocessing.cpu_count()
DEBUG = False


class DriveAbstract(object):
    def __init__(self,
                 tunes,
                 nattunes=None,
                 tolerance=DEF_TUNE_TOLERANCE,
                 start_turn=0,
                 end_turn=None,
                 sequential=False):
        self._tunes = tunes
        self._nattunes = nattunes
        self._tolerance = tolerance
        self._start_turn = start_turn
        self._end_turn = end_turn
        self._sequential = sequential
        self._compute_resonances_freqs()

    # Methods to override in subclasses:
    def _do_analysis(self):
        raise NotImplementedError("Dont instantiate this abstract class!")

    def _get_outfile_name(self, plane):
        raise NotImplementedError("Dont instantiate this abstract class!")
    ######

    def _compute_resonances_freqs(self):
        """
        Computes the frequencies for all the resonances listed in the
        constante RESONANCE_LISTS, together with the natural tunes
        frequencies if given.
        """
        tune_x, tune_y, tune_z = self._tunes
        self._resonances_freqs = {}
        for plane in ("X", "Y"):
            freqs = [(resonance_h * tune_x) +
                     (resonance_v * tune_y) +
                     (resonance_l * tune_z)
                     for (resonance_h,
                          resonance_v,
                          resonance_l) in RESONANCE_LISTS[plane]]
            # Move to [0, 1] domain.
            freqs = [freq + 1. if freq < 0. else freq for freq in freqs]
            self._resonances_freqs[plane] = dict(
                zip(RESONANCE_LISTS[plane], freqs)
            )
        if self._nattunes is not None:
            nattune_x, nattune_y, _ = self._nattunes  # TODO: nattunez?
            if nattune_x is not None:
                self._resonances_freqs["X"]["NATX"] = nattune_x
            if nattune_y is not None:
                self._resonances_freqs["Y"]["NATY"] = nattune_y

    def start_analysis(self):
        self._bpm_processors = []
        self._create_lin_files()
        iotools.create_dirs(self._spectr_outdir)
        self._do_analysis()
        self._write_full_results()

    def _create_lin_files(self):
        self._lin_outfiles = {}
        for plane in "X", "Y":
            file_name = self._get_outfile_name(plane)
            lin_outfile = tfs_file_writer.TfsFileWriter(
                os.path.join(self._output_dir, file_name)
            )
            headers = HEADERS[plane]
            for resonance in RESONANCE_LISTS[plane]:
                if resonance == MAIN_LINES[plane]:
                    continue
                x, y, z = resonance
                if z == 0:
                    resstr = (str(x) + str(y)).replace("-", "_")
                else:
                    resstr = (str(x) + str(y) + str(z)).replace("-", "_")
                headers.extend(["AMP" + resstr, "PHASE" + resstr])
            headers.extend(["NATTUNE" + plane,
                            "NATAMP" + plane])
            lin_outfile.add_column_names(headers)
            lin_outfile.add_column_datatypes(
                ["%s"] + ["%le"] * (len(headers) - 1))
            self._lin_outfiles[plane] = lin_outfile

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in xrange(0, len(l), n):  # noqa xrange doesn't exist in Python3
            yield l[i:i + n]

    def _write_full_results(self):
        for plane in ("X", "Y"):
            lin_outfile = self._lin_outfiles[plane]
            tune, rms_tune = self._compute_tune_stats(plane)
            for bpm_processor in self._bpm_processors:
                try:
                    bpm_results = bpm_processor.bpm_results
                except AttributeError:
                    continue
                if bpm_results.plane == plane:
                    (bpm_results.amp_from_avg,
                     bpm_results.phase_from_avg) = self._compute_from_avg(
                        tune,
                        bpm_processor
                    )
                    self._write_single_bpm_results(
                        lin_outfile,
                        bpm_results
                    )
            plane_number = "1" if plane == "X" else "2"
            lin_outfile.add_float_descriptor("Q" + plane_number, tune)
            lin_outfile.add_float_descriptor("Q" + plane_number + "RMS",
                                             rms_tune)
            lin_outfile.order_rows("S")
            lin_outfile.write_to_file()

    def _write_single_bpm_results(self, lin_outfile, bpm_results):
        row = [bpm_results.name, bpm_results.position, 0, 0, bpm_results.tune,
               0, bpm_results.peak_to_peak, bpm_results.closed_orbit,
               bpm_results.closed_orbit_rms, bpm_results.amplitude,
               bpm_results.phase, bpm_results.amp_from_avg,
               bpm_results.phase_from_avg]
        resonance_list = RESONANCE_LISTS[bpm_results.plane]
        main_resonance = MAIN_LINES[bpm_results.plane]
        for resonance in resonance_list:
            if resonance != main_resonance:
                if resonance in bpm_results.resonances:
                    _, coefficient = bpm_results.resonances[resonance]
                    row.append(np.abs(coefficient) / bpm_results.amplitude)
                    row.append(np.angle(coefficient) / (2 * np.pi))
                else:
                    row.append(0.0)
                    row.append(0.0)

        col_name = "NAT" + bpm_results.plane.upper()
        try:
            natural_coef = bpm_results.resonances[col_name]
            row.append(np.abs(natural_coef) / bpm_results.amplitude)
            row.append(np.angle(natural_coef) / (2 * np.pi))
        except KeyError:
            row.append(0.0)
            row.append(0.0)
        lin_outfile.add_table_row(row)

    def _compute_tune_stats(self, plane):
        tune_list = []
        for bpm_processor in self._bpm_processors:
            try:
                bpm_results = bpm_processor.bpm_results
            except AttributeError:
                continue
            if bpm_processor.bpm_results.plane == plane:
                tune_list.append(bpm_results.tune)
        return np.mean(tune_list), np.std(tune_list)

    def _compute_from_avg(self, tune, bpm_results):
        coef = bpm_results.get_coefficient_for_freq(tune)
        return np.abs(coef), np.angle(coef) / (2 * np.pi)


class DriveFile(DriveAbstract):

    def __init__(self,
                 input_file,
                 tunes,
                 nattunes=None,
                 tolerance=DEF_TUNE_TOLERANCE,
                 start_turn=0,
                 end_turn=None,
                 output_dir=None,
                 sequential=False):
        super(DriveFile, self).__init__(
            tunes,
            nattunes=None,
            tolerance=DEF_TUNE_TOLERANCE,
            start_turn=0,
            end_turn=None,
            sequential=False
        )
        self._input_file = input_file
        if output_dir is None:
            self._output_dir = os.path.dirname(input_file)
        else:
            self._output_dir = output_dir
        self._spectr_outdir = os.path.join(
            self._output_dir, "BPM"
        )

    def _get_outfile_name(self, plane):
        return os.path.basename(self._input_file) + "_lin" + plane.lower()

    def _do_analysis(self):
        lines = []
        with open(self._input_file, "r") as records:
            for line in records:
                bpm_data = line.split()
                if bpm_data[0] in ("0", "1"):
                    lines.append(line.split())
                else:
                    continue
        pool = multiprocessing.Pool(PROCESSES)
        num_of_chunks = int(len(lines) / PROCESSES) + 1
        for bpm_datas in DriveAbstract.chunks(lines, num_of_chunks):
            self._launch_bpm_chunk_analysis(bpm_datas, pool)
        pool.close()
        pool.join()

    def _launch_bpm_chunk_analysis(self, bpm_datas, pool):
        args = (self._start_turn, self._end_turn, self._tolerance,
                self._resonances_freqs, self._spectr_outdir, bpm_datas)
        if self._sequential:
            self._bpm_processors.extend(_analyze_bpm_chunk(*args))
        else:
            pool.apply_async(
                _analyze_bpm_chunk,
                args,
                callback=self._bpm_processors.extend
            )


# Global space ################################################
def _analyze_bpm_chunk(start_turn, end_turn, tolerance,
                       resonances_freqs, spectr_outdir, bpm_datas):
    """
    This function triggers the per BPM data processing.
    It has to be outside of the classes to make it pickable for
    the multiprocessing module.
    """
    results = []
    if DEBUG:
        print("Staring process with chunksize", len(bpm_datas))
    for bpm_data in bpm_datas:
        plane = N_TO_P[bpm_data.pop(0)]
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
                 bpm_matrix_x,
                 bpm_matrix_y,
                 tunes,
                 output_dir,
                 model_path,
                 nattunes=None,
                 tolerance=DEF_TUNE_TOLERANCE,
                 start_turn=0,
                 end_turn=None,
                 sequential=False):
        super(DriveFile, self).__init__(
            tunes,
            nattunes=None,
            tolerance=DEF_TUNE_TOLERANCE,
            start_turn=0,
            end_turn=None,
            sequential=False
        )
        self._bpm_names = bpm_names
        self._bpm_matrices = {"X": bpm_matrix_x,
                              "Y": bpm_matrix_y}
        self._model_path = model_path
        self._spectr_outdir = os.path.join(
            self._output_dir, "BPM"
        )

    def _get_outfile_name(self, plane):
        return "harmonics_" + plane.lower() + ".dat"

    def _do_analysis(self):
        model = metaclass.twiss(self._model_path)
        pool = multiprocessing.Pool(PROCESSES)
        for bpm_index in range(len(self._bpm_names)):
            for plane in ("X", "Y"):
                bpm_name = self._bpm_names[bpm_index]
                bpm_row = self._bpm_matrix[bpm_index]
                bpm_position = model.S[model.indx[bpm_name]]
                self._launch_bpm_row_analysis(plane, bpm_position, bpm_name,
                                              bpm_row, pool)
        pool.close()
        pool.join()

    def _launch_bpm_row_analysis(self, plane, bpm_position,
                                 bpm_name, bpm_row, pool):
        args = (plane, bpm_name, bpm_row, bpm_position,
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
    if DEBUG:
        print("Staring process for ", bpm_name)
    bpm_processor = _BpmProcessor(
        start_turn, end_turn, tolerance,
        resonances_freqs, spectr_outdir,
        bpm_plane, bpm_position, bpm_name, bpm_samples
    )
    bpm_processor.do_bpm_analysis()
    results.append(bpm_processor)
    return results
###############################################################


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
        self._harmonic_analysis = HarmonicAnalysis(samples)

    def do_bpm_analysis(self):
        frequencies, coefficients = self._harmonic_analysis.laskar_method(
            NUM_HARMS
        )
        resonances = self._resonance_search(
            frequencies, coefficients,
        )
        self._write_bpm_spectrum(self._name, self._plane,
                                 np.abs(coefficients), frequencies)
        if DEBUG:
            print("Done:", self._name, ", plane:", self._plane)
        self._get_bpm_results(resonances, frequencies, coefficients)

    def get_coefficient_for_freq(self, freq):
        return self._harmonic_analysis.get_coefficient_for_freq(freq)

    def _get_bpm_results(self, resonances, frequencies, coefficients):
        try:
            tune, main_coefficient = resonances[self._main_resonance]
        except KeyError:
            print("Cannot find main resonance for", self._name,
                  "in plane", self._plane)
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
        bpm_results.peak_to_peak = self._harmonic_analysis.peak_to_peak
        bpm_results.closed_orbit = self._harmonic_analysis.closed_orbit
        bpm_results.closed_orbit_rms = self._harmonic_analysis.closed_orbit_rms
        self.bpm_results = bpm_results

    @staticmethod
    def _compute_bpm_samples(bpm_samples_str, start_turn, end_turn):
        data_length = len(bpm_samples_str)
        if (end_turn is not None and end_turn < data_length):
            end_index = end_turn
        else:
            end_index = data_length
        return np.array(
            [float(sample) for sample in bpm_samples_str[start_turn:end_index]]
        )

    def _write_bpm_spectrum(self, bpm_name, bpm_plane, amplitudes, freqs):
        file_name = bpm_name + "." + bpm_plane.lower()
        spectr_outfile = tfs_file_writer.TfsFileWriter(
            os.path.join(self._spectr_outdir, file_name)
        )
        spectr_outfile.add_column_names(SPECTR_COLUMN_NAMES)
        spectr_outfile.add_column_datatypes(["%le"] * (len(SPECTR_COLUMN_NAMES)))
        for i in range(len(amplitudes)):
            spectr_outfile.add_table_row([freqs[i], amplitudes[i]])
        spectr_outfile.write_to_file()

    def _resonance_search(self, frequencies, coefficients):
        found_resonances = {}
        sorted_coefficients, sorted_frequencies = zip(*sorted(zip(coefficients, frequencies),
                                                              key=lambda tuple: np.abs(tuple[0]),
                                                              reverse=True))
        for index in range(len(sorted_frequencies)):
            coefficient = sorted_coefficients[index]
            frequency = sorted_frequencies[index]
            for resonance, resonance_freq in self._resonances_freqs.iteritems():
                min_freq = resonance_freq - self._tolerance
                max_freq = resonance_freq + self._tolerance
                if (frequency >= min_freq and
                        frequency <= max_freq and
                        resonance not in found_resonances):
                    found_resonances[resonance] = (frequency, coefficient)
                    break
        return found_resonances


class _BpmResults(object):

    def __init__(self, bpm_processor):
        self.name = bpm_processor._name
        self.position = bpm_processor._position
        self.plane = bpm_processor._plane
        self.tune = None
        self.phase = None
        self.avphase = None
        self.amplitude = None
        self.frequencies = None
        self.coefficients = None
        self.bpm_processor = None
