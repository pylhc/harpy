from __future__ import print_function
import sys
import os
import multiprocessing
import numpy as np
from harmonic_analysis import HarmonicAnalisys

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
                 "MUX", "AVG_MUX"],
           "Y": ["NAME", "S", "BINDEX", "SLABEL", "TUNEY",
                 "NOISE", "PK2PK", "CO", "CORMS",
                 "AMPY", "MUY", "AVG_MUY"]}

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


class Drive():

    def __init__(self,
                 input_file,
                 tunes,
                 nattunes=None,
                 tolerance=DEF_TUNE_TOLERANCE,
                 start_turn=0,
                 end_turn=None,
                 output_dir=None,
                 sequential=False):
        self._input_file = input_file
        if output_dir is None:
            self._output_dir = os.path.dirname(input_file)
        else:
            self._output_dir = output_dir
        self._spectr_outdir = os.path.join(
            self._output_dir, "BPM"
        )
        self._tunes = tunes
        self._nattunes = nattunes
        self._tolerance = tolerance
        self._start_turn = start_turn
        self._end_turn = end_turn
        self._sequential = sequential
        self._bpm_processor_list = None
        self._compute_resonances_freqs()
        self._analyze_tbt_data()

    def _analyze_tbt_data(self):
        (lin_outfile_x,
         lin_outfile_y) = self._create_lin_files()
        self._lin_outfiles = {"X": lin_outfile_x,
                              "Y": lin_outfile_y}
        iotools.create_dirs(self._spectr_outdir)
        self._read_input_file()
        full_results = self._analyze_all_bpms()
        self._write_full_results(full_results)

    def _write_full_results(self, full_results):
        for plane in ["X", "Y"]:
            lin_outfile = self._lin_outfiles[plane]
            tune, rms_tune = self._compute_tune_stats(full_results[plane])
            for bpm_results in full_results[plane]:
                exponents = np.exp(-PI2I * tune * np.arange(len(bpm_results.samples)))
                bpm_results.avphase = np.angle(np.sum(exponents * bpm_results.samples)) / (2 * np.pi)
                self._write_single_bpm_results(lin_outfile, bpm_results)
            plane_number = "1" if plane == "X" else "2"
            lin_outfile.add_float_descriptor("Q" + plane_number, tune)
            lin_outfile.add_float_descriptor("Q" + plane_number + "RMS", rms_tune)
            lin_outfile.write_to_file()

    def _write_single_bpm_results(self, lin_outfile, bpm_results):
        row = [bpm_results.name, bpm_results.position, 0, 0, bpm_results.tune,
               0, bpm_results.peak_to_peak, bpm_results.closed_orbit,
               bpm_results.closed_orbit_rms, bpm_results.amplitude, bpm_results.phase,
               bpm_results.avphase]
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

        if self._nattunes is not None:
            natural_coef = bpm_results.resonances["NAT" + bpm_results.plane.upper()]
            row.append(np.abs(natural_coef) / bpm_results.amplitude)
            row.append(np.angle(natural_coef) / (2 * np.pi))
        else:
            row.append(0.0)
            row.append(0.0)
        lin_outfile.add_table_row(row)

    def _compute_tune_stats(self, bpm_results):
        tune_list = []
        for results in bpm_results:
            tune_list.append(results.tune)
        return np.mean(tune_list), np.std(tune_list)

    def _compute_resonances_freqs(self):
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
            self._resonances_freqs["X"]["NATX"] = nattune_x
            self._resonances_freqs["Y"]["NATY"] = nattune_y

    def _create_lin_files(self):
        lin_outfiles = []
        for plane in "X", "Y":
            file_name = os.path.basename(self._input_file) + "_lin" + plane.lower()
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
                            "NATAMPX" + plane])
            lin_outfile.add_column_names(headers)
            lin_outfile.add_column_datatypes(["%s"] + ["%le"] * (len(headers) - 1))
            lin_outfiles.append(lin_outfile)
        return lin_outfiles

    def _read_input_file(self):
        self._bpm_processor_list = []
        with open(self._input_file, "r") as records:
            for line in records:
                bpm_data = line.split()
                try:
                    bpm_processor = _BpmProcessor(self, bpm_data)
                except KeyError:
                    continue  # Comments and empty lines
                self._bpm_processor_list.append(bpm_processor)

    def _analyze_all_bpms(self):
        full_results = {"X": [], "Y": []}
        pool = multiprocessing.Pool()
        for bpm_processor in self._bpm_processor_list:
            full_plane_results = full_results[bpm_processor._plane]
            self._launch_single_bpm_analysis(bpm_processor,
                                             full_plane_results,
                                             pool)
        pool.close()
        pool.join()
        return full_results

    def _launch_single_bpm_analysis(self, bpm_processor, results, pool):
        if self._sequential:
            results.append(
                _BpmProcessor.do_bpm_analysis(bpm_processor)
            )
        else:
            pool.apply_async(
                _analyze_single_bpm,
                (bpm_processor, ),
                callback=results.append
            )


# Global space ################################################
def _analyze_single_bpm(bpm_processor):
    """
    This function triggers the per BPM data processing.
    It has to be outside of the classes to make it pickable for
    the multiprocessing module.
    """
    return bpm_processor.do_bpm_analysis()
###############################################################


class _BpmProcessor(object):
    def __init__(self, drive, bpm_data):
        self._drive = drive
        self._bpm_data = bpm_data
        self._plane = N_TO_P[bpm_data.pop(0)]
        self._name = bpm_data.pop(0)
        self._position = bpm_data.pop(0)
        self._samples = self._compute_bpm_samples(bpm_data)
        self._main_resonance = MAIN_LINES[self._plane]
        self._resonances_freqs = drive._resonances_freqs[self._plane]

    def do_bpm_analysis(self):
        harmonic_analysis = HarmonicAnalisys(self._samples)
        frequencies, coefficients = harmonic_analysis.laskar_method(
            NUM_HARMS
        )
        resonances = self._resonance_search(
            frequencies, coefficients,
        )
        self._write_bpm_spectrum(self._name, self._plane,
                                 np.abs(coefficients), frequencies)
        if self._drive._sequential:
            print("Done:", self._name, ", plane:", self._plane)
        return self._get_bpm_results(resonances, frequencies, coefficients)

    def _get_bpm_results(self, resonances, frequencies, coefficients):
        tune, main_coefficient = resonances[self._main_resonance]
        amplitude = np.abs(main_coefficient)
        phase = np.angle(main_coefficient)

        bpm_results = _BpmResults(self)
        bpm_results.tune = tune
        bpm_results.phase = phase / (2 * np.pi)
        bpm_results.amplitude = amplitude
        bpm_results.frequencies = frequencies
        bpm_results.coefficients = coefficients
        bpm_results.resonances = resonances
        bpm_results.compute_orbit(self._samples)
        bpm_results.samples = (self._samples - np.average(self._samples))
        return bpm_results

    def _compute_bpm_samples(self, bpm_data):
        data_length = len(bpm_data)
        if (self._drive._end_turn is not None and self._drive._end_turn < data_length):
            end_index = self._drive._end_turn
        else:
            end_index = data_length
        return np.array(
            [float(sample) for sample in bpm_data[self._drive._start_turn:end_index]]
        )

    def _write_bpm_spectrum(self, bpm_name, bpm_plane, amplitudes, freqs):
        file_name = bpm_name + "." + bpm_plane.lower()
        spectr_outfile = tfs_file_writer.TfsFileWriter(
            os.path.join(self._drive._spectr_outdir, file_name)
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
                min_freq = resonance_freq - self._drive._tolerance
                max_freq = resonance_freq + self._drive._tolerance
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
        self.samples = None

    def compute_orbit(self, samples):
        self.closed_orbit = np.mean(samples)
        self.closed_orbit_rms = np.std(samples)
        self.peak_to_peak = np.max(samples) - np.min(samples)
