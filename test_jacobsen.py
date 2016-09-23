
import sys
import os
import numpy as np
import jacobsen
import time
from optparse import OptionParser
from multiprocessing import Pool

if "win" not in sys.platform:
    sys.path.append("/afs/cern.ch/work/j/jcoellod/public/Beta-Beat.src")
else:
    sys.path.append("\\\\AFS\\cern.ch\\work\\j\\jcoellod\\public\\Beta-Beat.src")

from Python_Classes4MAD import metaclass  # noqa
from Python_Classes4MAD.SDDSIlya import SDDSReader  # noqa
from Utilities import tfs_file_writer  # noqa
from Utilities import iotools  # noqa

PI2I = 2 * np.pi * complex(0, 1)

RESONANCE_LIST_X = [(1, 0, 0), (0, 1, 0), (-2, 0, 0), (0, 2, 0), (-3, 0, 0), (-1, -1, 0),
                    (2, -2, 0), (0, -2, 0), (1, -2, 0), (-1, 3, 0), (1, 2, 0), (-2, 1, 0),
                    (1, 1, 0), (2, 0, 0), (-1, -2, 0), (3, 0, 0), (0, 0, 1)]
RESONANCE_LIST_Y = [(0, 1, 0), (1, 0, 0), (-1, 1, 0), (-2, 0, 0), (1, -1, 0), (0, -2, 0),
                    (0, -3, 0), (-1, 1, 0), (2, 1, 0), (-1, 3, 0), (1, 1, 0), (-1, 2, 0),
                    (0, 0, 1)]


class RunModes:
    PARALLEL, SEQUENTIAL = range(2)
RUN_MODE = RunModes.PARALLEL


def _parse_args():
    parser = OptionParser()
    (_, args) = parser.parse_args()
    if len(args) == 2:
        return int(args[0]), int(args[1])
    else:
        return None, None


def main(turn_no_1, turn_no_2):
    output_dir = "./"
    tx = 0.268
    ty = 0.325
    tz = 0.00045
    input_sdds_file_path = "SVDFFT/Beam1@Turn@2016_06_10@02_17_15_001_0.sdds"
    if turn_no_1 is not None and turn_no_2 is not None:
        if turn_no_1 >= turn_no_2:
            raise ValueError("turn_no_1=" + str(turn_no_1) + " has to be lower than turn_no_2=" + str(turn_no_2))
    start = time.time()
    analyze_tbt_data(input_sdds_file_path, output_dir, tx, ty, tz, turn_no_1, turn_no_2)
    print "You took:", time.time() - start


def analyze_tbt_data(input_sdds_file_path, output_dir, tune_x, tune_y, tune_z, turn_no_1, turn_no_2):
    raw_sdds_data = SDDSReader(input_sdds_file_path)
    pool = Pool()

    linx_outfile = _create_lin_file(input_sdds_file_path + "_linx", "x")
    liny_outfile = _create_lin_file(input_sdds_file_path + "_liny", "y")
    spectr_outdir = os.path.join(os.path.dirname(input_sdds_file_path), "BPM")
    iotools.create_dirs(spectr_outdir)

    bpm_results_x = []
    bpm_results_y = []
    tune_tolerance = 0.01

    for bpm_data in raw_sdds_data.records:
        if bpm_data[0] == "0":
            bpm_results = bpm_results_x
        elif bpm_data[0] == "1":
            bpm_results = bpm_results_y
        else:
            continue
        if RUN_MODE == RunModes.PARALLEL:
            pool.apply_async(
                process_single_bpm,
                (bpm_data, tune_x, tune_y, tune_z, tune_tolerance,
                 turn_no_1, turn_no_2, spectr_outdir),
                callback=bpm_results.append
            )
        elif RUN_MODE == RunModes.SEQUENTIAL:
            _append_single_bpm_results(
                process_single_bpm(bpm_data, tune_x, tune_y, tune_z, tune_tolerance,
                                   turn_no_1, turn_no_2, spectr_outdir),
                bpm_results
            )
    pool.close()
    pool.join()

    tune_x, rms_tune_x = _compute_tune_stats(bpm_results_x)
    tune_y, rms_tune_y = _compute_tune_stats(bpm_results_y)
    for bpm in bpm_results_x:
        exponents = np.exp(-PI2I * tune_x * np.arange(len(bpm.samples)))
        bpm.avphase = np.angle(np.sum(exponents * bpm.samples)) / (2 * np.pi)
        _write_single_bpm_results(linx_outfile, bpm)
    for bpm in bpm_results_y:
        exponents = np.exp(-PI2I * tune_y * np.arange(len(bpm.samples)))
        bpm.avphase = np.angle(np.sum(exponents * bpm.samples)) / (2 * np.pi)
        _write_single_bpm_results(liny_outfile, bpm)
    linx_outfile.add_float_descriptor("Q1", tune_x)
    linx_outfile.add_float_descriptor("Q1RMS", rms_tune_x)
    liny_outfile.add_float_descriptor("Q2", tune_y)
    liny_outfile.add_float_descriptor("Q2RMS", rms_tune_y)

    linx_outfile.write_to_file()
    liny_outfile.write_to_file()


def process_single_bpm(bpm_data, tune_x, tune_y, tune_z, tune_tolerance,
                       turn_no_1, turn_no_2, spectr_outdir):
    bpm_plane = bpm_data.pop(0)
    bpm_name = bpm_data.pop(0)
    bpm_position = bpm_data.pop(0)

    data_length = len(bpm_data)
    start_index = turn_no_1 if turn_no_1 is not None else 0
    end_index = turn_no_2 + 1 if turn_no_2 is not None and turn_no_2 < data_length else data_length
    bpm_samples = np.array(map(float, bpm_data[start_index:end_index]))

    if bpm_plane == "0":
        main_resonance = (1, 0, 0)
        resonance_list = RESONANCE_LIST_X
    elif bpm_plane == "1":
        main_resonance = (0, 1, 0)
        resonance_list = RESONANCE_LIST_Y

    if _is_prime(len(bpm_samples)):
        # If we have a prime number of turns, use the frequency space method.
        frequencies, coefficients = jacobsen.laskar_method_freq_space(bpm_samples, 300)
    else:
        frequencies, coefficients = jacobsen.laskar_method_time_space(bpm_samples, 300)
    resonances = jacobsen.resonance_search(frequencies, coefficients,
                                           tune_x, tune_y, tune_z, tune_tolerance, resonance_list)

    _write_bpm_spectrum(spectr_outdir, bpm_name, bpm_plane, np.abs(coefficients), frequencies)
    tune, main_coefficient = resonances[main_resonance]
    amplitude = np.abs(main_coefficient)
    phase = np.angle(main_coefficient)

    bpm_results = BpmResults(bpm_name, bpm_position, bpm_plane)
    bpm_results.tune = tune
    bpm_results.phase = phase / (2 * np.pi)
    bpm_results.amplitude = amplitude
    bpm_results.frequencies = frequencies
    bpm_results.coefficients = coefficients
    bpm_results.resonances = resonances
    bpm_results.compute_orbit(bpm_samples)
    bpm_results.samples = (bpm_samples - np.average(bpm_samples))
    return bpm_results


def _is_prime(n):
    if n == 2 or n == 3:
        return True
    if n < 2 or n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    r = int(n**0.5)
    f = 5
    while f <= r:
        if n % f == 0:
            return False
        if n % (f + 2) == 0:
            return False
        f += 6
    return True


def _write_bpm_spectrum(spectr_outdir, bpm_name, bpm_plane, amplitudes, freqs):
    COLUMN_NAMES = ["FREQ", "AMP"]
    if bpm_plane == "0":
        suffix = "x"
    elif bpm_plane == "1":
        suffix = "y"
    file_name = bpm_name + "." + suffix
    spectr_outfile = tfs_file_writer.TfsFileWriter(os.path.join(spectr_outdir, file_name))
    spectr_outfile.add_column_names(COLUMN_NAMES)
    spectr_outfile.add_column_datatypes(["%le"] * (len(COLUMN_NAMES)))
    for i in range(len(amplitudes)):
        spectr_outfile.add_table_row([freqs[i], amplitudes[i]])
    spectr_outfile.write_to_file()


def _write_single_bpm_results(lin_outfile, bpm_results):
    row = [bpm_results.name, bpm_results.position, 0, 0, bpm_results.tune, 0, bpm_results.peak_to_peak, bpm_results.closed_orbit, bpm_results.closed_orbit_rms, bpm_results.amplitude, bpm_results.phase, bpm_results.avphase]
    if bpm_results.plane == "0":
        resonance_list = RESONANCE_LIST_X
        main_resonance = (1, 0, 0)
    elif bpm_results.plane == "1":
        resonance_list = RESONANCE_LIST_Y
        main_resonance = (0, 1, 0)
    for resonance in resonance_list:
        if resonance != main_resonance:
            if resonance in bpm_results.resonances:
                _, coefficient = bpm_results.resonances[resonance]
                row.append(np.abs(coefficient) / bpm_results.tune)
                row.append(np.angle(coefficient) / (2 * np.pi))
            else:
                row.append(0.0)
                row.append(0.0)

    # TODO: Natural tunes
    row.append(0.0)
    row.append(0.0)
    lin_outfile.add_table_row(row)


def _append_single_bpm_results(bpm_results, bpm_results_list):
    bpm_results_list.append(bpm_results)


def _create_lin_file(file_path, plane):
    lin_outfile = tfs_file_writer.TfsFileWriter(file_path)
    if plane.lower() == "x":
        headers = ["NAME", "S", "BINDEX", "SLABEL", "TUNEX", "NOISE", "PK2PK", "CO", "CORMS", "AMPX", "MUX", "AVG_MUX"]
        for resonance in RESONANCE_LIST_X:
            if resonance == (1, 0, 0):
                continue
            x, y, z = resonance
            if z == 0:
                resstr = (str(x) + str(y)).replace("-", "_")
            else:
                resstr = (str(x) + str(y) + str(z)).replace("-", "_")
            headers.extend(["AMP" + resstr, "PHASE" + resstr])
        headers.extend(["NATTUNEX", "NATAMPX"])

    elif plane.lower() == "y":
        headers = ["NAME", "S", "BINDEX", "SLABEL", "TUNEY", "NOISE", "PK2PK", "CO", "CORMS", "AMPY", "MUY", "AVG_MUY"]
        for resonance in RESONANCE_LIST_Y:
            if resonance == (0, 1, 0):
                continue
            x, y, z = resonance
            if z == 0:
                resstr = (str(x) + str(y)).replace("-", "_")
            else:
                resstr = (str(x) + str(y) + str(z)).replace("-", "_")
            headers.extend(["AMP" + resstr, "PHASE" + resstr])
        headers.extend(["NATTUNEY", "NATAMPY"])
    else:
        raise ValueError
    lin_outfile.add_column_names(headers)
    lin_outfile.add_column_datatypes(["%s"] + ["%le"] * (len(headers) - 1))
    return lin_outfile


def _compute_tune_stats(bpm_results):
    tune_list = []
    for results in bpm_results:
        tune_list.append(results.tune)
    return np.mean(tune_list), np.std(tune_list)


class BpmResults(object):

    def __init__(self, name, position, plane):
        self.name = name
        self.position = position
        self.plane = plane
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


if __name__ == "__main__":
    turn1, turn2 = _parse_args()
    main(turn1, turn2)
