
import sys
import os
import numpy as np
import jacobsen
import time
#from matplotlib import pyplot
from multiprocessing import Pool

if "win" not in sys.platform:
    #sys.path.append("/afs/cern.ch/work/j/jcoellod/public/Beta-Beat.src")
    sys.path.append("/afs/cern.ch/work/l/lmalina/Beta-Beat.src")
else:
    sys.path.append("\\\\AFS\\cern.ch\\work\\j\\jcoellod\\public\\Beta-Beat.src")

from Python_Classes4MAD import metaclass
from Python_Classes4MAD.SDDSIlya import SDDSReader
from drive import drive_runner
from Utilities import tfs_file_writer
PI2I = 2 * np.pi * complex(0, 1)

RESONANCE_LIST_X = [(1, 0), (0, 1), (-2, 0), (0, 2), (-3, 0), (-1, -1), (2, -2), (0, -2), (1, -2), (-1, 3), (1, 2), (-2, 1), (1, 1), (2, 0), (-1, -2), (3, 0)]
RESONANCE_LIST_Y = [(0, 1), (1, 0), (-1, 1), (-2, 0), (1, -1), (0, -2), (0, -3), (-1, 1), (2, 1), (-1, 3), (1, 1), (-1, 2)]


def main():
    output_dir = "./"
    tune_x = 44
    tune_y = 39
    t_x = 0.44
    t_y = 0.39
    for i in range (1,2):
        input_sdds_file_path = "Kick_1503_1.46x0.34_" + str(i) #/afs/cern.ch/work/l/lmalina/ESRF/ATNoScans/
        turnNo=200 + 10*i
        print input_sdds_file_path,turnNo
        start = time.time()
        drive_runner.run_drive(input_sdds_file_path, 0, turnNo, tune_x, tune_y, nat_tune_x=tune_x, nat_tune_y=tune_y, clean_up=False, stdout=open("drive_output.txt", "w"))
        print "Drive took:", time.time() - start

        start = time.time()
        analyze_tbt_data(input_sdds_file_path, output_dir, t_x, t_y, turnNo)
        print "You took:", time.time() - start
        


def analyze_tbt_data(input_sdds_file_path, output_dir, tune_x, tune_y, turnNo):
    raw_sdds_data = SDDSReader(input_sdds_file_path)
    pool = Pool()

    linx_outfile = _create_lin_file(input_sdds_file_path+"b_linx", "x")
    liny_outfile = _create_lin_file(input_sdds_file_path+"b_liny", "y")
    bpm_results_x = []
    bpm_results_y = []
    tune_tolerance = 0.01

    for bpm_data in raw_sdds_data.records:
        if bpm_data[0] == "0":
            pool.apply_async(process_single_bpm, (bpm_data, tune_x, tune_y, tune_tolerance, turnNo), callback=lambda results: _append_single_bpm_results(results, bpm_results_x))
            #_write_single_bpm_results(linx_outfile, apply(process_single_bpm, (bpm_data, tune_x, tune_y, tune_tolerance,turnNo)), bpm_results_x)
        elif bpm_data[0] == "1":
            pool.apply_async(process_single_bpm, (bpm_data, tune_x, tune_y, tune_tolerance, turnNo), callback=lambda results: _append_single_bpm_results(results, bpm_results_y))
            #_write_single_bpm_results(liny_outfile, apply(process_single_bpm, (bpm_data, tune_x, tune_y, tune_tolerance,turnNo)), bpm_results_y)
    pool.close()
    pool.join()
    
    tune_x, rms_tune_x = _compute_tune_stats(bpm_results_x)
    tune_y, rms_tune_y = _compute_tune_stats(bpm_results_y)
    for bpm in bpm_results_x:
        exponents = np.exp(-PI2I * tune_x * np.arange(len(bpm.samples)))
        bpm.avphase=np.angle(np.sum(exponents*bpm.samples))/ (2 * np.pi)
        _write_single_bpm_results(linx_outfile, bpm)
    for bpm in bpm_results_y:
        exponents = np.exp(-PI2I * tune_y * np.arange(len(bpm.samples)))
        bpm.avphase=np.angle(np.sum(exponents*bpm.samples))/ (2 * np.pi)
        _write_single_bpm_results(liny_outfile, bpm)
    linx_outfile.add_float_descriptor("Q1", tune_x)
    linx_outfile.add_float_descriptor("Q1RMS", rms_tune_x)
    liny_outfile.add_float_descriptor("Q2", tune_y)
    liny_outfile.add_float_descriptor("Q2RMS", rms_tune_y)

    linx_outfile.write_to_file()
    liny_outfile.write_to_file()


def process_single_bpm(bpm_data, tune_x, tune_y, tune_tolerance, turnNo):
    bpm_plane = bpm_data[0]
    bpm_name = bpm_data[1]
    bpm_position = bpm_data[2]
    bpm_samples = np.array(map(float, bpm_data[3:turnNo+3]))

    if bpm_plane == "0":
        main_resonance = (1, 0)
        resonance_list = RESONANCE_LIST_X
    elif bpm_plane == "1":
        main_resonance = (0, 1)
        resonance_list = RESONANCE_LIST_Y

    frequencies, coefficients = jacobsen.laskar_method(bpm_samples, 300)
    resonances = jacobsen.resonance_search(frequencies, coefficients,
                                           tune_x, tune_y, tune_tolerance, resonance_list)
    
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


def _write_single_bpm_results(lin_outfile, bpm_results):
    row = [bpm_results.name, bpm_results.position, 0, 0, bpm_results.tune, 0, bpm_results.peak_to_peak, bpm_results.closed_orbit, bpm_results.closed_orbit_rms, bpm_results.amplitude, bpm_results.phase, bpm_results.avphase]
    if bpm_results.plane == "0":
        resonance_list = RESONANCE_LIST_X
        main_resonance = (1, 0)
    elif bpm_results.plane == "1":
        resonance_list = RESONANCE_LIST_Y
        main_resonance = (0, 1)
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
        headers = ["NAME", "S", "BINDEX", "SLABEL", "TUNEX", "NOISE", "PK2PK", "CO", "CORMS", "AMPX", "MUX", "AVG_MUX", "AMP01", "PHASE01", "AMP_20", "PHASE_20", "AMP02", "PHASE02", "AMP_30", "PHASE_30", "AMP_1_1", "PHASE_1_1", "AMP2_2", "PHASE2_2", "AMP0_2", "PHASE0_2", "AMP1_2", "PHASE1_2", "AMP_13", "PHASE_13", "AMP12", "PHASE12", "AMP_21", "PHASE_21", "AMP11", "PHASE11", "AMP20", "PHASE20", "AMP_1_2", "PHASE_1_2", "AMP30", "PHASE30", "NATTUNEX", "NATAMPX"]
    elif plane.lower() == "y":
        headers = ["NAME", "S", "BINDEX", "SLABEL", "TUNEY", "NOISE", "PK2PK", "CO", "CORMS", "AMPY", "MUY", "AVG_MUY", "AMP10", "PHASE10", "AMP_1_1", "PHASE_1_1", "AMP_20", "PHASE_20", "AMP1_1", "PHASE1_1", "AMP0_2", "PHASE0_2", "AMP0_3", "PHASE0_3", "AMP_11", "PHASE_11", "AMP21", "PHASE21", "AMP_13", "PHASE_13", "AMP11", "PHASE11", "AMP_12", "PHASE_12", "NATTUNEY", "NATAMPY"]
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
    main()
