import sys
import optparse
import logging
import drive
from drive import DriveFile, DriveMatrix, DriveSvd

LOGGER = logging.getLogger(__name__)


def _parse_args():
    parser = optparse.OptionParser()

    # Obligatory arguments ###########
    parser.add_option(
        "--file", help="Input file.",
        dest="input_file", type=str,
    )
    parser.add_option(
        "--tunex", help="Guess for the main horizontal tune.",
        dest="tunex", type=float,
    )
    parser.add_option(
        "--tuney", help="Guess for the main vertical tune.",
        dest="tuney", type=float,
    )
    ################################

    # Optional arguments ###########
    parser.add_option(
        "--nattunex", help="Guess for the natural horizontal tune.",
        dest="nattunex", type=float,
    )
    parser.add_option(
        "--nattuney", help="Guess for the natural vertical tune.",
        dest="nattuney", type=float,
    )
    parser.add_option(
        "--tunez", help="Guess for the main synchrotron tune.",
        default=0.0,
        dest="tunez", type=float,
    )
    parser.add_option(
        "--nattunez", help="Guess for the natural synchrotron tune.",
        default=0.0,
        dest="nattunez", type=float,
    )
    parser.add_option(
        "--tolerance", help="Tolerance on the guess for the tunes.",
        default=0.01,
        dest="tolerance", type=float,
    )
    parser.add_option(
        "--start_turn", help="Index of the first turn to use. Zero by default.",
        default=0,
        dest="start_turn", type=int,
    )
    parser.add_option(
        "--end_turn", help="Index of the first turn to ignore. By default the length of the data.",
        dest="end_turn", type=int,
    )
    parser.add_option(
        "--output", help="Output directory. The default is the input file directory.",
        dest="output_dir", type=float,
    )
    parser.add_option(
        "--sequential", help="If set, it will run in only one process.",
        dest="sequential", action="store_true",
    )
    ################################

    options, _ = parser.parse_args()
    return options


def _init_from_args():
    options = _parse_args()
    input_file = options.input_file
    tunes = (options.tunex,
             options.tuney,
             options.tunez)
    output_dir = options.output_dir
    nattunes = (options.nattunex,
                options.nattuney,
                options.nattunez)
    tolerance = options.tolerance
    start_turn = options.start_turn
    end_turn = options.end_turn
    sequential = options.sequential
    init_from_file(input_file, tunes, nattunes, tolerance,
                   start_turn, end_turn, output_dir, sequential)


def init_from_file(input_file, tunes, nattunes=None,
                   tolerance=drive.DEF_TUNE_TOLERANCE,
                   start_turn=0, end_turn=None,
                   output_dir=None, sequential=False):
    for plane in ("X", "Y"):
        drive_file = DriveFile(input_file, tunes, plane, nattunes,
                               tolerance,
                               start_turn, end_turn,
                               output_dir, sequential)
        drive_file.start_analysis()
        drive_file.write_full_results()


def init_from_matrix(bpm_names, bpm_matrix, tunes, plane,
                     output_file, model_path, nattunes=None,
                     tolerance=drive.DEF_TUNE_TOLERANCE,
                     start_turn=0, end_turn=None, sequential=False):
    drive_matrix = DriveMatrix(bpm_names, bpm_matrix, tunes, plane,
                               output_file, model_path, nattunes,
                               tolerance,
                               start_turn, end_turn, sequential)
    drive_matrix.start_analysis()
    return drive_matrix


def init_from_svd(bpm_names, bpm_data, usv, tunes, plane,
                     output_file, model_path, nattunes=None,
                     tolerance=drive.DEF_TUNE_TOLERANCE,
                     start_turn=0, end_turn=None, sequential=False, fast=False):
    drive_svd = DriveSvd(bpm_names, bpm_data, usv, tunes, plane,
                               output_file, model_path, nattunes,
                               tolerance,
                               start_turn, end_turn, sequential, fast)
    drive_svd.start_analysis()
    return drive_svd


def _set_up_main_logger():
    main_logger = logging.getLogger("")
    main_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    main_logger.addHandler(console_handler)


if __name__ == "__main__":
    _set_up_main_logger()
    _init_from_args()
