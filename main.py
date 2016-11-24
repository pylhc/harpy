import optparse
from drive import Drive


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
    nattunes = (options.nattunex,
                options.nattuney,
                options.nattunez)
    tolerance = options.tolerance
    start_turn = options.start_turn
    end_turn = options.end_turn
    sequential = options.sequential
    return Drive(input_file, tunes, nattunes, tolerance,
                 start_turn, end_turn, sequential)


if __name__ == "__main__":
    _init_from_args()
