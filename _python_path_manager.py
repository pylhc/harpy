import sys
import os


SEARCH_LEVELS = 3


def append_betabeat():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    afs_prefix = "/afs"
    if "win" in sys.platform and sys.platform != "darwin":
        afs_prefix = "\\AFS"
    path_to_check = current_dir
    for _ in range(SEARCH_LEVELS):
        if os.path.basename(path_to_check) == "Beta-Beat.src":
            sys.path.append(path_to_check)
            return
        path_to_check = os.path.abspath(os.path.join(path_to_check, ".."))
    lintrack_path = os.path.join(afs_prefix, "cern.ch", "eng", "sl",
                                 "lintrack", "Beta-Beat.src")
    if os.path.isdir(lintrack_path):
        sys.path.append(lintrack_path)
        return
    raise ImportError("Cannot find Beta-Beat.src, harpy drive will not work.")
