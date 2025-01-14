# -*- coding: utf-8 -*-

import importlib
import os
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import sys
import numpy as np


def cli():
    parser = ArgumentParser()
    parser.add_argument(
            "input", help="Input archive (zip file) of your submission that should be checked for testability"
            )
    args = parser.parse_args()

    inpath = Path(args.input)

    assert inpath.is_file() and inpath.suffix == ".zip", "Input must be a zip file"
    with TemporaryDirectory(prefix="testability_extract_") as unpackdir:
        with ZipFile(inpath, "r") as submissions_archive:
            submissions_archive.extractall(unpackdir)
            filename = os.path.basename(args.input)[:-4]  # filename without path and without .zip extension

            sys.path.insert(1, f'{unpackdir}')
            module = importlib.import_module(f"{filename}.heatmaps")
            joints = [[1, 1, 1],
                      [1, 1, 0],
                      [0, 0, 0]]
            test_map = module.create_heatmaps(np.asarray(joints))
            assert test_map.shape[0] == 3

            module = importlib.import_module(f"{filename}.dataset")
            func1 = module.load_dataset
            func2 = module.create_skijump_subsets
            class1 = module.SkijumpDataset
            func3 = class1.augment

            module = importlib.import_module(f"{filename}.metric")
            func4 = module.pck

    print("Submission for Assignment 5 successfully checked. You can now upload it to Digicampus.")


if __name__ == "__main__":
    cli()
