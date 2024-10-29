# -*- coding: utf-8 -*-

import importlib
import os
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import sys


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
            filename = os.path.basename(args.input)[:-4]

            sys.path.insert(1, f'{unpackdir}')
            module = importlib.import_module(f"{filename}.dataset")
            func1 = module.get_simpsons_subsets
            func2 = module.get_dataloader
            class1 = module.SimpsonsDataset

            module = importlib.import_module(f"{filename}.models")
            class2 = module.ResNet18Model
            class2 = module.ConvNextTinyModel

            module = importlib.import_module(f"{filename}.training")
            func3 = module.train
            func4 = module.evaluation

    print("Submission for Assignment 1 successfully checked. You can now upload it to Digicampus.")


if __name__ == "__main__":
    cli()
