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
            module = importlib.import_module(f"{filename}.labels")
            x = module.label_to_color
            x = module.labels
            module = importlib.import_module(f"{filename}.mean_iou")
            x = module.MeanIntersectionOverUnion
            x = module.MeanIntersectionOverUnion(2, 1)
            y = x.update
            x.reset()
            x.compute()
            module = importlib.import_module(f"{filename}.seg_resnet")
            x = module.ResNetSegmentationModel(2, True)
            x = module.ResNetSegmentationModel(2, False)
            x = x.forward
            module = importlib.import_module(f"{filename}.segmentation_dataset")
            x = module.SegmentationDatasetFromList
            module = importlib.import_module(f"{filename}.train_seg")
            x = module.SegTrainer
            x = module.main
            module = importlib.import_module(f"{filename}.utils")
            x = module.set_deterministic()

    print("Submission for Assignment 2 successfully checked. You can now upload it to Digicampus.")


if __name__ == "__main__":
    cli()
