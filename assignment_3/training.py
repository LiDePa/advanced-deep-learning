from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tensorboard

from torch.utils.data import DataLoader
from .callbacks import ModelCheckpoint
from .dataset import get_segmentation_dataset
from .ema import EMAModel
from .engine import Engine, Event
from .metrics import MeanIntersectionOverUnion
from .pseudo_label_filtering import PseudoLabelFiltering
from .model import ResNetSegmentationModel
from .semi_supervised import get_ss_train_step_hl, get_ss_train_step_sl, get_supervised_train_step, get_validation_step
from .transforms import CutOut, Normalize, RandomResizeCrop
from .utils import Counter


def parse_args() -> argparse.Namespace:
    """Parses the program arguments for assignment 3.

    Returns:
        argparse.Namespace: The arguments for assignment 3.
    """
    parser = argparse.ArgumentParser(
        "Run a semi supervised training to the segmentation task.")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)

    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--disable-tqdm", required=False,
                        action="store_true", default=False)
    parser.add_argument("--use-dropout-perturbation",
                        required=False, default=False, action="store_true")
    parser.add_argument("--pseudo-label-filtering",
                        required=False, default=False, action="store_true")
    parser.add_argument("--cutout-perturbation", required=False,
                        default=False, action="store_true")
    parser.add_argument("--soft-labels", required=False,
                        default=False, action="store_true")

    parser.add_argument("--batch-size", default=8, required=False, type=int)
    parser.add_argument("--num-workers", default=16, required=False, type=int)

    parser.add_argument("--output-tag", type=str, required=False, default="")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    # Creating logging utilities
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(
        args.output_dir, ''.join((time_str, args.output_tag)))
    os.makedirs(output_dir, exist_ok=True)
    summary_writer = tensorboard.SummaryWriter(log_dir=output_dir)

    ### DATA SETUP ###

    train_dataset = get_segmentation_dataset(
        dataset_root=args.dataset_root,
        split='train',
        transforms=[
            Normalize(),
            RandomResizeCrop(crop_size=192)
        ]
    )

    # Needed to be able to use the same dataset split over multiple runs.
    generator = torch.Generator()
    generator.manual_seed(1337)
    supervised_train_dataset, unsupervised_train_dataset = torch.utils.data.random_split(
        train_dataset,
        [0.2, 0.8],
        generator=generator
    )

    supervised_train_loader = DataLoader(
        dataset=supervised_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True
    )

    unsupervised_train_loader = DataLoader(
        dataset=unsupervised_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True
    )

    # Creating the validation segmentation dataset
    validation_dataset = get_segmentation_dataset(
        dataset_root=args.dataset_root,
        split='val',
        transforms=[
            Normalize()
        ]
    )

    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    ### MODEL SETUP ###

    student = ResNetSegmentationModel(
        num_classes=15, use_intermediate_features=True).to(device=args.device)

    teacher = EMAModel(student, device=args.device)

    ### TRAINING SETUP ###

    loss_fn = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
    optimizer = optim.AdamW(student.parameters(), lr=1e-5)

    semi_supervision_transforms = []

    if args.pseudo_label_filtering:
        semi_supervision_transforms.append(
            PseudoLabelFiltering(
                num_classes=15,
                ignore_class=255
            )
        )

    if args.cutout_perturbation:
        semi_supervision_transforms.append(
            CutOut(
                scales=(0.3, 0.7),
                ignore_class=255
            )
        )

    supervised_train_engine = Engine(
        get_supervised_train_step(
            model=student,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=args.device
        )
    )

    semi_supervised_train_engine = Engine(
        get_ss_train_step_hl(
            student=student,
            teacher=teacher,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=args.device,
            transforms=semi_supervision_transforms,
            use_dropout_perturbation=args.use_dropout_perturbation
        )
    )

    if args.soft_labels:

        semi_supervised_train_engine = Engine(
            get_ss_train_step_sl(
                student=student,
                teacher=teacher,
                optimizer=optimizer,
                device=args.device,
            )
        )

    c = Counter()

    supervised_train_engine.add_event_handler(
        Event.ITERATION_COMPLETED,
        lambda _engine: c()
    )

    semi_supervised_train_engine.add_event_handler(
        Event.ITERATION_COMPLETED,
        lambda _engine: c()
    )

    supervised_train_engine.add_event_handler(
        Event.ITERATION_COMPLETED,
        lambda engine: summary_writer.add_scalar(
            "train//loss",
            engine.last_output["loss"],
            global_step=c.state
        )
    )

    semi_supervised_train_engine.add_event_handler(
        Event.ITERATION_COMPLETED,
        lambda engine: summary_writer.add_scalar(
            "train/ss/loss",
            engine.last_output["loss"],
            global_step=c.state
        )
    )

    supervised_train_engine.add_event_handler(
        Event.EPOCH_COMPLETED,
        lambda _engine: semi_supervised_train_engine.run(
            unsupervised_train_loader,
            disable_tqdm=True
        )
    )

    supervised_train_engine.add_event_handler(
        Event.ITERATION_COMPLETED,
        lambda _engine: teacher.update(student)
    )

    semi_supervised_train_engine.add_event_handler(
        Event.ITERATION_COMPLETED,
        lambda _engine: teacher.update(student)
    )

    # Creating validation utilities for the student model
    validation_engine = Engine(
        get_validation_step(
            model=student,
            device=args.device
        )
    )

    teacher_validation_engine = Engine(
        get_validation_step(
            model=teacher,
            device=args.device
        )
    )

    # Adding metric evaluation to the validation engine
    metric = MeanIntersectionOverUnion(
        num_classes=15,
        ignore_class=255,
        device=args.device
    )

    teacher_model_checkpoint = ModelCheckpoint(
        output_dir=output_dir,
        metric=metric,
        filename="teacher_model_checkpoint"
    )

    validation_engine.add_event_handler(
        Event.ITERATION_COMPLETED,
        lambda engine: metric.update(
            engine.last_output["prediction"],
            engine.last_output["y"]
        )
    )

    teacher_validation_engine.add_event_handler(
        Event.ITERATION_COMPLETED,
        lambda engine: metric.update(
            engine.last_output["prediction"],
            engine.last_output["y"]
        )
    )

    # Adding metric logging to the validation engine
    validation_engine.add_event_handler(
        Event.EPOCH_COMPLETED,
        lambda _engine: summary_writer.add_scalar(
            "val//miou",
            metric.compute().item(),
            global_step=c.state
        )
    )

    teacher_validation_engine.add_event_handler(
        Event.EPOCH_COMPLETED,
        lambda _engine: summary_writer.add_scalar(
            "val/ema/miou",
            metric.compute().item(),
            global_step=c.state
        )
    )

    teacher_validation_engine.add_event_handler(
        Event.EPOCH_COMPLETED,
        lambda _engine: teacher_model_checkpoint.checkpoint_if_necessary(
            dict(model=teacher._model.state_dict()),
            filename_suffix=f"_{supervised_train_engine.epoch}"
        )
    )

    # Resetting the metric after each finished validation run.
    validation_engine.add_event_handler(
        Event.ENGINE_FINISHED,
        lambda _engine: metric.reset()
    )

    teacher_validation_engine.add_event_handler(
        Event.ENGINE_FINISHED,
        lambda _engine: metric.reset()
    )

    # Adding the validation engines to the training
    supervised_train_engine.add_event_handler(
        Event.EPOCH_STARTED,
        lambda _engine: validation_engine.run(
            validation_data_loader,
            disable_tqdm=True
        ),
        every=25
    )
    supervised_train_engine.add_event_handler(
        Event.EPOCH_STARTED,
        lambda _engine: teacher_validation_engine.run(
            validation_data_loader,
            disable_tqdm=True
        ),
        every=25
    )

    supervised_train_engine.add_event_handler(
        Event.ENGINE_FINISHED,
        lambda _engine: validation_engine.run(
            validation_data_loader,
            disable_tqdm=True
        )
    )

    supervised_train_engine.add_event_handler(
        Event.ENGINE_FINISHED,
        lambda _engine: teacher_validation_engine.run(
            validation_data_loader,
            disable_tqdm=True
        )
    )

    supervised_train_engine.run(
        data_loader=supervised_train_loader,
        epochs=1750,
        disable_tqdm=args.disable_tqdm
    )


if __name__ == "__main__":
    # guaranteeing that "cuda:0" is available
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Emptying torch cahce to make sure that there are no malformed memory references
    torch.cuda.empty_cache()

    args = parse_args()
    main(args=args)
