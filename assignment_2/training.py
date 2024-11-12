import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from .checkpoint import BestCheckpointHandler
from .ema import EMAModel

from .carla_dataset import get_carla_dataset
from .metrics import MeanIntersectionOverUnion
from .model import ResNetSegmentationModel
from .supervised import get_train_step, get_validation_step
from .engine import Engine, Event
from .transforms import Normalize, RandomCrop, RandomResizeCrop
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

from .utils import collate_fn

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # for multiple GPUs available
    # Emptying torch cache to make sure, that there are no any malformed memory references.
    torch.cuda.empty_cache()

    parser = ArgumentParser("Run a training for the segmentation task.")
    parser.add_argument("--dataset-root",
                        required=True,
                        type=str
                        )
    parser.add_argument("--output-dir",
                        required=True,
                        type=str
                        )
    parser.add_argument("--device",
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        type=str
                        )

    parser.add_argument("--use-intermediate-features",
                        required=False,
                        default=False,
                        action="store_true")

    transforms_group = parser.add_mutually_exclusive_group()
    transforms_group.add_argument("--random-crop", action="store_true")
    transforms_group.add_argument("--random-resize-crop", action="store_true")

    args = parser.parse_args()

    # Creating logging utilities
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, time_str)
    os.makedirs(output_dir, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=output_dir)

    # Creating transforms according to given commandline arguments
    transforms = [Normalize()]
    transforms += ([RandomCrop(crop_size=192)] if args.random_crop else [])
    transforms += ([RandomResizeCrop(crop_size=192)] if args.random_resize_crop else [])

    # Creating the training segmentation dataset.
    train_dataset = get_carla_dataset(
            root=args.dataset_root,
            split="train",
            transforms=transforms
            )

    train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_fn
            )

    # Creating the validation segmentation dataset.
    validation_dataset = get_carla_dataset(
            root=args.dataset_root,
            split="val",
            transforms=[
                    Normalize(),
                    ]
            )

    validation_data_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn
            )

    print(f"USING: {args.use_intermediate_features}")
    # Creating training setup
    model = ResNetSegmentationModel(
            num_classes=15, use_intermediate_features=args.use_intermediate_features).to(device=args.device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    train_engine = Engine(
            get_train_step(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    device=args.device
                    )
            )

    # Adding tensorboard logging for the training loss.
    train_engine.add_event_handler(Event.ITERATION_COMPLETED, lambda engine: summary_writer.add_scalar(
            "train/loss", engine.last_output["loss"], global_step=engine.iteration))

    # Creating EMAModel and appending it to the training step.
    ema_model = EMAModel(model=model, decay_rate=0.998)
    train_engine.add_event_handler(
            Event.ITERATION_COMPLETED, lambda _engine: ema_model.update(model))

    # Creating validation utilities for the directly trained model.
    validation_engine = Engine(
            get_validation_step(
                    model=model,
                    device=args.device
                    )
            )

    metric = MeanIntersectionOverUnion(
            num_classes=15, ignore_class=255, device=args.device)
    model_checkpoint = BestCheckpointHandler(
            output_dir=output_dir, metric=metric, filename="model_checkpoint")

    # Adding metric evaluation to the validation step
    validation_engine.add_event_handler(Event.ITERATION_COMPLETED, lambda engine: metric.update(
            engine.last_output["predictions"], engine.last_output["y"]))

    # Adding metric logging for the directly trained model.
    validation_engine.add_event_handler(Event.EPOCH_COMPLETED, lambda _engine: summary_writer.add_scalar(
            "val//miou", metric.compute().item(), global_step=train_engine.iteration))

    # Checkpointing the directly trained model.
    validation_engine.add_event_handler(Event.EPOCH_COMPLETED, lambda _engine: model_checkpoint.checkpoint_if_necessary(
            dict(model=model), filename_suffix=f"_{train_engine.iteration}"))

    # Resetting the metric after each finished validation run.
    validation_engine.add_event_handler(
            Event.ENGINE_FINISHED, lambda _engine: metric.reset())

    # Creating validation utilities for the directly trained model.
    ema_validation_engine = Engine(
            get_validation_step(
                    model=ema_model,
                    device=args.device
                    )
            )

    ema_metric = MeanIntersectionOverUnion(
            num_classes=15, ignore_class=255, device=args.device)
    ema_model_checkpoint = BestCheckpointHandler(
            output_dir=output_dir, metric=ema_metric, filename="ema_model_checkpoint")

    # Adding metric evaluation to the validation step for the EMA model.
    ema_validation_engine.add_event_handler(Event.ITERATION_COMPLETED, lambda engine: ema_metric.update(
            engine.last_output["predictions"], engine.last_output["y"]))

    # Adding metric logging for the EMA model.
    ema_validation_engine.add_event_handler(Event.EPOCH_COMPLETED, lambda _engine: summary_writer.add_scalar(
            "val/ema/miou", ema_metric.compute().item(), global_step=train_engine.iteration))

    # Adding model checkpointing for each finished validation run.
    ema_validation_engine.add_event_handler(Event.EPOCH_COMPLETED, lambda _engine: ema_model_checkpoint.checkpoint_if_necessary(
            dict(ema_model=ema_model), filename_suffix=f"_{train_engine.iteration}"))

    # Resetting EMA metric after each validation run.
    ema_validation_engine.add_event_handler(
            Event.ENGINE_FINISHED, lambda _engine: ema_metric.reset())

    # Adding validation runs for every 10 epochs of training.
    train_engine.add_event_handler(Event.EPOCH_COMPLETED, lambda _engine: validation_engine.run(
            validation_data_loader), every=10)
    train_engine.add_event_handler(Event.EPOCH_COMPLETED, lambda _engine: ema_validation_engine.run(
            validation_data_loader), every=10)

    # Running the training.
    train_engine.run(train_data_loader, epochs=300)
