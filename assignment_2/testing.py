# TODO: delete this file



from .transforms import RandomCrop, Normalize
from .carla_dataset import get_carla_dataset


dataset = get_carla_dataset("datasets/carla3.0_for_students", split="train")
crop = RandomCrop(crop_size=192)
sample = dataset[11]
sample_cropped = crop(sample)

print(sample["y"], sample_cropped["y"])











"""

import torch
from .model import ResNetSegmentationModel


def test_resnet_segmentation_model():
    # Define the parameters
    num_classes = 15
    batch_size = 1
    channels = 3
    height, width = 256, 512
    input_shape = (batch_size, channels, height, width)

    # Initialize the model
    model = ResNetSegmentationModel(num_classes=num_classes)

    # Move the model to the same device as the dummy input
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dummy input
    dummy_input = torch.rand(input_shape).to(device)

    # Run a forward pass
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(dummy_input)

    # Check the output shape
    expected_output_shape = (batch_size, num_classes, height, width)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {expected_output_shape}")

    # Validate the output
    assert output.shape == expected_output_shape, (
        f"Test failed! Output shape {output.shape} does not match expected shape {expected_output_shape}."
    )
    print("Test passed! Output shape matches expected shape.")


test_resnet_segmentation_model()
















import torch
from .metrics import MeanIntersectionOverUnion


def calculate_expected_miou(tp, fp, fn, num_classes):
    ious = []
    for cls in range(num_classes):
        denom = tp[cls] + fp[cls] + fn[cls]
        iou = tp[cls] / denom if denom > 0 else 0.0
        ious.append(iou)
    return sum(ious) / len(ious)


def generate_test_case(batch_size, num_classes, img_size, ignore_class=None):
    predictions = torch.randint(0, num_classes, (batch_size, *img_size))
    labels = torch.randint(0, num_classes, (batch_size, *img_size))

    if ignore_class is not None:
        ignore_mask = torch.rand_like(predictions, dtype=torch.float32) < 0.1  # 10% ignored
        labels[ignore_mask] = ignore_class
    return predictions, labels


def test_miou():
    num_classes = 15
    ignore_class = 255
    batch_size = 64
    img_size = (256, 512)

    # Instantiate the metric
    metric = MeanIntersectionOverUnion(num_classes=num_classes, ignore_class=ignore_class)

    # Test cases
    test_cases = [
        ("Random Predictions", *generate_test_case(batch_size, num_classes, img_size, ignore_class)),
        ("Perfect Predictions",
         torch.randint(0, num_classes, (batch_size, *img_size)),
         lambda preds: preds.clone()),
        ("Completely Wrong Predictions",
         torch.randint(0, num_classes, (batch_size, *img_size)),
         lambda preds: (preds + 1) % num_classes),
        ("Only Ignore Class",
         torch.randint(0, num_classes, (batch_size, *img_size)),
         lambda preds: torch.full_like(preds, ignore_class))
    ]

    for case_name, predictions, labels_fn in test_cases:
        if callable(labels_fn):
            labels = labels_fn(predictions)
        else:
            labels = labels_fn

        # Reset metric for the new test case
        metric.reset()

        # Simulate batch processing
        for i in range(0, predictions.size(0), 16):  # Process in smaller batches
            batch_preds = predictions[i:i + 16]
            batch_labels = labels[i:i + 16]
            metric.update(batch_preds, batch_labels)

        # Compute mIoU
        calculated_miou = metric.compute()

        # Calculate expected mIoU
        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)

        for c in range(num_classes):
            tp[c] = ((predictions == c) & (labels == c)).sum()
            fp[c] = ((predictions == c) & (labels != c) & (labels != ignore_class)).sum()
            fn[c] = ((predictions != c) & (labels == c)).sum()

        expected_miou = calculate_expected_miou(tp, fp, fn, num_classes)

        print(f"Test Case: {case_name}")
        print(f"Expected mIoU: {expected_miou:.4f}, Calculated mIoU: {calculated_miou:.4f}")
        print("-" * 50)

test_miou()


"""