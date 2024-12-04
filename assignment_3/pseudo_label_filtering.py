from __future__ import annotations
from typing import Dict


from .transforms import Transform

import torch


class PseudoLabelFiltering(Transform):

    def __init__(
        self: PseudoLabelFiltering,
        num_classes: int,
        *,
        ignore_class: int = 255
    ) -> None:
        super(PseudoLabelFiltering, self).__init__()
        self._num_classes = num_classes
        self._ignore_class = ignore_class

    def __call__(
        self: PseudoLabelFiltering,
        sample: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Applies pseudo label filtering to the given data sample using the pseudo labels, class confidences and pseudo label threshold given in the class constructor.
        Sets filtered pseudo label pixels to the ignore class.

        Args:
            sample: (Dict[str, torch.Tensor]): The sample containing images, pseudo labels and their prediction confidences.
            pseudo_label_name (str): Name used to select the correct entry from the sample.
            confidences_name (str): Name used to select to correct entry from the sample.

        Returns:
            Dict[str, torch.Tensor]: A data sample with filtered pseudo labels, where filtered pixels are set to the ignore class.
        """
        pseudo_labels = sample['pseudo_labels']
        confidences = sample['confidences']
        filtered_pseudo_labels = pseudo_label_filtering(
            pseudo_labels=pseudo_labels,
            confidences=confidences,
            num_classes=self._num_classes,
            ignore_class=self._ignore_class
        )
        sample['pseudo_labels'] = filtered_pseudo_labels
        return sample


def pseudo_label_filtering(
    pseudo_labels: torch.Tensor,
    confidences: torch.Tensor,
    num_classes: int,
    ignore_class: int,
) -> torch.Tensor:
    """Filters all pseudo-labels for which their confidence is smaller that their per class mean + stdd.
    Filtered labels are set to the given ignore class.

    Args:
        pseudo_labels (torch.Tensor): The pseudo-labels being filtered.
        confidences (torch.Tensor): The confidences that lead to the predicted pseudo-labels
        num_classes (int): The number of classes.
        ignore_class (int): The value filtered pseudo-labels are set to.

    Returns:
        torch.Tensor: Filtered pseudo-labels with filtered labels set to the ignore class.
    """

    # iterate through each of the classes; everything is handled batch-wise
    for class_label in range(num_classes):
        # get all confidence values for the class; skip rest of loop if class doesn't appear in pseudo_labels
        mask_class = pseudo_labels == class_label
        if not mask_class.any():
            continue
        confidences_class = confidences[mask_class]

        # calculate mean and standard deviation of confidence values to define a threshold
        threshold_class = torch.mean(confidences_class) + torch.std(confidences_class)

        #set all pseudo labels with a confidence below or equal to the threshold to ignore_class
        pseudo_labels[mask_class & (confidences <= threshold_class)] = ignore_class

    return pseudo_labels