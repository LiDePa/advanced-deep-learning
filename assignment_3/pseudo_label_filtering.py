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

    raise NotImplementedError(
        f'{pseudo_label_filtering.__name__} has not been implemented yet.')

    # TODO: Filter the pseudo-labels per class. To do so, first calculate the mean confidence
    #       given a class prediction and their standard deviation. The filter pseudo-labels
    #       where the confidence that lead to a given prediction is smaller than mean + stdd.
    #       This means you should use a total of 15 thresholds.