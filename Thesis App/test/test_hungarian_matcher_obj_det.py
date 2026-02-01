# pylint: disable=W0621,R0801
import pytest
import torch

from losses.hungarian_matcher_object_detection import HungarianMatcherObjDet

# pylint: disable=W0621


@pytest.fixture
def matcher():
    """Fixture to initialize HungarianMatcherObjDet."""
    return HungarianMatcherObjDet()


def test_perfect_match(matcher):
    # Test case where predictions exactly match the targets
    outputs = {
        # High confidence in classes 1 and 0 respectively
        'pred_logits': torch.tensor([[[0., 10.], [10., 0.]]]),
        # Predicted bounding boxes
        'pred_boxes': torch.tensor([[[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]]])
    }

    targets = [
        {'labels': torch.tensor([1, 0]),  # Ground truth classes
         'boxes': torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]])}  # Ground truth boxes
    ]

    indices = matcher(outputs, targets)
    # Since the predicted boxes and labels exactly match the targets, we expect a perfect 1-to-1 match
    # Match for the first batch element
    assert indices[0][0].tolist() == [0, 1]
    # Match for the second batch element
    assert indices[0][1].tolist() == [0, 1]


def test_cost_weight_variation():
    # Test case with different weights for each cost component
    matcher_with_diff_costs = HungarianMatcherObjDet(
        cost_class=2.0, cost_bbox=1.0, cost_giou=0.5)

    outputs = {
        # Predicted logits with lower confidence
        'pred_logits': torch.tensor([[[0., 2.], [2., 0.]]]),
        # Predicted boxes
        'pred_boxes': torch.tensor([[[0.1, 0.1, 0.2, 0.2], [0.9, 0.9, 0.8, 0.8]]])
    }

    targets = [
        {'labels': torch.tensor([0, 1]),
         'boxes': torch.tensor([[0.0, 0.0, 0.2, 0.2], [1.0, 1.0, 0.8, 0.8]])}
    ]

    indices = matcher_with_diff_costs(outputs, targets)
    # The expected match depends on the weights and costs; we ensure it completes without errors
    assert isinstance(indices, list)
    assert len(indices) == 1
    assert isinstance(indices[0], tuple)
    assert len(indices[0]) == 2


def test_partial_match(matcher):
    # Test case where there are more predictions than targets
    outputs = {
        # Predictions
        'pred_logits': torch.tensor([[[0., 10.], [10., 0.], [5., 5.]]]),
        'pred_boxes': torch.tensor([[[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4], [0.3, 0.3, 0.5, 0.5]]])
    }

    targets = [
        {'labels': torch.tensor([1, 0]),  # Only 2 ground truth objects
         'boxes': torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]])}
    ]

    indices = matcher(outputs, targets)
    # Ensure the length matches the minimum number of targets or predictions
    # There are only two target boxes, so only two matches
    assert len(indices[0][0]) == 2
    assert len(indices[0][1]) == 2
