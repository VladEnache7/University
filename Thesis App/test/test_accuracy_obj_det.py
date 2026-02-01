from unittest.mock import MagicMock

import pytest
import torch
from torchmetrics.detection import MeanAveragePrecision

from metrics import ObjectDetectionAccuracy

# pylint: disable=W0621

@pytest.fixture
def matcher():
    """Fixture to create a mock matcher."""
    return MagicMock()


@pytest.fixture
def min_iou():
    """Fixture for minimum IoU."""
    return 0.5


@pytest.fixture
def object_detection_accuracy(matcher, min_iou):
    """Fixture to initialize ObjectDetectionAccuracy with a mock matcher."""
    return ObjectDetectionAccuracy(matcher=matcher, min_iou=min_iou)


def test_initialization(object_detection_accuracy, matcher, min_iou):
    """Test initialization of ObjectDetectionAccuracy."""
    assert isinstance(object_detection_accuracy.map, MeanAveragePrecision)
    assert object_detection_accuracy.map.iou_thresholds == torch.linspace(
        min_iou, 0.95, round((0.95 - min_iou) / 0.05) + 1).tolist()
    assert object_detection_accuracy.matcher == matcher


def test_accuracy_for_perfect_match(object_detection_accuracy, matcher):
    """Test that the accuracy is 1 for a perfect match."""
    # Perfect match outputs and targets
    outputs = {
        'pred_logits': torch.tensor([[[0., 10.], [10., 0.]]]),
        'pred_boxes': torch.tensor([[[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]]])
    }

    targets = [
        {'labels': torch.tensor([1, 0]), 'boxes': torch.tensor(
            [[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]])}
    ]

    # Mock the matcher return for perfect match
    matcher.return_value = [
        (torch.tensor([0, 1]), torch.tensor([0, 1]))
    ]

    # Update the metric state
    object_detection_accuracy.update(outputs=outputs, targets=targets)

    # Compute the accuracy
    result = object_detection_accuracy.compute()

    # Assert that the accuracy is 1 for perfect match
    assert result.item() == 1.0


def test_accuracy_for_imperfect_match(object_detection_accuracy, matcher):
    """Test that the accuracy value is computed manually for an imperfect match."""
    # Mock the matcher return
    # Assume first prediction matches the first target, second prediction matches the second target incorrectly
    matcher.return_value = [
        # Matches: first is correct, second is incorrect
        (torch.tensor([0, 1]), torch.tensor([0, 1]))
    ]

    imperfect_outputs = {
        # Logits: one correct, one uncertain
        'pred_logits': torch.tensor([[[0., 10.], [5., 5.]]]),
        # One box matches, one does not
        'pred_boxes': torch.tensor([[[0.5, 0.5, 1.0, 1.0], [0.4, 0.4, 0.6, 0.6]]])
    }
    imperfect_targets = [
        {'labels': torch.tensor([1, 0]), 'boxes': torch.tensor(
            [[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]])}
    ]

    # Update the metric state
    object_detection_accuracy.update(
        outputs=imperfect_outputs, targets=imperfect_targets)

    # Compute the result
    result = object_detection_accuracy.compute()

    # Manually calculate the expected accuracy:
    # Assume IoU >= min_iou (0.5) for box 1 (perfect match) but IoU < 0.5 for box 2 (imperfect match)
    # Classification is correct for box 1, incorrect for box 2

    # Example calculation for Mean Average Precision:
    # Box 1: TP (True Positive), Box 2: FP (False Positive)
    # Expected mAP: 1 (for TP) / 2 (total boxes) = 0.5 (example value)

    # Adjust this based on your manual mAP computation.
    expected_accuracy = 0.5

    # Assert that the computed result matches the manually calculated value
    assert torch.isclose(result, torch.tensor(expected_accuracy),
                         atol=1e-2), f"Expected {expected_accuracy}, but got {result.item()}"
