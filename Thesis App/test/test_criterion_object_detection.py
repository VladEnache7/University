# pylint: disable=W0621,R0801
from unittest.mock import MagicMock

import pytest
import torch

from losses.criterion_object_detection import SetCriterionObjDet

# pylint: disable=W0621


@pytest.fixture
def num_classes():
    """Fixture to define the number of classes for the object detection model.

    Returns:
        int: Number of classes in the dataset (2 in this case).
    """
    return 2


@pytest.fixture
def matcher():
    """Fixture to mock the matcher that returns predefined indices for testing.

    Returns:
        MagicMock: Mock object that mimics the behavior of a matcher.
    """
    # Mock matcher to return predefined indices for testing
    mock = MagicMock()
    mock.return_value = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
    return mock


@pytest.fixture
def weight_dict():
    """Fixture to define the loss weights for the criterion.

    Returns:
        dict: A dictionary containing the weights for different loss components.
    """
    return {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}


@pytest.fixture
def eos_coef():
    """Fixture to define the end-of-sequence coefficient.

    Returns:
        float: The coefficient for handling 'no object' predictions.
    """
    return 0.1


@pytest.fixture
def losses():
    """Fixture to define the types of losses to be calculated.

    Returns:
        list: A list containing the names of the loss components to be computed.
    """
    return ['labels', 'cardinality', 'boxes']


@pytest.fixture
def criterion(num_classes, matcher, weight_dict, eos_coef, losses):
    """Fixture to create an instance of the SetCriterionObjDet class.

    Args:
        num_classes (int): Number of classes in the dataset.
        matcher (MagicMock): Mock matcher object.
        weight_dict (dict): Dictionary containing loss weights.
        eos_coef (float): End-of-sequence coefficient.
        losses (list): List of loss components to compute.

    Returns:
        SetCriterionObjDet: Instance of the SetCriterionObjDet class.
    """
    return SetCriterionObjDet(num_classes, matcher, weight_dict, eos_coef, losses)


@pytest.fixture
def outputs():
    """Fixture to provide example outputs from a model for testing.

    Returns:
        dict: A dictionary containing predicted logits and bounding boxes.
    """
    return {
        # batch size x num_queries x num_classes
        'pred_logits': torch.tensor([[[0.5, 0.5, 10.0], [0.5, 0.5, 0.1]]]),
        # batch size x num_queries x 4
        'pred_boxes': torch.tensor([[[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]]])
    }


@pytest.fixture
def targets():
    """Fixture to provide example targets for testing.

    Returns:
        list: A list of dictionaries containing ground truth labels and bounding boxes.
    """
    return [
        {'labels': torch.tensor([2, 1]), 'boxes': torch.tensor(
            [[0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.4, 0.4]])}
    ]


@pytest.fixture
def indices():
    """Fixture to provide example matched indices for testing.

    Returns:
        list: A list of tuples containing matched indices for predicted and target boxes.
    """
    return [(torch.tensor([0, 1]), torch.tensor([0, 1]))]


def test_loss_labels(criterion, outputs, targets, indices):
    """Test the label loss computation in the criterion.

    Args:
        criterion (SetCriterionObjDet): The criterion object to compute losses.
        outputs (dict): Model outputs containing predicted logits and bounding boxes.
        targets (list): Ground truth targets.
        indices (list): Matched indices for predicted and target boxes.
    """
    num_boxes = 2  # Example number of boxes
    losses = criterion.loss_labels(outputs, targets, indices, num_boxes)
    assert 'loss_ce' in losses
    assert losses['loss_ce'] > 0


def test_loss_cardinality(criterion, outputs, targets, indices):
    """Test the cardinality loss computation in the criterion.

    Args:
        criterion (SetCriterionObjDet): The criterion object to compute losses.
        outputs (dict): Model outputs containing predicted logits and bounding boxes.
        targets (list): Ground truth targets.
        indices (list): Matched indices for predicted and target boxes.
    """
    num_boxes = 2
    losses = criterion.loss_cardinality(outputs, targets, indices, num_boxes)
    assert 'cardinality_error' in losses
    assert losses['cardinality_error'] >= 0


def test_loss_boxes(criterion, outputs, targets, indices):
    """Test the bounding box and generalized IoU loss computation in the criterion.

    Args:
        criterion (SetCriterionObjDet): The criterion object to compute losses.
        outputs (dict): Model outputs containing predicted logits and bounding boxes.
        targets (list): Ground truth targets.
        indices (list): Matched indices for predicted and target boxes.
    """
    num_boxes = 2
    losses = criterion.loss_boxes(outputs, targets, indices, num_boxes)
    assert 'loss_bbox' in losses
    assert 'loss_giou' in losses
    assert losses['loss_bbox'] >= 0
    assert losses['loss_giou'] >= 0


def test_forward(criterion, outputs, targets):
    """Test the overall forward pass of the criterion for loss computation.

    Args:
        criterion (SetCriterionObjDet): The criterion object to compute losses.
        outputs (dict): Model outputs containing predicted logits and bounding boxes.
        targets (list): Ground truth targets.
    """
    losses = criterion(outputs, targets)
    assert 'loss_ce' in losses
    assert 'loss_bbox' in losses
    assert 'loss_giou' in losses
    assert 'cardinality_error' in losses
    assert losses['loss_ce'] > 0
    assert losses['loss_bbox'] >= 0
    assert losses['loss_giou'] >= 0
    assert losses['cardinality_error'] >= 0
