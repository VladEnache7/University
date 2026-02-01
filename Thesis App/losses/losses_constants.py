from .criterion_keypoints import SetCriterion
from .hungarian_matcher import HungarianMatcher

TASK_LOSS = {
    "Hungarian": SetCriterion
}

TASK_MATCHER = {
    "Hungarian": HungarianMatcher
}
