from .detr_head import build
from .linear_head_classification import build_linear_head

TASK_DEC = {
    'decoder_object_detection': build_linear_head,
    'detr_head': build
}
