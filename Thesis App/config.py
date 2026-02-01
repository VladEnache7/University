
"""Configuration constants and parameters for the detection model."""

# Image normalization parameters (ImageNet standards)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Model configuration
MODEL_CONFIG = {
    'd_model': 256,
    'nhead': 4,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 1024,
    'dropout': 0,
    'activation': "relu",
    'return_intermediate_dec': True,
    'num_feature_levels': 4,
    'dec_n_points': 4,
    'enc_n_points': 4,
    'two_stage': False
}

BACKBONE_CONFIG = {
    'type_backbone': 'resnext50_32x4d',
    'hidden_dim': 256,
    'type_embedding': 'sine',
    'return_interm_layers': True,
    'train_backbone': False,
    'dilation': False
}

DETR_CONFIG = {
    'num_queries': 20,
    'num_feature_levels': 4,
    'aux_loss': False,
    'with_box_refine': True,
    'two_stage': False
}

# Default paths
DEFAULT_CHECKPOINT_PATH = 'checkpoints/model_latest.pth'
RESULTS_DIR = 'results'

# Device configuration
def get_device():
    """Get the appropriate device for computation."""
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
