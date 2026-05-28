"""
DeepSeek V3 Learning Implementation
===================================

A complete, educational implementation of DeepSeek V3 architecture.

Key Features:
- Multi-head Latent Attention (MLA)
- DeepSeekMoE (Mixture of Experts)
- Multi-Token Prediction (MTP)
- GRPO (Group Relative Policy Optimization)

Usage:
    from deepseek import DeepSeekV3Model, load_config
    
    config = load_config("configs/config_default.yaml")
    model = DeepSeekV3Model(config.model)
"""

# Import from config module (root level)
from config import (
    DeepSeekV3Config,
    ModelConfig,
    TrainingConfig,
    SFTConfig,
    RLConfig,
    DataConfig,
    VisualizationConfig,
    InferenceConfig,
    MoEConfig,
    MTPConfig,
    load_config,
    get_device,
)

# Re-export from new package structure
from deepseek import (
    # Model
    DeepSeekV3Model,
    TransformerBlock,
    DeepSeekMoE,
    Expert,
    MTPHead,
    SwiGLU,
    count_parameters,
    print_model_summary,
    # Attention
    MultiHeadLatentAttention,
    StandardAttention,
    RotaryEmbedding,
    RMSNorm,
    apply_rotary_pos_emb,
    # Dataset
    get_tokenizer,
    create_dataloaders,
    PretrainDataset,
    SFTDataset,
    RLDataset,
    create_rl_dataloaders,
    DPODataset,
    GRPODataset,
    PPODataset,
    # Trainer
    BaseTrainer,
    PretrainTrainer,
    SFTTrainer,
    create_trainer,
    Visualizer,
    DPOTrainer,
    GRPOTrainer,
    PPOTrainer,
    RuleBasedReward,
    CompositeReward,
    LengthReward,
    # Logger
    get_logger,
    set_log_level,
    setup_file_logging,
    get_training_logger,
    get_model_logger,
    get_data_logger,
    get_inference_logger,
    DeepSeekLogger,
    ColoredFormatter,
)

# Import inference separately (it's at root level)
from deepseek.inference.inference import (
    DeepSeekInference,
    load_model_for_inference,
)

__version__ = "0.1.0"
__author__ = "DeepSeek V3 Learning Project"

__all__ = [
    # Config
    "DeepSeekV3Config",
    "ModelConfig",
    "TrainingConfig",
    "SFTConfig",
    "RLConfig",
    "DataConfig",
    "VisualizationConfig",
    "InferenceConfig",
    "MoEConfig",
    "MTPConfig",
    "load_config",
    "get_device",
    # Model
    "DeepSeekV3Model",
    "TransformerBlock",
    "DeepSeekMoE",
    "Expert",
    "MTPHead",
    "SwiGLU",
    "count_parameters",
    "print_model_summary",
    # Attention
    "MultiHeadLatentAttention",
    "StandardAttention",
    "RotaryEmbedding",
    "RMSNorm",
    "apply_rotary_pos_emb",
    # Dataset
    "get_tokenizer",
    "create_dataloaders",
    "PretrainDataset",
    "SFTDataset",
    "RLDataset",
    "create_rl_dataloaders",
    "DPODataset",
    "GRPODataset",
    "PPODataset",
    # Trainer
    "BaseTrainer",
    "PretrainTrainer",
    "SFTTrainer",
    "create_trainer",
    "Visualizer",
    "DPOTrainer",
    "GRPOTrainer",
    "PPOTrainer",
    "RuleBasedReward",
    "CompositeReward",
    "LengthReward",
    # Inference
    "DeepSeekInference",
    "load_model_for_inference",
    # Logger
    "get_logger",
    "set_log_level",
    "setup_file_logging",
    "get_training_logger",
    "get_model_logger",
    "get_data_logger",
    "get_inference_logger",
    "DeepSeekLogger",
    "ColoredFormatter",
]
