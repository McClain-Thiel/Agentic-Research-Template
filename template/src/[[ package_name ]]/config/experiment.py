"""Experiment configuration models with validation and YAML serialization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Model architecture configuration.

    Defines the neural network architecture, dimensions, and
    regularization parameters for an experiment.
    """

    name: str = Field(default="base", description="Model architecture name")
    hidden_size: int = Field(default=128, description="Hidden layer dimension", ge=1)
    num_layers: int = Field(default=2, description="Number of layers in the model", ge=1)
    num_heads: int = Field(default=4, description="Number of attention heads", ge=1)
    dropout: float = Field(
        default=0.1,
        description="Dropout probability applied to hidden layers",
        ge=0.0,
        lt=1.0,
    )
    activation: Literal["relu", "gelu", "silu"] = Field(
        default="gelu", description="Activation function for hidden layers"
    )
    vocab_size: int | None = Field(
        default=None, description="Vocabulary size (None = infer from data)", ge=1
    )
    max_position_embeddings: int = Field(
        default=512, description="Maximum position embedding index", ge=1
    )
    tie_weights: bool = Field(default=True, description="Tie input/output embedding weights")

    @field_validator("hidden_size")
    @classmethod
    def _validate_hidden_size(cls, v: int) -> int:
        """Ensure hidden_size is even for clean head splitting."""
        if v % 2 != 0:
            raise ValueError("hidden_size must be even (for head splitting)")
        return v

    @field_validator("num_heads")
    @classmethod
    def _validate_num_heads(cls, v: int, info: Any) -> int:
        """Ensure num_heads divides hidden_size evenly."""
        # We defer the divisibility check to model_validator since hidden_size
        # may not yet be set when num_heads is validated.
        return v

    @model_validator(mode="after")
    def _validate_head_dim(self) -> ModelConfig:
        """Verify that hidden_size is divisible by num_heads."""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads}) — got remainder "
                f"{self.hidden_size % self.num_heads}"
            )
        return self


class DataConfig(BaseModel):
    """Dataset and data loading configuration.

    Controls dataset selection, batch sizing, data splitting, and
    dataloader worker parallelism.
    """

    dataset: str = Field(default="dummy", description="Dataset name (HuggingFace) or local path")
    dataset_subset: str | None = Field(
        default=None, description="Subset / config name for the dataset"
    )
    dataset_revision: str | None = Field(
        default=None, description="Git revision (branch/tag/commit) for the dataset"
    )
    text_column: str = Field(default="text", description="Column name containing text data")
    label_column: str = Field(default="label", description="Column name containing labels")
    batch_size: int = Field(default=32, description="Training batch size per device", ge=1)
    eval_batch_size: int = Field(default=64, description="Evaluation batch size per device", ge=1)
    num_workers: int = Field(default=4, description="Number of dataloader worker processes", ge=0)
    pin_memory: bool = Field(
        default=True, description="Pin memory in DataLoader for faster GPU transfer"
    )
    persistent_workers: bool = Field(
        default=True, description="Keep dataloader workers alive across epochs"
    )
    train_split: float = Field(default=0.8, description="Fraction of data for training (0.0–1.0)")
    val_split: float = Field(default=0.1, description="Fraction of data for validation (0.0–1.0)")
    test_split: float = Field(default=0.1, description="Fraction of data for testing (0.0–1.0)")
    max_seq_length: int = Field(
        default=512, description="Maximum sequence length after tokenization", ge=1
    )
    streaming: bool = Field(default=False, description="Use streaming mode for large datasets")
    shuffle_buffer_size: int = Field(
        default=10_000,
        description="Shuffle buffer size for streaming datasets",
        ge=1,
    )

    @model_validator(mode="after")
    def _validate_splits(self) -> DataConfig:
        """Ensure train/val/test splits sum to exactly 1.0."""
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Data splits must sum to 1.0, got {total:.6f} "
                f"(train={self.train_split}, val={self.val_split}, "
                f"test={self.test_split})"
            )
        if any(s < 0 for s in (self.train_split, self.val_split, self.test_split)):
            raise ValueError("All data splits must be non-negative")
        return self

    @model_validator(mode="after")
    def _validate_streaming(self) -> DataConfig:
        """Ensure streaming datasets don't have standard splits."""
        if self.streaming and (self.train_split != 0.8 or self.val_split != 0.1):
            logger.info(
                "Streaming datasets typically use predefined splits; "
                "custom split ratios may be ignored"
            )
        return self


class TrainingConfig(BaseModel):
    """Training hyperparameters and optimization configuration.

    Covers the full training loop: epochs, learning rate schedule,
    gradient handling, checkpointing, and mixed-precision training.
    """

    epochs: int = Field(default=10, description="Number of training epochs to run", ge=1)
    lr: float = Field(default=1e-3, description="Initial learning rate", gt=0.0)
    weight_decay: float = Field(
        default=0.01, description="Weight decay coefficient (L2 penalty)", ge=0.0
    )
    beta1: float = Field(default=0.9, description="Adam beta1 (first moment decay)", ge=0.0, lt=1.0)
    beta2: float = Field(
        default=0.999, description="Adam beta2 (second moment decay)", ge=0.0, lt=1.0
    )
    eps: float = Field(default=1e-8, description="Adam epsilon for numerical stability", gt=0.0)
    optimizer: Literal["adamw", "adam", "sgd"] = Field(
        default="adamw", description="Optimizer algorithm"
    )
    scheduler: Literal["cosine", "linear", "constant", "cosine_with_restarts"] = Field(
        default="cosine", description="Learning rate scheduler type"
    )
    warmup_ratio: float = Field(
        default=0.0,
        description="Fraction of total steps to use for linear warmup (0.0 = no warmup)",
        ge=0.0,
        le=1.0,
    )
    warmup_steps: int = Field(
        default=0, description="Number of linear warmup steps (overrides warmup_ratio)", ge=0
    )
    grad_clip: float = Field(default=1.0, description="Max gradient norm for clipping", gt=0.0)
    accum_steps: int = Field(
        default=1,
        description="Gradient accumulation steps (effective batch = batch_size * accum_steps)",
        ge=1,
    )
    eval_every: int = Field(default=1000, description="Run evaluation every N training steps", ge=1)
    save_every: int = Field(
        default=5000, description="Save checkpoint every N training steps", ge=1
    )
    save_total_limit: int = Field(
        default=3, description="Maximum number of checkpoints to retain", ge=1
    )
    early_stopping_patience: int = Field(
        default=0,
        description="Early stopping patience in evaluation rounds (0 = disabled)",
        ge=0,
    )
    early_stopping_metric: str = Field(
        default="eval_loss", description="Metric to monitor for early stopping"
    )
    early_stopping_mode: Literal["min", "max"] = Field(
        default="min", description="Whether to minimize or maximize the monitored metric"
    )
    mixed_precision: Literal["no", "fp16", "bf16", "fp8"] = Field(
        default="bf16", description="Mixed precision training mode"
    )
    compile_model: bool = Field(
        default=False, description="Use torch.compile() for model optimisation"
    )
    ddp: bool = Field(default=False, description="Use DistributedDataParallel training")
    ddp_find_unused_parameters: bool = Field(
        default=False, description="Set find_unused_parameters in DDP"
    )
    logging_steps: int = Field(default=10, description="Log metrics every N steps", ge=1)
    max_steps: int | None = Field(
        default=None, description="Hard limit on total training steps (None = epoch-based)", ge=1
    )
    resume_from_checkpoint: str | None = Field(
        default=None, description="Path to checkpoint to resume training from"
    )

    @field_validator("warmup_steps")
    @classmethod
    def _validate_warmup(cls, v: int, info: Any) -> int:
        """Warmup steps must be non-negative."""
        if v < 0:
            raise ValueError("warmup_steps must be >= 0")
        return v

    @model_validator(mode="after")
    def _validate_early_stopping(self) -> TrainingConfig:
        """Ensure early stopping patience is sensible."""
        if self.early_stopping_patience > 0 and self.eval_every <= 0:
            raise ValueError("eval_every must be > 0 when early_stopping_patience is enabled")
        return self


class EvalConfig(BaseModel):
    """Evaluation configuration.

    Specifies which metrics to compute, how often to evaluate during
    training, and whether to persist predictions.
    """

    metrics: list[str] = Field(
        default_factory=lambda: ["accuracy", "f1"],
        description="Metrics to compute (e.g., accuracy, f1, precision, recall, bleu, rouge)",
    )
    eval_frequency: int = Field(
        default=1000, description="Run evaluation every N training steps", ge=1
    )
    save_predictions: bool = Field(
        default=False, description="Save model predictions to disk for analysis"
    )
    predictions_format: Literal["json", "jsonl", "csv"] = Field(
        default="jsonl", description="Output format for saved predictions"
    )
    num_eval_batches: int | None = Field(
        default=None,
        description="Limit evaluation to N batches (None = full evaluation)",
        ge=1,
    )
    generation_max_length: int = Field(
        default=128, description="Maximum generation length for seq2seq evaluation", ge=1
    )
    generation_num_beams: int = Field(
        default=1, description="Number of beams for beam search generation", ge=1
    )
    compute_perplexity: bool = Field(
        default=True, description="Compute perplexity on the evaluation set"
    )

    @field_validator("metrics", mode="before")
    @classmethod
    def _coerce_metrics(cls, v: str | list[str]) -> list[str]:
        """Allow metrics to be specified as a comma-separated string."""
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v


class ExperimentConfig(BaseModel):
    """Full experiment configuration composing all sub-configs.

    This is the top-level configuration object that specifies an entire
    experiment including model architecture, data, training, and evaluation.
    All sub-configurations are nested and validated together.

    Example:
        Load from YAML::

            cfg = ExperimentConfig.from_yaml("configs/experiment.yaml")

        Save to YAML::

            cfg.to_yaml("configs/experiment.yaml")
    """

    name: str = Field(default="experiment", description="Experiment name for logging")
    description: str = Field(default="", description="Human-readable experiment description")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    tags: list[str] = Field(
        default_factory=list, description="Tags for W&B experiment organisation"
    )
    output_dir: str = Field(
        default="./results", description="Directory for checkpoints, logs, and outputs"
    )
    overwrite_output_dir: bool = Field(
        default=False, description="Overwrite existing output directory contents"
    )
    report_to: list[Literal["wandb", "tensorboard", "none"]] = Field(
        default_factory=lambda: ["wandb"],
        description="Experiment tracking integrations to enable",
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def _validate_consistency(self) -> ExperimentConfig:
        """Cross-subconfig validation for common misconfigurations."""
        # Ensure eval frequencies match between training and eval configs
        if self.training.eval_every != self.eval.eval_frequency:
            # Prefer the training config value — they should be in sync
            self.eval.eval_frequency = self.training.eval_every

        # Streaming datasets need streaming-compatible settings
        if self.data.streaming and self.data.num_workers > 0:
            # Streaming + multiprocessing can be problematic — warn but don't error
            pass

        return self

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> ExperimentConfig:
        """Load experiment configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Parsed :class:`ExperimentConfig` instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML content is malformed or contains
                invalid configuration values.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if not isinstance(raw, dict):
            raise ValueError(
                f"YAML file must contain a top-level mapping, got {type(raw).__name__}"
            )

        try:
            return cls(**raw)
        except Exception as exc:
            raise ValueError(f"Invalid configuration in {path}: {exc}") from exc

    def to_yaml(self, path: str | os.PathLike) -> None:
        """Save experiment configuration to a YAML file.

        The parent directory is created automatically if it does not exist.

        Args:
            path: Path where the YAML file will be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Pydantic v2 model_dump() replaces dict()
        payload = self.model_dump(mode="json")

        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(
                payload,
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=120,
            )
