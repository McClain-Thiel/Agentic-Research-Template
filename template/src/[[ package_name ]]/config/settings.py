"""Application settings loaded from environment variables.

All settings are loaded from environment variables or a ``.env`` file at
project root.  See ``.env.example`` for the full list of available settings.
"""

from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.table import Table


class Settings(BaseSettings):
    """Project settings loaded from environment variables.

    All settings can be overridden via environment variables or a ``.env`` file.
    See ``.env.example`` for the full list of available settings.

    Attributes:
        WANDB_API_KEY: Weights & Biases API key for experiment tracking.
        WANDB_PROJECT: W&B project name.
        WANDB_ENTITY: W&B team or user entity.
        HF_TOKEN: Hugging Face access token.
        HF_HOME: Directory for Hugging Face cache files.
        STORAGE_BACKEND: Storage backend type (local, s3, or gcs).
        STORAGE_ROOT: Root path for results storage.
        AWS_ACCESS_KEY_ID: AWS access key for S3 storage.
        AWS_SECRET_ACCESS_KEY: AWS secret key for S3 storage.
        AWS_DEFAULT_REGION: Default AWS region.
        S3_BUCKET: S3 bucket name for cloud storage.
        GOOGLE_APPLICATION_CREDENTIALS: Path to GCS service account key.
        GCS_BUCKET: GCS bucket name for cloud storage.
        SLURM_MODULES: Comma-separated list of SLURM modules to load.
        SLURM_ACCOUNT: SLURM account name for job submission.
        SLURM_PARTITION: SLURM partition/queue name.
        SLURM_QOS: SLURM quality-of-service tier.
        SLURM_TIME: Maximum wall-clock time per job (HH:MM:SS).
        SLURM_GPUS: Number of GPUs to request per job.
        SLURM_CPUS_PER_GPU: CPU cores per GPU.
        SLURM_MEM: Memory allocation per job (e.g., ``16G``).
        EC2_INSTANCE_ID: EC2 instance ID for direct compute.
        EC2_LAUNCH_TEMPLATE: EC2 launch template ID.
        EC2_KEY_NAME: SSH key pair name for EC2 access.
        EC2_SSH_USER: SSH username for EC2 instances.
        EC2_REGION: EC2/AWS region.
        BATCH_JOB_DEFINITION: AWS Batch job definition ARN.
        BATCH_JOB_QUEUE: AWS Batch job queue name.
        BATCH_REGION: AWS Batch region.
        RAY_CLUSTER_ADDRESS: Ray cluster address (host:port).
        ANYSCALE_API_KEY: Anyscale API key for Ray cluster management.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── Weights & Biases ───
    WANDB_API_KEY: str = Field(default="", description="Weights & Biases API key")
    WANDB_PROJECT: str = Field(default="[[ package_name ]]", description="W&B project name")
    WANDB_ENTITY: str = Field(default="", description="W&B team or user entity")

    # ─── Hugging Face ───
    HF_TOKEN: str = Field(default="", description="Hugging Face access token")
    HF_HOME: str = Field(default="./data/hf_cache", description="Hugging Face cache directory")

    # ─── Storage Backend ───
    STORAGE_BACKEND: Literal["local", "s3", "gcs"] = Field(
        default="local", description="Storage backend: local, s3, or gcs"
    )
    STORAGE_ROOT: str = Field(default="./results", description="Root path for results storage")

    # ─── AWS / S3 ───
    AWS_ACCESS_KEY_ID: str = Field(default="", description="AWS access key ID")
    AWS_SECRET_ACCESS_KEY: str = Field(default="", description="AWS secret access key")
    AWS_DEFAULT_REGION: str = Field(default="us-east-1", description="Default AWS region")
    S3_BUCKET: str = Field(default="", description="S3 bucket name")

    # ─── Google Cloud Storage ───
    GOOGLE_APPLICATION_CREDENTIALS: str = Field(
        default="", description="Path to GCS service account JSON key"
    )
    GCS_BUCKET: str = Field(default="", description="GCS bucket name")

    # ─── SLURM ───
    SLURM_MODULES: str = Field(default="", description="Comma-separated SLURM modules to load")
    SLURM_ACCOUNT: str = Field(default="", description="SLURM account name")
    SLURM_PARTITION: str = Field(default="", description="SLURM partition name")
    SLURM_QOS: str = Field(default="", description="SLURM QoS tier")
    SLURM_TIME: str = Field(default="01:00:00", description="Max wall time (HH:MM:SS)")
    SLURM_GPUS: int = Field(default=1, description="GPUs per job", ge=0)
    SLURM_CPUS_PER_GPU: int = Field(default=4, description="CPU cores per GPU", ge=1)
    SLURM_MEM: str = Field(default="16G", description="Memory per job")

    # ─── EC2 ───
    EC2_INSTANCE_ID: str = Field(default="", description="EC2 instance ID")
    EC2_LAUNCH_TEMPLATE: str = Field(default="", description="EC2 launch template ID")
    EC2_KEY_NAME: str = Field(default="", description="EC2 SSH key pair name")
    EC2_SSH_USER: str = Field(default="ubuntu", description="EC2 SSH username")
    EC2_REGION: str = Field(default="us-east-1", description="EC2 region")

    # ─── AWS Batch ───
    BATCH_JOB_DEFINITION: str = Field(default="", description="AWS Batch job definition ARN")
    BATCH_JOB_QUEUE: str = Field(default="", description="AWS Batch job queue name")
    BATCH_REGION: str = Field(default="us-east-1", description="AWS Batch region")

    # ─── Ray / Anyscale ───
    RAY_CLUSTER_ADDRESS: str = Field(default="", description="Ray cluster host:port")
    ANYSCALE_API_KEY: str = Field(default="", description="Anyscale API key")

    # ─── Validators ───

    @field_validator("SLURM_MODULES")
    @classmethod
    def _parse_slurm_modules(cls, v: str) -> str:
        """Strip whitespace from comma-separated module list."""
        if not v:
            return v
        modules = [m.strip() for m in v.split(",") if m.strip()]
        return ",".join(modules)

    # ─── Verification ───

    def verify(self) -> bool:
        """Verify that all required settings are configured.

        Prints a Rich table showing each setting's name, status (green
        check / red cross), and value (secrets are masked).  Returns
        whether **all** required fields are populated.

        Returns:
            ``True`` if all required settings are configured, ``False``
            otherwise.
        """
        console = Console()
        table = Table(title="[[ package_name ]] — Configuration Status", show_lines=True)
        table.add_column("Setting", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center", style="bold")
        table.add_column("Value", style="dim")

        # Define required fields and their conditional requirements
        required_when: dict[str, bool] = {
            "WANDB_API_KEY": bool(self.WANDB_PROJECT),  # needed if tracking enabled
            "HF_TOKEN": True,  # Always warn if unset
        }

        # Cloud-specific required fields
        if self.STORAGE_BACKEND == "s3":
            required_when["S3_BUCKET"] = True
        elif self.STORAGE_BACKEND == "gcs":
            required_when["GCS_BUCKET"] = True

        # Secret fields that should be masked in output
        secret_fields: set[str] = {
            "WANDB_API_KEY",
            "HF_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "ANYSCALE_API_KEY",
        }

        all_ok = True
        for field_name in self.model_fields:
            value: str | int = getattr(self, field_name)
            is_required = field_name in required_when and required_when[field_name]
            is_set = bool(value) and str(value) != ""

            if is_required and not is_set:
                status = "[red]✗[/red]"
                all_ok = False
                logger.warning(
                    "Required setting {} is not set — some features may be unavailable",
                    field_name,
                )
            elif is_set:
                status = "[green]✓[/green]"
            else:
                status = "[dim]−[/dim]"

            # Mask secret values
            if field_name in secret_fields and is_set and isinstance(value, str):
                display_value = value[:4] + "***" if len(value) > 4 else "***"
            else:
                display_value = str(value)

            table.add_row(field_name, status, display_value)

        console.print(table)
        return all_ok


# Module-level singleton — imported once and reused across the package.
settings = Settings()
