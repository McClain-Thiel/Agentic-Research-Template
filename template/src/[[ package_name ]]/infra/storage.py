"""Storage abstraction using fsspec for unified local/S3/GCS access.

All file I/O should go through this module — never use raw ``open()`` for data files.
The backend is auto-detected from :data:`settings.STORAGE_BACKEND` and the
filesystem is created lazily on first access.

Example::

    from [[ package_name ]].infra.storage import storage_path, push, pull, sync_results

    # Get a full storage URI for a relative path
    path = storage_path("runs/experiment-1/checkpoint.pt")

    # Upload a local file
    push("./local_model.pt", "runs/experiment-1/model.pt")

    # Sync all results to cloud storage
    sync_results()
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import fsspec
from loguru import logger
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)

from [[ package_name ]].config.settings import settings

# ---------------------------------------------------------------------------
# Filesystem factory
# ---------------------------------------------------------------------------


def _get_fs() -> fsspec.AbstractFileSystem:
    """Lazily create the fsspec filesystem based on :data:`settings`."""
    backend: str = settings.STORAGE_BACKEND

    if backend == "local":
        logger.debug("Using local filesystem backend (root={})", settings.STORAGE_ROOT)
        return fsspec.filesystem("file")

    if backend == "s3":
        logger.debug(
            "Using S3 backend (bucket={}, region={})",
            settings.S3_BUCKET,
            settings.AWS_DEFAULT_REGION,
        )
        return fsspec.filesystem(
            "s3",
            key=settings.AWS_ACCESS_KEY_ID or None,
            secret=settings.AWS_SECRET_ACCESS_KEY or None,
            client_kwargs={"region_name": settings.AWS_DEFAULT_REGION},
        )

    if backend == "gcs":
        logger.debug("Using GCS backend (bucket={})", settings.GCS_BUCKET)
        return fsspec.filesystem(
            "gcs",
            token=settings.GOOGLE_APPLICATION_CREDENTIALS or None,
        )

    raise ValueError(f"Unknown storage backend '{backend}'. Choose one of: local, s3, gcs")


# Module-level filesystem singleton — lazily initialised.
_fs_instance: fsspec.AbstractFileSystem | None = None


def fs() -> fsspec.AbstractFileSystem:
    """Return the configured fsspec filesystem instance.

    The filesystem is created on first call and reused thereafter.
    """
    global _fs_instance
    if _fs_instance is None:
        _fs_instance = _get_fs()
    return _fs_instance


def _reset_fs() -> None:
    """Reset the filesystem singleton (useful in tests)."""
    global _fs_instance
    _fs_instance = None


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def storage_path(relative: str) -> str:
    """Prepend the storage root to a relative path.

    Args:
        relative: Path relative to the storage root.

    Returns:
        Full storage URI (e.g., ``s3://bucket/results/foo`` or ``./results/foo``).
    """
    relative = relative.lstrip("/")
    root: str = settings.STORAGE_ROOT

    if settings.STORAGE_BACKEND == "s3":
        bucket: str = settings.S3_BUCKET
        return f"s3://{bucket}/{root.strip('/')}/{relative}"
    elif settings.STORAGE_BACKEND == "gcs":
        bucket = settings.GCS_BUCKET
        return f"gs://{bucket}/{root.strip('/')}/{relative}"
    else:
        # local
        return os.path.join(root, relative)


# ---------------------------------------------------------------------------
# Upload / download with progress
# ---------------------------------------------------------------------------


def push(local: str, remote: str) -> None:
    """Upload a local file or directory to storage.

    Args:
        local: Local path to upload (file or directory).
        remote: Remote storage path (relative or absolute URI).
    """
    local_path = Path(local)
    if not local_path.exists():
        raise FileNotFoundError(f"Local path not found: {local}")

    filesystem = fs()
    remote_uri = storage_path(remote) if not remote.startswith(("s3://", "gs://")) else remote

    if local_path.is_dir():
        _push_dir(local_path, remote_uri, filesystem)
    else:
        _push_file(local, remote_uri, filesystem)


def _push_file(local_file: str, remote_uri: str, filesystem: fsspec.AbstractFileSystem) -> None:
    """Upload a single file with a Rich progress bar."""
    total_size = os.path.getsize(local_file)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Uploading[/bold blue] {task.fields[filename]}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        " ",
        DownloadColumn(),
        " ",
        TransferSpeedColumn(),
        " ",
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("upload", total=total_size, filename=Path(local_file).name)

        with open(local_file, "rb") as src:
            with filesystem.open(remote_uri, "wb") as dst:
                chunk_size = 1024 * 1024  # 1 MiB
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    progress.update(task, advance=len(chunk))

    logger.info("Uploaded {} → {}", local_file, remote_uri)


def _push_dir(local_dir: Path, remote_uri: str, filesystem: fsspec.AbstractFileSystem) -> None:
    """Upload an entire directory tree."""
    files = list(local_dir.rglob("*"))
    files = [f for f in files if f.is_file()]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Uploading directory[/bold blue] {task.fields[dir]}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        " ",
        TextColumn("{task.completed}/{task.total} files"),
        transient=True,
    ) as progress:
        task = progress.add_task("upload_dir", total=len(files), dir=local_dir.name)

        for file_path in files:
            rel = file_path.relative_to(local_dir)
            remote_file = f"{remote_uri}/{rel}".replace("//", "/")
            filesystem.makedirs(str(remote_file).rsplit("/", 1)[0], exist_ok=True)
            with open(file_path, "rb") as src:
                with filesystem.open(remote_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            progress.update(task, advance=1)

    logger.info(
        "Uploaded directory {} → {} ({} files)",
        local_dir,
        remote_uri,
        len(files),
    )


def pull(remote: str, local: str) -> None:
    """Download a file or directory from storage to local.

    Args:
        remote: Remote storage path (relative or absolute URI).
        local: Local destination path.
    """
    filesystem = fs()
    remote_uri = storage_path(remote) if not remote.startswith(("s3://", "gs://")) else remote

    # Check if remote is a directory by listing
    try:
        entries = filesystem.listdir(remote_uri)
        is_dir = any(e for e in entries)
    except OSError:
        is_dir = False

    local_path = Path(local)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if is_dir:
        _pull_dir(remote_uri, local_path, filesystem)
    else:
        _pull_file(remote_uri, local, filesystem)


def _pull_file(remote_uri: str, local_file: str, filesystem: fsspec.AbstractFileSystem) -> None:
    """Download a single file with a Rich progress bar."""
    info = filesystem.info(remote_uri)
    total_size = info.get("size", 0)

    Path(local_file).parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Downloading[/bold green] {task.fields[filename]}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        " ",
        DownloadColumn(),
        " ",
        TransferSpeedColumn(),
        " ",
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("download", total=total_size, filename=Path(local_file).name)

        with filesystem.open(remote_uri, "rb") as src:
            with open(local_file, "wb") as dst:
                chunk_size = 1024 * 1024  # 1 MiB
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
                    progress.update(task, advance=len(chunk))

    logger.info("Downloaded {} → {}", remote_uri, local_file)


def _pull_dir(remote_uri: str, local_dir: Path, filesystem: fsspec.AbstractFileSystem) -> None:
    """Download an entire directory tree."""
    # Recursively find all files
    all_files: list[str] = []
    _walk_remote(filesystem, remote_uri, all_files)

    local_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Downloading directory[/bold green] {task.fields[dir]}"),
        BarColumn(bar_width=40),
        "[progress.percentage]{task.percentage:>3.0f}%",
        " ",
        TextColumn("{task.completed}/{task.total} files"),
        transient=True,
    ) as progress:
        task = progress.add_task("pull_dir", total=len(all_files), dir=local_dir.name)

        for remote_file in all_files:
            rel = remote_file[len(remote_uri) :].lstrip("/")
            local_file = local_dir / rel
            local_file.parent.mkdir(parents=True, exist_ok=True)
            with filesystem.open(remote_file, "rb") as src:
                with open(local_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            progress.update(task, advance=1)

    logger.info(
        "Downloaded directory {} → {} ({} files)",
        remote_uri,
        local_dir,
        len(all_files),
    )


def _walk_remote(
    filesystem: fsspec.AbstractFileSystem,
    path: str,
    accumulator: list[str],
) -> None:
    """Recursively list all files under *path* into *accumulator*."""
    try:
        entries = filesystem.listdir(path)
    except Exception:
        return
    for entry in entries:
        name = entry.get("name", "")
        kind = entry.get("type", "")
        if kind == "directory" or entry.get("StorageClass") == "DIRECTORY":
            _walk_remote(filesystem, name, accumulator)
        elif kind == "file":
            accumulator.append(name)


# ---------------------------------------------------------------------------
# Batch sync operations
# ---------------------------------------------------------------------------


def sync_results() -> None:
    """Sync local ``results/`` and ``journal/runs/`` directories to storage.

        This should be called after training completes to persist all outputs
    to the configured cloud or local storage backend.
    """
    for local_dir, remote_prefix in [
        ("./results", "results"),
        ("./journal/runs", "journal/runs"),
    ]:
        if not Path(local_dir).exists():
            logger.debug("Skipping sync — local directory not found: {}", local_dir)
            continue
        remote = storage_path(remote_prefix)
        logger.info("Syncing {} → {}", local_dir, remote)
        push(local_dir, remote)
    logger.info("Result sync complete")


def pull_results() -> None:
    """Sync results from storage to local.

    This is the inverse of :func:`sync_results` — useful when resuming work
    on a different machine or pulling results from cloud storage.
    """
    for remote_prefix, local_dir in [
        ("results", "./results"),
        ("journal/runs", "./journal/runs"),
    ]:
        remote = storage_path(remote_prefix)
        logger.info("Pulling {} → {}", remote, local_dir)
        try:
            pull(remote, local_dir)
        except Exception as exc:
            logger.warning("Failed to pull {}: {}", remote_prefix, exc)
    logger.info("Result pull complete")
