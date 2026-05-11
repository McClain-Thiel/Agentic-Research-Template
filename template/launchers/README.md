# Launchers

Launchers are scripts that wrap training commands and manage compute resources.

## Status File Format

Each launcher writes and updates a status YAML file in `journal/runs/{run_id}.yaml`:

```yaml
run_id: <run_id>
launcher: <local|slurm|ec2|aws_batch|ray>
status: <pending|running|completed|failed>
pid: <process_id or job_id>
host: <hostname or instance_id>
started_at: <ISO timestamp>
completed_at: <ISO timestamp or null>
command: <full command that was run>
```

## Launcher Contract

1. Write a `pending` status file before starting
2. Update to `running` once the job starts
3. Run the actual training command
4. On success: update to `completed`, call `push-results`
5. On failure: update to `failed`, include error message
6. Write `completed_at` timestamp on terminal state

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LAUNCHER` | Which launcher to use (default: `local`) |
| `ENV_FILE` | Path to env file to source before running |
| `STORAGE_BACKEND` | Storage backend: `local`, `s3`, or `gcs` |
| `STORAGE_ROOT` | Root path for storage |
