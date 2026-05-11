# Evaluation Scripts

Place evaluation scripts in this directory. Each script should:

1. Accept `--run-id` as a CLI argument
2. Load the model/checkpoint for that run
3. Run evaluation on the appropriate dataset
4. Save results to `results/{run_id}/{eval_name}.json`

## Running Evals

```bash
# Run a single eval
just eval evals/my_eval.py run_20240101_120000

# Run all evals
just eval-all run_20240101_120000
```

## Result Format

Each eval script should produce a JSON file matching the `EvalResult` schema:
```json
{
  "run_id": "run_20240101_120000",
  "eval_name": "my_eval",
  "timestamp": "2024-01-01T12:00:00Z",
  "metrics": {
    "accuracy": 0.95,
    "f1": 0.94
  },
  "config": {}
}
```
