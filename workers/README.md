# Worker

Run the background worker with:

```bash
python -m workers.worker
```

The worker listens on Redis, executes cancellable stage jobs, and persists job status in each project's `jobs/` folder.

