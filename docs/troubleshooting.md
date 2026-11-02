# Troubleshooting

## `ModuleNotFoundError: No module named 'nba_shots'`

**Symptom:** `pytest` or `nba-shots` fails with this error after working earlier,
often intermittently. `uv sync` does not fix it.

**Cause:** Python skips hidden `.pth` files, and `.venv` uses one to find `src/`.
Seen on macOS when something outside the repo keeps setting the hidden flag on
dot-folders. To check, run:

```bash
ls -lO .venv/lib/python3.11/site-packages/*.pth
```

If the output shows `hidden`, this is the problem. Running `chflags nohidden` does
not last if the flag gets set again; in our case it returned within a second.

**Workaround:** add `PYTHONPATH=src` before commands, from the repo root:

```bash
PYTHONPATH=src uv run --frozen --all-extras pytest -q
PYTHONPATH=src uv run --frozen --all-extras nba-shots --help
```

Or set it once for the terminal session:

```bash
export PYTHONPATH=src
```
