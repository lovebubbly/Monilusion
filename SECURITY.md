# Security Notes

## Secrets

- Keep real credentials only in `.env`.
- Commit `.env.example`, never `.env`.
- Use Binance API keys with withdrawal disabled, minimum required futures permissions, and IP allowlisting where possible.
- Rotate any key that was ever committed, pasted into logs, shared in chat, or used from an untrusted machine.

## Live Trading Guard

`real_M1.py` is configured to be safe by default:

- `BINANCE_USE_TESTNET=true` unless explicitly set otherwise.
- `DRY_RUN=true` unless explicitly set otherwise.
- Live trading on a non-testnet account requires `ALLOW_LIVE_TRADING=YES_I_UNDERSTAND`.

Recommended production launch environment:

```text
BINANCE_USE_TESTNET=false
DRY_RUN=false
ALLOW_LIVE_TRADING=YES_I_UNDERSTAND
```

Do not set these values until the strategy has been validated with backtests, paper trading, and testnet execution.

## Artifacts

Model files, pickles, numpy arrays, parquet files, logs, and runtime state are ignored by default. Store large or executable artifacts in an external artifact store, Git LFS, DVC, or a private release process with checksums and model manifests.

Pickle, Joblib, and Torch checkpoint files can execute code during loading. Only load artifacts created by this project or a trusted source.

## Already Committed Artifacts

Removing files from `.gitignore` is not enough when they are already tracked. Remove generated/binary artifacts from Git tracking with `git rm --cached` and commit the deletion. If a real secret was committed, history must also be rewritten with a tool such as `git filter-repo` or BFG, followed by key rotation.
