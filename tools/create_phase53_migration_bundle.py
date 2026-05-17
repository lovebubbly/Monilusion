from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "migration_bundles"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file():
                yield child


def _default_include_paths(root: Path) -> list[Path]:
    output = root / "wfa_optimized_params_output"
    return [
        root / "data" / "BTCUSDT_1h.csv",
        output / "futures_context" / "BTCUSDT_funding_rate_8h_20190101_20260516.csv",
        output / "phase53_active_portfolio_shadow_manifest.json",
        output / "phase53_active_portfolio_shadow_registry.json",
        output / "wfo_portfolio_phase53_dsr_meta2_full_n1771_20260517.json",
        output / "phase53_current_train_refresh_20260517_active_registry_from_existing",
        output / "phase53_active_live_shadow",
    ]


def create_bundle(*, out: Path, root: Path, include_paths: list[Path]) -> dict[str, object]:
    files: list[Path] = []
    missing: list[str] = []
    for path in include_paths:
        resolved = path if path.is_absolute() else root / path
        if not resolved.exists():
            missing.append(str(resolved))
            continue
        files.extend(_iter_files(resolved))
    files = sorted({path.resolve() for path in files})
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            rel = path.relative_to(root)
            zf.write(path, rel.as_posix())
            manifest_files.append({
                "path": rel.as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            })
        bundle_manifest = {
            "schema_version": 1,
            "mode": "phase53_mac_migration_bundle",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(root),
            "file_count": len(manifest_files),
            "missing_paths": missing,
            "files": manifest_files,
        }
        zf.writestr("MIGRATION_MANIFEST.json", json.dumps(bundle_manifest, ensure_ascii=False, indent=2))
    return {
        "bundle": str(out),
        "file_count": len(manifest_files),
        "missing_paths": missing,
        "sha256": _sha256(out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a zip bundle of ignored Phase53 live-shadow artifacts for Mac migration.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--include", action="append", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or DEFAULT_OUTPUT_DIR / f"phase53_mac_migration_bundle_{stamp}.zip"
    include_paths = args.include or _default_include_paths(ROOT)
    report = create_bundle(out=out if out.is_absolute() else ROOT / out, root=ROOT, include_paths=include_paths)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
