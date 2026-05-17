from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OLD_ROOT = r"C:\Monilusion"
ACTIVE_REFRESH_DIR = (
    ROOT
    / "wfa_optimized_params_output"
    / "phase53_current_train_refresh_20260517_active_registry_from_existing"
)


def _normalize_path_text(value: str) -> str:
    return value.replace("\\", "/").rstrip("/")


def _rebase_string(value: str, *, old_root: str, new_root: Path) -> tuple[str, bool]:
    old_norm = _normalize_path_text(old_root)
    value_norm = value.replace("\\", "/")
    if value_norm.lower() == old_norm.lower():
        return str(new_root), True
    prefix = old_norm + "/"
    if not value_norm.lower().startswith(prefix.lower()):
        return value, False
    relative = value_norm[len(prefix):]
    return str(new_root / Path(relative)), True


def _rebind_obj(value: Any, *, old_root: str, new_root: Path) -> tuple[Any, int]:
    if isinstance(value, dict):
        changed = 0
        out: dict[str, Any] = {}
        for key, item in value.items():
            rebound, count = _rebind_obj(item, old_root=old_root, new_root=new_root)
            out[key] = rebound
            changed += count
        return out, changed
    if isinstance(value, list):
        changed = 0
        out = []
        for item in value:
            rebound, count = _rebind_obj(item, old_root=old_root, new_root=new_root)
            out.append(rebound)
            changed += count
        return out, changed
    if isinstance(value, str):
        rebound, did_change = _rebase_string(value, old_root=old_root, new_root=new_root)
        return rebound, int(did_change)
    return value, 0


def _iter_default_files(root: Path) -> list[Path]:
    output = root / "wfa_optimized_params_output"
    paths = [
        output / "phase53_active_portfolio_shadow_manifest.json",
        output / "phase53_active_portfolio_shadow_registry.json",
        output / "wfo_portfolio_phase53_dsr_meta2_full_n1771_20260517.json",
    ]
    active_live = output / "phase53_active_live_shadow"
    if active_live.exists():
        paths.extend(path for path in active_live.rglob("*") if path.suffix in {".json", ".jsonl"})
    if ACTIVE_REFRESH_DIR.exists():
        paths.extend(path for path in ACTIVE_REFRESH_DIR.rglob("*.json"))
    return sorted({path.resolve() for path in paths if path.exists()})


def _rebind_json(path: Path, *, old_root: str, new_root: Path, write: bool) -> dict[str, Any]:
    original = json.loads(path.read_text(encoding="utf-8-sig"))
    rebound, changed = _rebind_obj(original, old_root=old_root, new_root=new_root)
    if write and changed:
        path.write_text(json.dumps(rebound, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(path), "changed_strings": changed, "written": bool(write and changed)}


def _rebind_jsonl(path: Path, *, old_root: str, new_root: Path, write: bool) -> dict[str, Any]:
    rows: list[Any] = []
    changed = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        rebound, count = _rebind_obj(row, old_root=old_root, new_root=new_root)
        rows.append(rebound)
        changed += count
    if write and changed:
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return {"path": str(path), "changed_strings": changed, "written": bool(write and changed)}


def rebind_files(files: list[Path], *, old_root: str, new_root: Path, write: bool) -> dict[str, Any]:
    results = []
    for path in files:
        if path.suffix == ".jsonl":
            results.append(_rebind_jsonl(path, old_root=old_root, new_root=new_root, write=write))
        elif path.suffix == ".json":
            results.append(_rebind_json(path, old_root=old_root, new_root=new_root, write=write))
    return {
        "schema_version": 1,
        "mode": "phase53_path_rebind",
        "old_root": old_root,
        "new_root": str(new_root),
        "write": write,
        "file_count": len(results),
        "changed_file_count": sum(1 for row in results if int(row["changed_strings"]) > 0),
        "changed_string_count": sum(int(row["changed_strings"]) for row in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebind Phase53 artifact paths from a Windows root to this checkout.")
    parser.add_argument("--old-root", default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=ROOT)
    parser.add_argument("--write", action="store_true", help="Write changed JSON/JSONL files. Without this, dry-run only.")
    parser.add_argument("--file", action="append", type=Path, default=None, help="Specific JSON/JSONL file to rebind.")
    args = parser.parse_args()

    new_root = args.new_root.resolve()
    files = [path if path.is_absolute() else ROOT / path for path in args.file] if args.file else _iter_default_files(ROOT)
    report = rebind_files(files, old_root=args.old_root, new_root=new_root, write=args.write)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
