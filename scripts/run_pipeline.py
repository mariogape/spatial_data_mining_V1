#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Spatial Data Mining job YAML through the pipeline."
    )
    parser.add_argument(
        "job",
        help="Path to a job YAML file (e.g., config/jobs/example.yaml)",
    )
    parser.add_argument(
        "--base-config",
        default="config/base.yaml",
        help="Path to base config YAML (default: config/base.yaml)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress callback output (logging still applies)",
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    if not (project_root / "src").exists():
        project_root = Path(__file__).resolve().parents[1]

    job_path = _resolve_path(project_root, args.job)
    base_path = _resolve_path(project_root, args.base_config)

    if not job_path.exists():
        print(f"Job config not found: {job_path}", file=sys.stderr)
        return 1
    if not base_path.exists():
        print(f"Base config not found: {base_path}", file=sys.stderr)
        return 1

    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from spatial_data_mining.orchestrator import run_pipeline

    def progress(message: str) -> None:
        print(message)

    results = run_pipeline(
        str(job_path),
        progress_cb=None if args.quiet else progress,
        base_config_path=str(base_path),
    )

    if not results:
        print("Pipeline completed but returned no outputs (check logs).")
        return 0

    print("Pipeline completed. Outputs:")
    for res in results:
        print(
            f"- {res.get('aoi')} {res.get('variable')} ({res.get('year')} {res.get('season')}): "
            f"local={res.get('local_path')} gcs={res.get('gcs_uri')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
