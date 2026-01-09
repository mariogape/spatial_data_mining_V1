#!/usr/bin/env python3
"""
Cloud Run compatible script for running the spatial data mining pipeline.
This script replaces the notebook UI with command-line arguments.
"""
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / 'src'
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from spatial_data_mining.orchestrator import run_pipeline_from_dict
from spatial_data_mining.utils.cancellation import PipelineCancelled


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run spatial data mining pipeline on Cloud Run',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Job configuration
    parser.add_argument(
        '--job-name',
        default='cloudrun_job',
        help='Job name for logging'
    )

    # AOI configuration
    parser.add_argument(
        '--aoi-paths',
        nargs='+',
        required=True,
        help='Paths to AOI files (can be GCS URIs like gs://bucket/path/aoi.geojson)'
    )

    # Spatial configuration
    parser.add_argument(
        '--target-crs',
        default='EPSG:4326',
        help='Target CRS for outputs'
    )

    parser.add_argument(
        '--resolution-m',
        type=float,
        default=20.0,
        help='Resolution in meters (set to 0 for native resolution)'
    )

    # Temporal configuration
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        required=True,
        help='Years to process (e.g., 2020 2021 2022)'
    )

    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['summer'],
        help='Seasons to process (winter, spring, summer, autumn, annual)'
    )

    # Variables
    parser.add_argument(
        '--variables',
        nargs='+',
        required=True,
        help='Variables to extract (e.g., ndvi ndmi bsi)'
    )

    # Storage configuration
    parser.add_argument(
        '--storage-kind',
        choices=['local_cog', 'gcs_cog'],
        default='gcs_cog',
        help='Storage backend'
    )

    parser.add_argument(
        '--output-dir',
        default='/tmp/outputs',
        help='Local output directory (for local_cog storage)'
    )

    parser.add_argument(
        '--gcs-bucket',
        help='GCS bucket for outputs (required for gcs_cog storage)'
    )

    parser.add_argument(
        '--gcs-prefix',
        default='spatial/outputs',
        help='GCS prefix for outputs'
    )

    # Variable-specific options
    parser.add_argument(
        '--clcplus-input-dir',
        help='Directory containing CLCplus data (required if clcplus variable is selected)'
    )

    parser.add_argument(
        '--swi-date',
        help='Date for SWI variable (YYYY-MM-DD format, within season)'
    )

    parser.add_argument(
        '--rgb-date',
        help='Date for RGB variable (YYYY-MM-DD format, within season)'
    )

    # Performance
    parser.add_argument(
        '--max-concurrent-tasks',
        type=int,
        default=2,
        help='Maximum concurrent tasks (clamped to 2 for openEO free tier)'
    )

    parser.add_argument(
        '--base-config',
        default='config/base.yaml',
        help='Path to base config YAML'
    )

    return parser.parse_args()


def validate_args(args):
    """Validate arguments and provide helpful error messages."""
    errors = []

    # GCS storage validation
    if args.storage_kind == 'gcs_cog' and not args.gcs_bucket:
        errors.append('--gcs-bucket is required when --storage-kind=gcs_cog')

    # Variable-specific validation
    variables_lower = [v.lower() for v in args.variables]

    if 'clcplus' in variables_lower and not args.clcplus_input_dir:
        errors.append('--clcplus-input-dir is required when clcplus variable is selected')

    if 'swi' in variables_lower and len(args.years) > 1 or len(args.seasons) > 1:
        if not args.swi_date:
            print('WARNING: SWI with multiple years/seasons will use default mid-season dates')

    if 'rgb' in variables_lower and len(args.years) > 1 or len(args.seasons) > 1:
        if not args.rgb_date:
            print('WARNING: RGB with multiple years/seasons will use default mid-season dates')

    if errors:
        for error in errors:
            print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)


def build_job_config(args) -> dict:
    """Build job configuration dictionary from command-line arguments."""

    # Basic storage config
    storage_cfg = {
        'kind': args.storage_kind,
        'output_dir': args.output_dir
    }

    if args.storage_kind == 'gcs_cog':
        storage_cfg['bucket'] = args.gcs_bucket
        storage_cfg['prefix'] = args.gcs_prefix

    # Resolution (None = native)
    resolution_value = None if args.resolution_m == 0 else args.resolution_m

    # Build job section
    job_section = {
        'name': args.job_name,
        'aoi_path': args.aoi_paths[0],  # First one for backward compat
        'aoi_paths': args.aoi_paths,
        'target_crs': args.target_crs,
        'resolution_m': resolution_value,
        'year': args.years[0],  # First one for backward compat
        'years': args.years,
        'season': args.seasons[0],  # First one for backward compat
        'seasons': args.seasons,
        'variables': args.variables,
        'storage': storage_cfg,
    }

    # Optional configs
    if args.clcplus_input_dir:
        job_section['clcplus_input_dir'] = args.clcplus_input_dir

    if args.swi_date:
        job_section['swi_date'] = args.swi_date

    if args.rgb_date:
        job_section['rgb_date'] = args.rgb_date

    return job_section


def progress_callback(message: str):
    """Progress callback that prints to stdout for Cloud Run logging."""
    print(f'[PROGRESS] {message}', flush=True)


def main() -> int:
    args = parse_args()
    validate_args(args)

    # Set performance environment variable
    os.environ['SDM_MAX_CONCURRENT_TASKS'] = str(args.max_concurrent_tasks)

    print('=' * 80)
    print('Spatial Data Mining Pipeline - Cloud Run')
    print('=' * 80)
    print(f'Job name: {args.job_name}')
    print(f'AOIs: {len(args.aoi_paths)} file(s)')
    print(f'Variables: {", ".join(args.variables)}')
    print(f'Years: {", ".join(map(str, args.years))}')
    print(f'Seasons: {", ".join(args.seasons)}')
    print(f'Storage: {args.storage_kind}')
    if args.storage_kind == 'gcs_cog':
        print(f'GCS: gs://{args.gcs_bucket}/{args.gcs_prefix}')
    print(f'Max concurrent tasks: {args.max_concurrent_tasks}')
    print('=' * 80)
    print()

    # Build job config
    job_section = build_job_config(args)

    # Resolve base config path
    base_config_path = args.base_config
    if not Path(base_config_path).is_absolute():
        base_config_path = str(PROJECT_ROOT / base_config_path)

    try:
        results = run_pipeline_from_dict(
            job_section,
            progress_cb=progress_callback,
            should_stop=None,  # No interactive stop on Cloud Run
            base_config_path=base_config_path
        )
    except PipelineCancelled:
        print('Pipeline was cancelled.', file=sys.stderr)
        return 1
    except Exception as exc:
        print(f'Pipeline failed: {exc}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    if not results:
        print('Pipeline completed but returned no outputs (check logs for errors).')
        return 0

    print()
    print('=' * 80)
    print('Pipeline completed successfully!')
    print('=' * 80)
    print(f'Outputs ({len(results)} total):')
    for res in results:
        print(
            f"  - {res.get('aoi')} / {res.get('variable')} / "
            f"{res.get('year')} / {res.get('season')}"
        )
        if res.get('gcs_uri'):
            print(f"    GCS: {res.get('gcs_uri')}")
        if res.get('local_path'):
            print(f"    Local: {res.get('local_path')}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
