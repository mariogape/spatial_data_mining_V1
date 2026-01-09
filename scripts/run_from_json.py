#!/usr/bin/env python3
"""
Run spatial data mining pipeline from JSON configuration file.
This script is used by the VM deployment to execute jobs.
"""
import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / 'src'
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from spatial_data_mining.orchestrator import run_pipeline_from_dict
from spatial_data_mining.utils.cancellation import PipelineCancelled

# Import metadata generation functions
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
try:
    from generate_job_metadata import generate_metadata, write_metadata_json, write_metadata_csv, write_summary_txt
except ImportError:
    generate_metadata = None
    print("WARNING: Could not import generate_job_metadata, tracking files will not be generated")


def load_json_config(config_path: str) -> Dict[str, Any]:
    """Load and validate JSON configuration."""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with config_file.open('r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)

    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate required fields in JSON configuration."""
    required_fields = {
        'job_name': str,
        'aoi_paths': list,
        'spatial_config': dict,
        'temporal_config': dict,
        'variables': list,
        'storage': dict,
    }

    errors = []

    for field, field_type in required_fields.items():
        if field not in config:
            errors.append(f"Missing required field: '{field}'")
        elif not isinstance(config[field], field_type):
            errors.append(f"Field '{field}' must be of type {field_type.__name__}")

    # Validate nested fields
    if 'spatial_config' in config:
        spatial = config['spatial_config']
        for key in ['target_crs', 'resolution_m']:
            if key not in spatial:
                errors.append(f"Missing required field in spatial_config: '{key}'")

    if 'temporal_config' in config:
        temporal = config['temporal_config']
        for key in ['years', 'seasons']:
            if key not in temporal:
                errors.append(f"Missing required field in temporal_config: '{key}'")

    if 'storage' in config:
        storage = config['storage']
        if 'kind' not in storage:
            errors.append("Missing required field in storage: 'kind'")
        elif storage['kind'] == 'gcs_cog' and 'bucket' not in storage:
            errors.append("Field 'bucket' is required when storage.kind='gcs_cog'")

    if errors:
        print("ERROR: Configuration validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


def json_to_pipeline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON config to pipeline configuration dict."""
    spatial = config['spatial_config']
    temporal = config['temporal_config']
    var_specific = config.get('variable_specific', {})
    performance = config.get('performance', {})

    # Handle resolution
    resolution_m = None
    if not spatial.get('use_native_resolution', False):
        resolution_m = spatial['resolution_m']

    # Build pipeline config
    pipeline_config = {
        'name': config['job_name'],
        'aoi_path': config['aoi_paths'][0],  # First one for backward compat
        'aoi_paths': config['aoi_paths'],
        'target_crs': spatial['target_crs'],
        'resolution_m': resolution_m,
        'year': temporal['years'][0],  # First one for backward compat
        'years': temporal['years'],
        'season': temporal['seasons'][0],  # First one for backward compat
        'seasons': temporal['seasons'],
        'variables': config['variables'],
        'storage': config['storage'],
    }

    # Add variable-specific configs if present
    if var_specific.get('clcplus_input_dir'):
        pipeline_config['clcplus_input_dir'] = var_specific['clcplus_input_dir']

    if var_specific.get('swi_date'):
        pipeline_config['swi_date'] = var_specific['swi_date']

    if var_specific.get('rgb_date'):
        pipeline_config['rgb_date'] = var_specific['rgb_date']

    return pipeline_config


def progress_callback(message: str):
    """Progress callback that prints to stdout."""
    print(f'[PROGRESS] {message}', flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run spatial data mining pipeline from JSON configuration'
    )

    parser.add_argument(
        '--job-config',
        required=True,
        help='Path to JSON job configuration file'
    )

    parser.add_argument(
        '--base-config',
        default='config/base.yaml',
        help='Path to base YAML configuration'
    )

    parser.add_argument(
        '--git-commit',
        help='Git commit SHA for tracking'
    )

    parser.add_argument(
        '--build-id',
        help='Cloud Build ID for tracking'
    )

    parser.add_argument(
        '--vm-name',
        help='VM instance name for tracking'
    )

    args = parser.parse_args()

    # Load and validate JSON config
    print('=' * 80)
    print('Spatial Data Mining Pipeline - JSON Config Mode')
    print('=' * 80)
    print(f'Job config: {args.job_config}')
    print(f'Base config: {args.base_config}')
    print('=' * 80)
    print()

    json_config = load_json_config(args.job_config)
    validate_config(json_config)

    # Set performance environment variables
    performance = json_config.get('performance', {})
    max_concurrent = performance.get('max_concurrent_tasks', 2)
    os.environ['SDM_MAX_CONCURRENT_TASKS'] = str(max_concurrent)

    # Convert to pipeline config
    pipeline_config = json_to_pipeline_config(json_config)

    # Print summary
    print(f'Job name: {pipeline_config["name"]}')
    print(f'Description: {json_config.get("description", "N/A")}')
    print(f'AOIs: {len(pipeline_config["aoi_paths"])} file(s)')
    print(f'Variables: {", ".join(pipeline_config["variables"])}')
    print(f'Years: {", ".join(map(str, pipeline_config["years"]))}')
    print(f'Seasons: {", ".join(pipeline_config["seasons"])}')
    print(f'Storage: {pipeline_config["storage"]["kind"]}')
    if pipeline_config["storage"]["kind"] == 'gcs_cog':
        bucket = pipeline_config["storage"]["bucket"]
        prefix = pipeline_config["storage"].get("prefix", "")
        print(f'Output: gs://{bucket}/{prefix}')
    print(f'Max concurrent tasks: {max_concurrent}')
    print('=' * 80)
    print()

    # Resolve base config path
    base_config_path = args.base_config
    if not Path(base_config_path).is_absolute():
        base_config_path = str(PROJECT_ROOT / base_config_path)

    # Run pipeline
    try:
        results = run_pipeline_from_dict(
            pipeline_config,
            progress_cb=progress_callback,
            should_stop=None,
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
        results = []  # Empty list for metadata generation

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

    # Generate tracking/metadata files
    print()
    print('=' * 80)
    print('Generating Tracking Metadata')
    print('=' * 80)

    if generate_metadata:
        try:
            # Determine output path
            storage = pipeline_config['storage']
            if storage['kind'] == 'gcs_cog':
                bucket = storage['bucket']
                prefix = storage.get('prefix', '')
                metadata_prefix = f"{prefix}/metadata" if prefix else "metadata"
                metadata_base_path = f"gs://{bucket}/{metadata_prefix}"
            else:
                output_dir = storage.get('output_dir', '/tmp/outputs')
                metadata_base_path = f"{output_dir}/metadata"

            # Generate metadata
            metadata = generate_metadata(
                job_config=pipeline_config,
                results=results,
                git_commit=args.git_commit,
                build_id=args.build_id,
                vm_name=args.vm_name,
                output_path=metadata_base_path,
            )

            # Create local temp directory for metadata files
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                job_name = pipeline_config['name']
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

                # Write files locally first
                json_file = write_metadata_json(
                    metadata,
                    f"{tmp_dir}/{job_name}_{timestamp}_metadata.json"
                )
                csv_file = write_metadata_csv(
                    metadata,
                    f"{tmp_dir}/{job_name}_{timestamp}_outputs.csv"
                )
                txt_file = write_summary_txt(
                    metadata,
                    f"{tmp_dir}/{job_name}_{timestamp}_summary.txt"
                )

                print(f"✅ Metadata files generated locally:")
                print(f"   - JSON: {json_file}")
                print(f"   - CSV:  {csv_file}")
                print(f"   - TXT:  {txt_file}")

                # Upload to GCS if using GCS storage
                if storage['kind'] == 'gcs_cog':
                    try:
                        from google.cloud import storage as gcs_storage
                        client = gcs_storage.Client()
                        bucket_obj = client.bucket(bucket)

                        for local_file in [json_file, csv_file, txt_file]:
                            filename = Path(local_file).name
                            blob_name = f"{metadata_prefix}/{filename}"
                            blob = bucket_obj.blob(blob_name)
                            blob.upload_from_filename(local_file)
                            gcs_uri = f"gs://{bucket}/{blob_name}"
                            print(f"📤 Uploaded: {gcs_uri}")

                        print(f"✅ Tracking metadata uploaded to GCS")
                    except Exception as e:
                        print(f"⚠️  Failed to upload metadata to GCS: {e}")
                        print(f"   Local files preserved in: {tmp_dir}")

        except Exception as e:
            print(f"⚠️  Failed to generate metadata: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️  Metadata generation not available (missing generate_job_metadata module)")

    print('=' * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
