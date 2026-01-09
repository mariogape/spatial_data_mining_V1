#!/usr/bin/env python3
"""
Generate job metadata files for tracking pipeline runs.
Creates CSV and JSON files with job information, git commit, and outputs.
"""
import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


def generate_metadata(
    job_config: Dict[str, Any],
    results: List[Dict[str, Any]],
    git_commit: Optional[str] = None,
    build_id: Optional[str] = None,
    vm_name: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for a pipeline run.

    Args:
        job_config: Original job configuration
        results: List of output results from pipeline
        git_commit: Git commit SHA
        build_id: Cloud Build ID
        vm_name: VM instance name
        output_path: Base output path (local or GCS)

    Returns:
        Dictionary containing all metadata
    """
    timestamp = datetime.utcnow().isoformat() + 'Z'

    # Extract job info
    job_name = job_config.get('name', 'unknown')
    variables = job_config.get('variables', [])
    years = job_config.get('years', [])
    seasons = job_config.get('seasons', [])
    aoi_paths = job_config.get('aoi_paths', [])

    # Storage info
    storage = job_config.get('storage', {})
    storage_kind = storage.get('kind', 'unknown')
    bucket = storage.get('bucket', None)
    prefix = storage.get('prefix', None)

    # Build metadata
    metadata = {
        'job_info': {
            'job_name': job_name,
            'timestamp': timestamp,
            'git_commit': git_commit or 'unknown',
            'build_id': build_id or 'unknown',
            'vm_name': vm_name or 'unknown',
        },
        'configuration': {
            'variables': variables,
            'years': years,
            'seasons': seasons,
            'num_aois': len(aoi_paths),
            'aoi_paths': aoi_paths,
            'target_crs': job_config.get('target_crs'),
            'resolution_m': job_config.get('resolution_m'),
        },
        'storage': {
            'kind': storage_kind,
            'bucket': bucket,
            'prefix': prefix,
            'base_path': output_path,
        },
        'execution': {
            'total_outputs': len(results),
            'successful_outputs': len([r for r in results if r.get('gcs_uri') or r.get('local_path')]),
            'failed_outputs': len(results) - len([r for r in results if r.get('gcs_uri') or r.get('local_path')]),
        },
        'outputs': results,
    }

    return metadata


def write_metadata_json(metadata: Dict[str, Any], output_path: str) -> str:
    """Write metadata to JSON file."""
    json_path = Path(output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open('w') as f:
        json.dump(metadata, f, indent=2)

    return str(json_path)


def write_metadata_csv(metadata: Dict[str, Any], output_path: str) -> str:
    """Write flattened metadata to CSV file."""
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    job_info = metadata['job_info']
    config = metadata['configuration']
    storage = metadata['storage']
    execution = metadata['execution']

    # Create one row per output
    rows = []
    for output in metadata['outputs']:
        row = {
            # Job info
            'job_name': job_info['job_name'],
            'timestamp': job_info['timestamp'],
            'git_commit': job_info['git_commit'],
            'build_id': job_info['build_id'],
            'vm_name': job_info['vm_name'],

            # Configuration
            'target_crs': config['target_crs'],
            'resolution_m': config['resolution_m'],
            'num_aois': config['num_aois'],

            # Storage
            'storage_kind': storage['kind'],
            'bucket': storage['bucket'],
            'prefix': storage['prefix'],

            # Output info
            'aoi': output.get('aoi'),
            'aoi_path': output.get('aoi_path'),
            'variable': output.get('variable'),
            'year': output.get('year'),
            'season': output.get('season'),
            'local_path': output.get('local_path'),
            'gcs_uri': output.get('gcs_uri'),
            'filename': output.get('filename'),
        }
        rows.append(row)

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        # Empty results, write header only
        fieldnames = [
            'job_name', 'timestamp', 'git_commit', 'build_id', 'vm_name',
            'target_crs', 'resolution_m', 'num_aois', 'storage_kind',
            'bucket', 'prefix', 'aoi', 'aoi_path', 'variable', 'year',
            'season', 'local_path', 'gcs_uri', 'filename'
        ]
        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    return str(csv_path)


def write_summary_txt(metadata: Dict[str, Any], output_path: str) -> str:
    """Write human-readable summary to text file."""
    txt_path = Path(output_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    job_info = metadata['job_info']
    config = metadata['configuration']
    storage = metadata['storage']
    execution = metadata['execution']

    lines = [
        "=" * 80,
        "SPATIAL DATA MINING PIPELINE - EXECUTION SUMMARY",
        "=" * 80,
        "",
        "JOB INFORMATION",
        "-" * 80,
        f"Job Name:       {job_info['job_name']}",
        f"Timestamp:      {job_info['timestamp']}",
        f"Git Commit:     {job_info['git_commit']}",
        f"Build ID:       {job_info['build_id']}",
        f"VM Instance:    {job_info['vm_name']}",
        "",
        "CONFIGURATION",
        "-" * 80,
        f"Variables:      {', '.join(config['variables'])}",
        f"Years:          {', '.join(map(str, config['years']))}",
        f"Seasons:        {', '.join(config['seasons'])}",
        f"AOIs:           {config['num_aois']} file(s)",
        f"Target CRS:     {config['target_crs']}",
        f"Resolution:     {config['resolution_m']} meters",
        "",
        "STORAGE",
        "-" * 80,
        f"Storage Type:   {storage['kind']}",
        f"Bucket:         {storage['bucket'] or 'N/A'}",
        f"Prefix:         {storage['prefix'] or 'N/A'}",
        f"Base Path:      {storage['base_path'] or 'N/A'}",
        "",
        "EXECUTION RESULTS",
        "-" * 80,
        f"Total Outputs:      {execution['total_outputs']}",
        f"Successful:         {execution['successful_outputs']}",
        f"Failed:             {execution['failed_outputs']}",
        "",
        "OUTPUTS",
        "-" * 80,
    ]

    for i, output in enumerate(metadata['outputs'], 1):
        aoi = output.get('aoi', 'unknown')
        var = output.get('variable', 'unknown')
        year = output.get('year', 'unknown')
        season = output.get('season', 'unknown')
        gcs_uri = output.get('gcs_uri')
        local_path = output.get('local_path')

        lines.append(f"{i}. {aoi} / {var} / {year} / {season}")
        if gcs_uri:
            lines.append(f"   GCS:   {gcs_uri}")
        if local_path:
            lines.append(f"   Local: {local_path}")
        lines.append("")

    lines.extend([
        "=" * 80,
        "END OF SUMMARY",
        "=" * 80,
    ])

    with txt_path.open('w') as f:
        f.write('\n'.join(lines))

    return str(txt_path)


def main():
    """CLI interface for generating metadata files."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate job metadata files for pipeline tracking'
    )
    parser.add_argument('--job-config', required=True, help='Path to job config JSON')
    parser.add_argument('--results', required=True, help='Path to results JSON')
    parser.add_argument('--output-dir', required=True, help='Output directory for metadata files')
    parser.add_argument('--git-commit', help='Git commit SHA')
    parser.add_argument('--build-id', help='Cloud Build ID')
    parser.add_argument('--vm-name', help='VM instance name')

    args = parser.parse_args()

    # Load job config
    with open(args.job_config) as f:
        job_config = json.load(f)

    # Load results
    with open(args.results) as f:
        results = json.load(f)

    # Generate metadata
    metadata = generate_metadata(
        job_config=job_config,
        results=results,
        git_commit=args.git_commit,
        build_id=args.build_id,
        vm_name=args.vm_name,
        output_path=args.output_dir,
    )

    # Write files
    output_dir = Path(args.output_dir)
    job_name = job_config.get('name', 'job')
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    json_file = write_metadata_json(
        metadata,
        str(output_dir / f"{job_name}_{timestamp}_metadata.json")
    )
    csv_file = write_metadata_csv(
        metadata,
        str(output_dir / f"{job_name}_{timestamp}_outputs.csv")
    )
    txt_file = write_summary_txt(
        metadata,
        str(output_dir / f"{job_name}_{timestamp}_summary.txt")
    )

    print(f"Metadata files generated:")
    print(f"  JSON: {json_file}")
    print(f"  CSV:  {csv_file}")
    print(f"  TXT:  {txt_file}")


if __name__ == '__main__':
    main()
