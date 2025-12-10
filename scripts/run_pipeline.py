#!/usr/bin/env python
import click
from spatial_data_mining.orchestrator import run_pipeline


@click.command()
@click.option("--config", "config_path", required=True, help="Path to job config YAML.")
def main(config_path: str):
    run_pipeline(config_path)


if __name__ == "__main__":
    main()
