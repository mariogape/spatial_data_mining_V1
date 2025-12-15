import argparse
import sys
from pathlib import Path

# Ensure the repo's src/ is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from spatial_data_mining.extract.alpha_earth import AlphaEarthExtractor


def _tile_index(path: Path) -> int:
    """
    Extract the numeric tile index from filenames like alphaearth_2023_tile5.tif.
    Falls back to 0 if no index is found so sorting is stable.
    """
    stem = path.stem
    if "tile" in stem:
        try:
            return int(stem.split("tile")[-1])
        except ValueError:
            return 0
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Merge previously downloaded AlphaEarth tiles without re-downloading."
    )
    parser.add_argument(
        "--tile-dir",
        required=True,
        help="Directory containing alphaearth_*_tile*.tif files (e.g. D:\\OpenPas Spatial Data\\Alpha Earth)",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year used in the tile filenames (e.g. 2023).",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base name for the merged output (default: alphaearth_<year>).",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional glob pattern for tiles; defaults to alphaearth_<year>_tile*.tif",
    )

    args = parser.parse_args()

    tile_dir = Path(args.tile_dir).expanduser()
    if not tile_dir.exists():
        raise SystemExit(f"Tile directory does not exist: {tile_dir}")

    pattern = args.pattern or f"alphaearth_{args.year}_tile*.tif"
    tiles = sorted(tile_dir.glob(pattern), key=_tile_index)
    if not tiles:
        raise SystemExit(f"No tiles found in {tile_dir} with pattern {pattern}")

    output_name = args.output_name or f"alphaearth_{args.year}"

    extractor = AlphaEarthExtractor()
    merged = extractor._merge_tiles(tiles, output_name, tile_dir)
    print(f"Merged {len(tiles)} tiles -> {merged}")


if __name__ == "__main__":
    main()
