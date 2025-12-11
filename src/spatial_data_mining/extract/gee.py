import logging
import tempfile
from pathlib import Path
from typing import Tuple

import ee
import requests

SEASON_RANGES = {
    "spring": ("03-01", "05-31"),
    "summer": ("06-01", "08-31"),
    "autumn": ("09-01", "11-30"),
    "fall": ("09-01", "11-30"),
    "winter": ("12-01", "02-28"),
}


def season_date_range(year: int, season: str) -> Tuple[str, str]:
    season_l = season.lower()
    if season_l == "winter":
        return (f"{year}-12-01", f"{year + 1}-02-28")
    if season_l in SEASON_RANGES:
        start, end = SEASON_RANGES[season_l]
        return (f"{year}-{start}", f"{year}-{end}")
    return (f"{year}-01-01", f"{year}-12-31")


class GEEExtractor:
    def __init__(self, index: str):
        self.index = index.upper()
        self.logger = logging.getLogger(__name__)

    def _initialize(self) -> None:
        try:
            ee.Initialize()
        except Exception:
            self.logger.info("Authenticating with Google Earth Engine...")
            ee.Authenticate()
            ee.Initialize()

    def _apply_index(self, image: ee.Image) -> ee.Image:
        if self.index == "NDVI":
            return image.normalizedDifference(["B8", "B4"]).rename("ndvi")
        if self.index == "NDMI":
            return image.normalizedDifference(["B8", "B11"]).rename("ndmi")
        if self.index == "MSI":
            return image.select("B11").divide(image.select("B8")).rename("msi")
        raise ValueError(f"Unsupported index for GEEExtractor: {self.index}")

    def extract(
        self,
        aoi_geojson: dict,
        year: int,
        season: str,
        resolution_m: float,
    ) -> Path:
        """
        Download a seasonal median Sentinel-2 index image clipped to AOI.
        Returns a local temporary GeoTIFF path (EPSG:4326).
        """
        self._initialize()
        region = ee.Geometry(aoi_geojson)
        start_date, end_date = season_date_range(year, season)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        )
        image = collection.median()
        image = self._apply_index(image)

        url = image.getDownloadURL(
            {
                "scale": resolution_m,
                "crs": "EPSG:4326",
                "region": region,
                "fileFormat": "GeoTIFF",
            }
        )

        temp_file = Path(tempfile.gettempdir()) / f"{self.index.lower()}_{year}_{season}.tif"
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        temp_file.write_bytes(resp.content)
        return temp_file
