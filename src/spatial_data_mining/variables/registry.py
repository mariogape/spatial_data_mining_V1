from spatial_data_mining.extract.gee import GEEExtractor
from spatial_data_mining.transform.raster_ops import process_raster_to_target

VARIABLES = {
    "ndvi": {"extractor": GEEExtractor("NDVI"), "transform": process_raster_to_target},
    "ndmi": {"extractor": GEEExtractor("NDMI"), "transform": process_raster_to_target},
    "msi": {"extractor": GEEExtractor("MSI"), "transform": process_raster_to_target},
}


def get_variable(name: str):
    key = name.lower()
    if key not in VARIABLES:
        raise KeyError(f"Variable not registered: {name}")
    return VARIABLES[key]
