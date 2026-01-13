from functools import partial

from spatial_data_mining.extract.alpha_earth import AlphaEarthExtractor
from spatial_data_mining.extract.clcplus import CLCPlusExtractor
from spatial_data_mining.extract.gbif_occurrences import GBIFOccurrenceExtractor
from spatial_data_mining.extract.openeo_indices import OpenEOIndexExtractor, OpenEOFVCExtractor
from spatial_data_mining.extract.openeo_rgb import OpenEORGBExtractor
from spatial_data_mining.extract.openeo_swi import OpenEOSoilWaterIndexExtractor
from spatial_data_mining.extract.soilgrids import SoilGridsExtractor
from spatial_data_mining.transform.raster_ops import (
    process_clcplus_to_target,
    process_fvc_to_target,
    process_rgb_true_color,
    process_raster_to_target,
)
from spatial_data_mining.transform.vector_ops import (
    normalize_vector_format,
    process_vector_to_target,
)

VARIABLES = {
    # Sentinel-2 indices are fetched via Copernicus Data Space openEO (faster/more scalable than GEE downloads).
    "ndvi": {"extractor_factory": lambda _job=None: OpenEOIndexExtractor("NDVI"), "transform": process_raster_to_target},
    "ndmi": {"extractor_factory": lambda _job=None: OpenEOIndexExtractor("NDMI"), "transform": process_raster_to_target},
    "msi": {"extractor_factory": lambda _job=None: OpenEOIndexExtractor("MSI"), "transform": process_raster_to_target},
    "bsi": {"extractor_factory": lambda _job=None: OpenEOIndexExtractor("BSI"), "transform": process_raster_to_target},
    "fvc": {"extractor_factory": lambda _job=None: OpenEOFVCExtractor(), "transform": process_fvc_to_target},
    "swi": {
        "extractor_factory": lambda job=None: OpenEOSoilWaterIndexExtractor(
            collection_id=getattr(job, "swi_collection_id", None),
            band=getattr(job, "swi_band", None),
            temporal_agg=getattr(job, "swi_aggregation", None),
            swi_date=getattr(job, "swi_date", None),
            oidc_provider_id=getattr(job, "swi_oidc_provider_id", None),
            backend_url=getattr(job, "swi_backend_url", None),
        ),
        "transform": process_raster_to_target,
    },
    "rgb": {
        "extractor_factory": lambda job=None: OpenEORGBExtractor(
            rgb_date=getattr(job, "rgb_date", None),
            search_days=getattr(job, "rgb_search_days", None),
            collection_id="SENTINEL2_L1C",
            bands="B04,B03,B02",
            cloud_cover_max=getattr(job, "rgb_cloud_cover_max", None),
            cloud_cover_property=getattr(job, "rgb_cloud_cover_property", None),
            cloud_mask=False,
            oidc_provider_id=getattr(job, "rgb_oidc_provider_id", None),
            stac_url=getattr(job, "rgb_stac_url", None),
            stac_collection_id="sentinel-2-l1c",
            prefilter=getattr(job, "rgb_prefilter", None),
            backend_url=getattr(job, "rgb_backend_url", None),
            variable_name="rgb",
        ),
        "transform": process_rgb_true_color,
    },
    "rgb_raw": {
        "extractor_factory": lambda job=None: OpenEORGBExtractor(
            rgb_date=getattr(job, "rgb_date", None),
            search_days=getattr(job, "rgb_search_days", None),
            collection_id=getattr(job, "rgb_collection_id", None),
            bands=getattr(job, "rgb_bands", None),
            cloud_cover_max=getattr(job, "rgb_cloud_cover_max", None),
            cloud_cover_property=getattr(job, "rgb_cloud_cover_property", None),
            oidc_provider_id=getattr(job, "rgb_oidc_provider_id", None),
            stac_url=getattr(job, "rgb_stac_url", None),
            stac_collection_id=getattr(job, "rgb_stac_collection_id", None),
            prefilter=getattr(job, "rgb_prefilter", None),
            backend_url=getattr(job, "rgb_backend_url", None),
            variable_name="rgb_raw",
        ),
        "transform": process_raster_to_target,
    },
    "alpha_earth": {
        "extractor_factory": lambda _job=None: AlphaEarthExtractor(),
        "transform": process_raster_to_target,
    },
    "clcplus": {
        "extractor_factory": lambda job=None: CLCPlusExtractor(
            getattr(job, "clcplus_input_dir", None)
        ),
        "transform": process_clcplus_to_target,
    },
    "bdod": {
        "extractor_factory": lambda job=None: SoilGridsExtractor(
            "bdod",
            depth=getattr(job, "soilgrids_depth", None),
            stat=getattr(job, "soilgrids_stat", None),
            base_url=getattr(job, "soilgrids_base_url", None),
            tile_index_path=getattr(job, "soilgrids_tile_index_path", None),
        ),
        "transform": process_raster_to_target,
    },
    "clay": {
        "extractor_factory": lambda job=None: SoilGridsExtractor(
            "clay",
            depth=getattr(job, "soilgrids_depth", None),
            stat=getattr(job, "soilgrids_stat", None),
            base_url=getattr(job, "soilgrids_base_url", None),
            tile_index_path=getattr(job, "soilgrids_tile_index_path", None),
        ),
        "transform": process_raster_to_target,
    },
    "cfvo": {
        "extractor_factory": lambda job=None: SoilGridsExtractor(
            "cfvo",
            depth=getattr(job, "soilgrids_depth", None),
            stat=getattr(job, "soilgrids_stat", None),
            base_url=getattr(job, "soilgrids_base_url", None),
            tile_index_path=getattr(job, "soilgrids_tile_index_path", None),
        ),
        "transform": process_raster_to_target,
    },
    "sand": {
        "extractor_factory": lambda job=None: SoilGridsExtractor(
            "sand",
            depth=getattr(job, "soilgrids_depth", None),
            stat=getattr(job, "soilgrids_stat", None),
            base_url=getattr(job, "soilgrids_base_url", None),
            tile_index_path=getattr(job, "soilgrids_tile_index_path", None),
        ),
        "transform": process_raster_to_target,
    },
    "silt": {
        "extractor_factory": lambda job=None: SoilGridsExtractor(
            "silt",
            depth=getattr(job, "soilgrids_depth", None),
            stat=getattr(job, "soilgrids_stat", None),
            base_url=getattr(job, "soilgrids_base_url", None),
            tile_index_path=getattr(job, "soilgrids_tile_index_path", None),
        ),
        "transform": process_raster_to_target,
    },
    "cec": {
        "extractor_factory": lambda job=None: SoilGridsExtractor(
            "cec",
            depth=getattr(job, "soilgrids_depth", None),
            stat=getattr(job, "soilgrids_stat", None),
            base_url=getattr(job, "soilgrids_base_url", None),
            tile_index_path=getattr(job, "soilgrids_tile_index_path", None),
        ),
        "transform": process_raster_to_target,
    },
    "gbif_occurrences": {
        "extractor_factory": lambda job=None: GBIFOccurrenceExtractor(
            taxon_key=getattr(job, "gbif_taxon_key", None),
            dataset_key=getattr(job, "gbif_dataset_key", None),
            basis_of_record=getattr(job, "gbif_basis_of_record", None),
            occurrence_status=getattr(job, "gbif_occurrence_status", None),
            kingdom_keys=getattr(job, "gbif_kingdom_keys", None),
            animal_class_keys=getattr(job, "gbif_animal_class_keys", None),
            max_records=getattr(job, "gbif_max_records", None),
        ),
        "transform_factory": lambda job=None: partial(
            process_vector_to_target,
            output_format=getattr(job, "gbif_format", None),
        ),
        "output_type": "vector",
        "output_ext": lambda job=None: normalize_vector_format(getattr(job, "gbif_format", None)),
        "include_time": False,
    },
}


def _resolve_extractor(var_def: dict, job_cfg=None):
    if "extractor_factory" in var_def:
        return var_def["extractor_factory"](job_cfg)
    return var_def.get("extractor")


def _resolve_transform(var_def: dict, job_cfg=None):
    if "transform_factory" in var_def:
        return var_def["transform_factory"](job_cfg)
    return var_def.get("transform")


def _resolve_output_ext(var_def: dict, job_cfg=None):
    output_ext = var_def.get("output_ext")
    if callable(output_ext):
        return output_ext(job_cfg)
    return output_ext


def get_variable(name: str, job_cfg=None):
    key = name.lower()
    if key not in VARIABLES:
        raise KeyError(f"Variable not registered: {name}")
    var_def = VARIABLES[key]
    extractor = _resolve_extractor(var_def, job_cfg=job_cfg)
    transform = _resolve_transform(var_def, job_cfg=job_cfg)
    output_type = var_def.get("output_type", "raster")
    output_ext = _resolve_output_ext(var_def, job_cfg=job_cfg)
    if not output_ext:
        output_ext = "tif" if output_type == "raster" else "geojson"
    include_time = var_def.get("include_time", True)
    return {
        "extractor": extractor,
        "transform": transform,
        "output_type": output_type,
        "output_ext": output_ext,
        "include_time": include_time,
    }
