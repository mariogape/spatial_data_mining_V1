import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import requests
from shapely.geometry import Point, shape
from shapely.prepared import prep

from spatial_data_mining.utils.cancellation import check_cancelled

logger = logging.getLogger(__name__)


class GBIFOccurrenceExtractor:
    """
    Extract GBIF occurrence points within an AOI.
    Returns a GeoJSON file (EPSG:4326) with selected occurrence attributes.
    """

    BASE_URL = "https://api.gbif.org/v1/occurrence/search"
    DEFAULT_PAGE_LIMIT = 300
    OFFSET_LIMIT = 200000
    DEFAULT_KINGDOM_KEYS = (1, 6)
    DEFAULT_ANIMAL_CLASS_KEYS = (212, 359)
    KINGDOM_LABELS = {
        1: "Animalia",
        2: "Archaea",
        3: "Bacteria",
        4: "Chromista",
        5: "Fungi",
        6: "Plantae",
        7: "Protozoa",
        8: "Viruses",
    }
    CLASS_LABELS = {
        212: "Aves",
        359: "Mammalia",
    }

    RECORD_FIELDS = [
        "gbifID",
        "occurrenceID",
        "datasetKey",
        "basisOfRecord",
        "occurrenceStatus",
        "scientificName",
        "species",
        "speciesKey",
        "genus",
        "genusKey",
        "family",
        "familyKey",
        "order",
        "orderKey",
        "class",
        "classKey",
        "phylum",
        "phylumKey",
        "kingdom",
        "kingdomKey",
        "taxonRank",
        "taxonKey",
        "eventDate",
        "year",
        "month",
        "day",
        "country",
        "countryCode",
        "stateProvince",
        "locality",
        "decimalLatitude",
        "decimalLongitude",
        "coordinateUncertaintyInMeters",
        "elevation",
        "depth",
        "recordedBy",
        "identifiedBy",
        "institutionCode",
        "collectionCode",
        "catalogNumber",
        "license",
        "rightsHolder",
        "references",
        "lastInterpreted",
        "issues",
    ]

    def __init__(
        self,
        *,
        taxon_key: int | None = None,
        dataset_key: str | None = None,
        basis_of_record: str | None = None,
        occurrence_status: str | None = None,
        kingdom_keys: list[int] | None = None,
        animal_class_keys: list[int] | None = None,
        max_records: int | None = None,
        page_limit: int | None = None,
        timeout_s: float | None = None,
    ):
        self.taxon_key = int(taxon_key) if taxon_key is not None else None
        self.dataset_key = dataset_key
        self.basis_of_record = basis_of_record.upper() if basis_of_record else None
        self.occurrence_status = occurrence_status.upper() if occurrence_status else None
        self.kingdom_keys = self._normalize_kingdom_keys(kingdom_keys)
        self.animal_class_keys = self._normalize_class_keys(animal_class_keys)
        self.max_records = self._env_int("SDM_GBIF_MAX_RECORDS", max_records)
        self.page_limit = self._clamp_page_limit(page_limit)
        self.timeout_s = self._env_float("SDM_GBIF_TIMEOUT_S", timeout_s, default=60.0)
        self.session = requests.Session()

    @staticmethod
    def _env_int(name: str, fallback: int | None) -> int | None:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return fallback
        try:
            return int(str(raw).strip())
        except ValueError:
            return fallback

    @staticmethod
    def _env_float(name: str, fallback: float | None, default: float) -> float:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            return fallback if fallback is not None else default
        try:
            return float(str(raw).strip())
        except ValueError:
            return fallback if fallback is not None else default

    def _clamp_page_limit(self, page_limit: int | None) -> int:
        limit = page_limit if page_limit is not None else self.DEFAULT_PAGE_LIMIT
        try:
            limit = int(limit)
        except Exception:
            limit = self.DEFAULT_PAGE_LIMIT
        if limit < 1:
            limit = self.DEFAULT_PAGE_LIMIT
        return min(limit, self.DEFAULT_PAGE_LIMIT)

    def _normalize_kingdom_keys(self, keys: list[int] | None) -> list[int]:
        if keys is None:
            return list(self.DEFAULT_KINGDOM_KEYS)
        normalized: list[int] = []
        for key in keys:
            try:
                val = int(key)
            except Exception as exc:
                raise ValueError(f"Invalid GBIF kingdom key: {key!r}") from exc
            if val <= 0:
                raise ValueError(f"Invalid GBIF kingdom key: {key!r}")
            if val not in normalized:
                normalized.append(val)
        return normalized

    def _normalize_class_keys(self, keys: list[int] | None) -> list[int]:
        if keys is None:
            return list(self.DEFAULT_ANIMAL_CLASS_KEYS)
        normalized: list[int] = []
        for key in keys:
            try:
                val = int(key)
            except Exception as exc:
                raise ValueError(f"Invalid GBIF class key: {key!r}") from exc
            if val <= 0:
                raise ValueError(f"Invalid GBIF class key: {key!r}")
            if val not in normalized:
                normalized.append(val)
        return normalized

    def _kingdom_labels(self) -> str:
        labels = [self.KINGDOM_LABELS.get(k, str(k)) for k in self.kingdom_keys]
        return ", ".join(labels) if labels else "all"

    def _kingdom_labels_for(self, keys: list[int]) -> str:
        labels = [self.KINGDOM_LABELS.get(k, str(k)) for k in keys]
        return ", ".join(labels) if labels else "all"

    def _class_labels(self) -> str:
        labels = [self.CLASS_LABELS.get(k, str(k)) for k in self.animal_class_keys]
        return ", ".join(labels) if labels else "all"

    @staticmethod
    def _notify(cb: Optional[Callable[[str], None]], message: str) -> None:
        if cb:
            cb(message)

    @staticmethod
    def _bbox_wkt(bounds: tuple[float, float, float, float]) -> str:
        minx, miny, maxx, maxy = bounds
        return (
            "POLYGON(("
            f"{minx:.6f} {miny:.6f}, "
            f"{minx:.6f} {maxy:.6f}, "
            f"{maxx:.6f} {maxy:.6f}, "
            f"{maxx:.6f} {miny:.6f}, "
            f"{minx:.6f} {miny:.6f}"
            "))"
        )

    @staticmethod
    def _parse_year(value: str | None) -> int | None:
        if not value:
            return None
        for idx in range(len(value) - 3):
            chunk = value[idx : idx + 4]
            if chunk.isdigit():
                return int(chunk)
        return None

    def _request_with_retries(self, params: dict) -> dict:
        last_exc = None
        for attempt in range(1, 4):
            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=self.timeout_s)
                if resp.status_code in {429} or resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"GBIF API error {resp.status_code}", response=resp
                    )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt >= 3:
                    break
                sleep_s = min(2 ** (attempt - 1), 6)
                time.sleep(sleep_s)
        if last_exc:
            raise last_exc
        raise RuntimeError("GBIF request failed without a captured exception.")

    def _normalize_record(self, rec: dict) -> dict:
        row = {key: rec.get(key) for key in self.RECORD_FIELDS}
        if not row.get("gbifID"):
            row["gbifID"] = rec.get("key")
        row["species"] = row.get("species") or row.get("scientificName")
        row["year"] = row.get("year") or self._parse_year(row.get("eventDate"))
        issues = row.get("issues")
        if isinstance(issues, (list, tuple)):
            row["issues"] = ";".join(str(i) for i in issues if i is not None)
        return row

    def extract(
        self,
        aoi_geojson: dict,
        year: int | str,
        season: str,
        resolution_m: float | None,
        temp_dir: str | Path | None = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Path:
        check_cancelled(should_stop)
        geom = shape(aoi_geojson)
        if not geom.is_valid:
            try:
                geom = geom.buffer(0)
            except Exception:
                pass
        if geom.is_empty:
            raise ValueError("AOI geometry is empty after normalization.")

        geom_prep = prep(geom)
        bbox_wkt = self._bbox_wkt(geom.bounds)

        base_params: dict[str, str | int] = {
            "limit": self.page_limit,
            "offset": 0,
            "hasCoordinate": "true",
            "geometry": bbox_wkt,
        }
        if self.taxon_key:
            base_params["taxonKey"] = self.taxon_key
        if self.dataset_key:
            base_params["datasetKey"] = self.dataset_key
        if self.basis_of_record:
            base_params["basisOfRecord"] = self.basis_of_record
        if self.occurrence_status:
            base_params["occurrenceStatus"] = self.occurrence_status

        param_sets: list[tuple[str, dict[str, str | int]]] = []
        kingdom_keys = list(self.kingdom_keys) if self.kingdom_keys else []
        if kingdom_keys and 1 in kingdom_keys and self.animal_class_keys:
            animal_params = dict(base_params)
            animal_params["kingdomKey"] = [1]
            animal_params["classKey"] = list(self.animal_class_keys)
            param_sets.append((f"Animalia ({self._class_labels()})", animal_params))

            other_kingdoms = [k for k in kingdom_keys if k != 1]
            if other_kingdoms:
                other_params = dict(base_params)
                other_params["kingdomKey"] = other_kingdoms
                param_sets.append(
                    (f"Other kingdoms ({self._kingdom_labels_for(other_kingdoms)})", other_params)
                )
        else:
            params = dict(base_params)
            if kingdom_keys:
                params["kingdomKey"] = kingdom_keys
            param_sets.append((f"Kingdoms ({self._kingdom_labels()})", params))

        results: list[dict] = []
        geometries: list[Point] = []
        fetched_total = 0

        def _fetch_param_set(label: str, params: dict[str, str | int]) -> bool:
            nonlocal fetched_total
            total = None
            params = dict(params)
            params["offset"] = 0
            while True:
                check_cancelled(should_stop)
                data = self._request_with_retries(params)
                page_results = data.get("results") or []
                if total is None:
                    total = data.get("count")
                    if (
                        total is not None
                        and self.max_records is None
                        and int(total) > self.OFFSET_LIMIT
                    ):
                        raise RuntimeError(
                            "GBIF result set exceeds API paging limit; set gbif_max_records "
                            "or use the GBIF download service for large requests."
                        )
                if not page_results:
                    break

                for rec in page_results:
                    if self.max_records is not None and fetched_total >= self.max_records:
                        self._notify(
                            progress_cb,
                            f"GBIF: stopping at max_records={self.max_records}",
                        )
                        return False
                    rec_key = None
                    if self.kingdom_keys:
                        rec_key = rec.get("kingdomKey")
                        try:
                            rec_key = int(rec_key)
                        except Exception:
                            continue
                        if rec_key not in self.kingdom_keys:
                            continue
                    if self.animal_class_keys and rec_key == 1:
                        rec_class = rec.get("classKey")
                        try:
                            rec_class = int(rec_class)
                        except Exception:
                            continue
                        if rec_class not in self.animal_class_keys:
                            continue
                    lat = rec.get("decimalLatitude")
                    lon = rec.get("decimalLongitude")
                    if lat is None or lon is None:
                        continue
                    try:
                        point = Point(float(lon), float(lat))
                    except Exception:
                        continue
                    if not geom_prep.intersects(point):
                        continue
                    row = self._normalize_record(rec)
                    results.append(row)
                    geometries.append(point)
                    fetched_total += 1

                if total:
                    self._notify(
                        progress_cb,
                        f"GBIF ({label}): fetched {min(params['offset'] + len(page_results), total)} / {total} records (offset {params['offset']})",
                    )
                else:
                    self._notify(
                        progress_cb,
                        f"GBIF ({label}): fetched {params['offset'] + len(page_results)} records (offset {params['offset']})",
                    )

                if data.get("endOfRecords"):
                    break

                next_offset = params["offset"] + self.page_limit
                if next_offset >= self.OFFSET_LIMIT:
                    raise RuntimeError(
                        "GBIF paging limit reached; set gbif_max_records "
                        "or use the GBIF download service for large requests."
                    )
                params["offset"] = next_offset
            return True

        for label, params in param_sets:
            self._notify(progress_cb, f"GBIF: querying {label}")
            if not _fetch_param_set(label, params):
                break

        tmp_dir = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_dir / "gbif_occurrences_raw.geojson"

        try:
            import geopandas as gpd
        except Exception as exc:
            raise RuntimeError("geopandas is required to write GBIF GeoJSON output.") from exc

        if results:
            gdf = gpd.GeoDataFrame(results, geometry=geometries, crs="EPSG:4326")
        else:
            gdf = gpd.GeoDataFrame(results, geometry=gpd.GeoSeries([], crs="EPSG:4326"))

        gdf.to_file(out_path, driver="GeoJSON")
        return out_path
