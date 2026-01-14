import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional
import threading

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
    MAX_TILE_DEPTH = 6
    DEFAULT_TILE_TARGET = 5000
    DEFAULT_TILE_WORKERS = 1
    DEFAULT_MAX_ATTEMPTS = 10
    DEFAULT_MIN_REQUEST_INTERVAL_S = 0.2
    COMMERCIAL_LICENSE_FLAG = "COMMERCIAL"
    DEFAULT_ALLOWED_LICENSES = (COMMERCIAL_LICENSE_FLAG,)
    COMMERCIAL_LICENSES = {"CC0", "CC_BY", "CC_BY_SA", "CC_BY_ND"}
    NONCOMMERCIAL_LICENSES = {"CC_BY_NC", "CC_BY_NC_SA", "CC_BY_NC_ND"}
    LICENSE_ALIASES = {
        "CC0": "CC0",
        "CC0_1_0": "CC0",
        "PUBLIC_DOMAIN": "CC0",
        "PUBLIC_DOMAIN_MARK": "CC0",
        "PUBLIC_DOMAIN_DEDICATION": "CC0",
        "CC_BY": "CC_BY",
        "CC_BY_4_0": "CC_BY",
        "CC_BY_3_0": "CC_BY",
        "CC_BY_2_0": "CC_BY",
        "CC_BY_SA": "CC_BY_SA",
        "CC_BY_SA_4_0": "CC_BY_SA",
        "CC_BY_SA_3_0": "CC_BY_SA",
        "CC_BY_ND": "CC_BY_ND",
        "CC_BY_ND_4_0": "CC_BY_ND",
        "CC_BY_NC": "CC_BY_NC",
        "CC_BY_NC_4_0": "CC_BY_NC",
        "CC_BY_NC_SA": "CC_BY_NC_SA",
        "CC_BY_NC_SA_4_0": "CC_BY_NC_SA",
        "CC_BY_NC_ND": "CC_BY_NC_ND",
        "CC_BY_NC_ND_4_0": "CC_BY_NC_ND",
        "COMMERCIAL": "COMMERCIAL",
        "COMMERCIAL_USE": "COMMERCIAL",
        "ALL_COMMERCIAL": "COMMERCIAL",
    }
    LICENSE_URL_MAP = {
        "creativecommons.org/publicdomain/zero": "CC0",
        "creativecommons.org/licenses/by-nc-nd": "CC_BY_NC_ND",
        "creativecommons.org/licenses/by-nc-sa": "CC_BY_NC_SA",
        "creativecommons.org/licenses/by-nc": "CC_BY_NC",
        "creativecommons.org/licenses/by-nd": "CC_BY_ND",
        "creativecommons.org/licenses/by-sa": "CC_BY_SA",
        "creativecommons.org/licenses/by": "CC_BY",
    }
    API_LICENSE_MAP = {
        "CC0": "CC0_1_0",
        "CC_BY": "CC_BY_4_0",
        "CC_BY_NC": "CC_BY_NC_4_0",
    }
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
        allowed_licenses: list[str] | None = None,
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
        self.allowed_licenses = self._normalize_licenses(allowed_licenses)
        self.max_records = self._env_int("SDM_GBIF_MAX_RECORDS", max_records)
        self.page_limit = self._clamp_page_limit(page_limit)
        self.timeout_s = self._env_float("SDM_GBIF_TIMEOUT_S", timeout_s, default=60.0)
        self.tile_target = self._positive_int(
            self._env_int("SDM_GBIF_TILE_TARGET", self.DEFAULT_TILE_TARGET),
            self.DEFAULT_TILE_TARGET,
        )
        self.tile_workers = self._positive_int(
            self._env_int("SDM_GBIF_TILE_WORKERS", self.DEFAULT_TILE_WORKERS),
            self.DEFAULT_TILE_WORKERS,
        )
        self.max_attempts = self._positive_int(
            self._env_int("SDM_GBIF_MAX_ATTEMPTS", self.DEFAULT_MAX_ATTEMPTS),
            self.DEFAULT_MAX_ATTEMPTS,
        )
        self.min_request_interval_s = self._env_float(
            "SDM_GBIF_MIN_REQUEST_INTERVAL_S",
            self.DEFAULT_MIN_REQUEST_INTERVAL_S,
            default=self.DEFAULT_MIN_REQUEST_INTERVAL_S,
        )
        self._throttle_s = max(0.0, float(self.min_request_interval_s))
        self._rate_lock = threading.Lock()
        self._next_request_ts = 0.0
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

    @staticmethod
    def _positive_int(value: int | None, default: int) -> int:
        if value is None:
            return default
        try:
            value_int = int(value)
        except Exception:
            return default
        return value_int if value_int > 0 else default

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

    @staticmethod
    def _tokenize_license(value: str) -> list[str]:
        return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]

    def _canonical_license(self, value: str | None) -> str | None:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        key = re.sub(r"[^A-Za-z0-9]+", "_", raw.upper()).strip("_")
        mapped = self.LICENSE_ALIASES.get(key)
        if mapped:
            return mapped
        lower = raw.lower()
        for needle, mapped in self.LICENSE_URL_MAP.items():
            if needle in lower:
                return mapped
        tokens = self._tokenize_license(lower)
        if "cc0" in tokens or "publicdomain" in tokens or ("public" in tokens and "domain" in tokens):
            return "CC0"
        is_cc = (
            "creativecommons" in tokens
            or ("creative" in tokens and "commons" in tokens)
            or "cc" in tokens
        )
        if not is_cc:
            return None
        has_by = "by" in tokens or "attribution" in tokens
        has_nc = "nc" in tokens or "noncommercial" in tokens
        has_sa = "sa" in tokens or "sharealike" in tokens
        has_nd = "nd" in tokens or "noderivatives" in tokens or ("no" in tokens and "derivatives" in tokens)
        if has_nc:
            if has_sa:
                return "CC_BY_NC_SA"
            if has_nd:
                return "CC_BY_NC_ND"
            return "CC_BY_NC"
        if has_sa:
            return "CC_BY_SA"
        if has_nd:
            return "CC_BY_ND"
        if has_by:
            return "CC_BY"
        return None

    def _normalize_licenses(self, licenses: list[str] | None) -> list[str]:
        if licenses is None:
            return list(self.DEFAULT_ALLOWED_LICENSES)
        normalized: list[str] = []
        for lic in licenses:
            canon = self._canonical_license(lic)
            if canon:
                normalized.append(canon)
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for lic in normalized:
            if lic not in seen:
                deduped.append(lic)
                seen.add(lic)
        return deduped

    def _kingdom_labels(self) -> str:
        labels = [self.KINGDOM_LABELS.get(k, str(k)) for k in self.kingdom_keys]
        return ", ".join(labels) if labels else "all"

    def _kingdom_labels_for(self, keys: list[int]) -> str:
        labels = [self.KINGDOM_LABELS.get(k, str(k)) for k in keys]
        return ", ".join(labels) if labels else "all"

    def _class_labels(self) -> str:
        labels = [self.CLASS_LABELS.get(k, str(k)) for k in self.animal_class_keys]
        return ", ".join(labels) if labels else "all"

    def _license_labels(self) -> str:
        if not self.allowed_licenses:
            return "all"
        if self.COMMERCIAL_LICENSE_FLAG in self.allowed_licenses:
            return "commercial-use (CC0/CC BY variants)"
        return ", ".join(self.allowed_licenses)

    def _api_license_filters(self) -> list[str] | None:
        if not self.allowed_licenses:
            return None
        if self.COMMERCIAL_LICENSE_FLAG in self.allowed_licenses:
            return None
        filters: list[str] = []
        for lic in self.allowed_licenses:
            mapped = self.API_LICENSE_MAP.get(lic)
            if not mapped:
                return None
            filters.append(mapped)
        return filters or None

    def _license_allowed(self, license_value: str | None) -> bool:
        if not self.allowed_licenses:
            return True
        canon = self._canonical_license(license_value)
        if not canon:
            return False
        if self.COMMERCIAL_LICENSE_FLAG in self.allowed_licenses:
            return canon in self.COMMERCIAL_LICENSES or canon in self.allowed_licenses
        return canon in self.allowed_licenses

    def _estimate_count(self, params: dict[str, str | int], bbox_wkt: str) -> int:
        count_params = dict(params)
        count_params["geometry"] = bbox_wkt
        count_params["limit"] = 1
        count_params["offset"] = 0
        data = self._request_with_retries(count_params)
        try:
            return int(data.get("count") or 0)
        except Exception:
            return 0

    def _split_bounds(
        self, bounds: tuple[float, float, float, float]
    ) -> list[tuple[float, float, float, float]]:
        minx, miny, maxx, maxy = bounds
        midx = (minx + maxx) / 2.0
        midy = (miny + maxy) / 2.0
        return [
            (minx, miny, midx, midy),
            (midx, miny, maxx, midy),
            (minx, midy, midx, maxy),
            (midx, midy, maxx, maxy),
        ]

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

    def _request_with_retries(
        self, params: dict, session: requests.Session | None = None
    ) -> dict:
        last_exc = None
        client = session or self.session
        for attempt in range(1, self.max_attempts + 1):
            try:
                self._throttle_request()
                resp = client.get(self.BASE_URL, params=params, timeout=self.timeout_s)
                if resp.status_code == 429:
                    retry_after = self._parse_retry_after(resp)
                    wait_s = retry_after if retry_after is not None else self._backoff_seconds(attempt)
                    self._bump_throttle(wait_s, attempt)
                    if attempt >= self.max_attempts:
                        raise requests.HTTPError(
                            f"GBIF API error {resp.status_code}", response=resp
                        )
                    time.sleep(wait_s)
                    continue
                if resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"GBIF API error {resp.status_code}", response=resp
                    )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_attempts:
                    break
                sleep_s = self._backoff_seconds(attempt)
                time.sleep(sleep_s)
        if last_exc:
            raise last_exc
        raise RuntimeError("GBIF request failed without a captured exception.")

    def _throttle_request(self) -> None:
        if self._throttle_s <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            if now < self._next_request_ts:
                wait_s = self._next_request_ts - now
                self._next_request_ts += self._throttle_s
            else:
                wait_s = 0.0
                self._next_request_ts = now + self._throttle_s
        if wait_s > 0:
            time.sleep(wait_s)

    def _backoff_seconds(self, attempt: int) -> float:
        base = min(2 ** (attempt - 1), 30)
        jitter = min(0.5, 0.1 * attempt)
        return base + (jitter * (attempt % 3))

    @staticmethod
    def _parse_retry_after(resp: requests.Response) -> float | None:
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    def _bump_throttle(self, wait_s: float, attempt: int) -> None:
        target = min(max(wait_s / 2.0, 0.2 + 0.1 * attempt), 2.0)
        self._throttle_s = max(self._throttle_s, target)

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
        if self.allowed_licenses:
            api_filters = self._api_license_filters()
            if api_filters:
                base_params["license"] = api_filters
            self._notify(
                progress_cb,
                f"GBIF: filtering to licenses {self._license_labels()}",
            )

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
        seen_ids: set[str] = set()

        def _fetch_tile(
            label: str,
            params: dict[str, str | int],
            bounds: tuple[float, float, float, float],
            tile_index: int,
            tile_total: int,
            stop_event: threading.Event | None,
        ) -> tuple[list[dict], list[Point]]:
            local_results: list[dict] = []
            local_geometries: list[Point] = []
            local_seen: set[str] = set()
            tile_label = label if tile_total <= 1 else f"{label} tile {tile_index}/{tile_total}"
            session = requests.Session()
            local_geom_prep = prep(geom)

            tile_params = dict(params)
            tile_params["geometry"] = self._bbox_wkt(bounds)
            tile_params["offset"] = 0
            total = None

            while True:
                check_cancelled(should_stop)
                if stop_event and stop_event.is_set():
                    break
                data = self._request_with_retries(tile_params, session=session)
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
                    if stop_event and stop_event.is_set():
                        break
                    rec_key = None
                    if self.kingdom_keys:
                        rec_key = rec.get("kingdomKey")
                        try:
                            rec_key = int(rec_key)
                        except Exception:
                            continue
                        if rec_key not in self.kingdom_keys:
                            continue
                    if self.allowed_licenses and not self._license_allowed(rec.get("license")):
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
                    if not local_geom_prep.intersects(point):
                        continue
                    row = self._normalize_record(rec)
                    record_id = row.get("gbifID") or row.get("occurrenceID")
                    if record_id is not None:
                        record_id = str(record_id)
                        if record_id in local_seen:
                            continue
                        local_seen.add(record_id)
                    local_results.append(row)
                    local_geometries.append(point)

                if total:
                    self._notify(
                        progress_cb,
                        f"GBIF ({tile_label}): fetched {min(tile_params['offset'] + len(page_results), total)} / {total} records (offset {tile_params['offset']})",
                    )
                else:
                    self._notify(
                        progress_cb,
                        f"GBIF ({tile_label}): fetched {tile_params['offset'] + len(page_results)} records (offset {tile_params['offset']})",
                    )

                if data.get("endOfRecords"):
                    break

                next_offset = tile_params["offset"] + self.page_limit
                if next_offset >= self.OFFSET_LIMIT:
                    raise RuntimeError(
                        "GBIF paging limit reached; set gbif_max_records "
                        "or use the GBIF download service for large requests."
                    )
                tile_params["offset"] = next_offset

            return local_results, local_geometries

        def _merge_tile_results(tile_rows: list[dict], tile_geoms: list[Point]) -> bool:
            nonlocal fetched_total
            for row, geom_row in zip(tile_rows, tile_geoms):
                record_id = row.get("gbifID") or row.get("occurrenceID")
                if record_id is not None:
                    record_id = str(record_id)
                    if record_id in seen_ids:
                        continue
                    seen_ids.add(record_id)
                results.append(row)
                geometries.append(geom_row)
                fetched_total += 1
                if self.max_records is not None and fetched_total >= self.max_records:
                    return False
            return True

        def _build_tiles(
            params: dict[str, str | int],
            bounds: tuple[float, float, float, float],
            depth: int = 0,
        ) -> list[tuple[float, float, float, float]]:
            count = self._estimate_count(params, self._bbox_wkt(bounds))
            if count == 0:
                return []
            target = max(1, self.tile_target)
            allow_tiling = self.max_records is None or self.max_records > target
            if count <= target or not allow_tiling:
                return [bounds]
            if depth >= self.MAX_TILE_DEPTH:
                if count > self.OFFSET_LIMIT:
                    raise RuntimeError(
                        "GBIF result set exceeds API paging limit even after tiling; "
                        "set gbif_max_records, reduce AOI, or use the GBIF download service."
                    )
                return [bounds]
            tiles: list[tuple[float, float, float, float]] = []
            for sub_bounds in self._split_bounds(bounds):
                tiles.extend(_build_tiles(params, sub_bounds, depth + 1))
            return tiles

        for label, params in param_sets:
            if self.max_records is not None and fetched_total >= self.max_records:
                break
            self._notify(progress_cb, f"GBIF: querying {label}")
            tiles = _build_tiles(params, geom.bounds)
            if not tiles:
                continue
            if len(tiles) > 1:
                self._notify(
                    progress_cb,
                    f"GBIF: tiling AOI into {len(tiles)} parts (target <= {self.tile_target} records per tile).",
                )
            if len(tiles) == 1 or self.tile_workers <= 1:
                for idx, bounds in enumerate(tiles, start=1):
                    tile_rows, tile_geoms = _fetch_tile(
                        label, params, bounds, idx, len(tiles), None
                    )
                    if not _merge_tile_results(tile_rows, tile_geoms):
                        break
                continue

            stop_event = threading.Event()
            max_workers = min(max(1, self.tile_workers), len(tiles))
            self._notify(progress_cb, f"GBIF: fetching tiles with up to {max_workers} workers.")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _fetch_tile,
                        label,
                        params,
                        bounds,
                        idx + 1,
                        len(tiles),
                        stop_event,
                    ): idx
                    for idx, bounds in enumerate(tiles)
                }
                for future in as_completed(futures):
                    tile_rows, tile_geoms = future.result()
                    if not _merge_tile_results(tile_rows, tile_geoms):
                        stop_event.set()
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
