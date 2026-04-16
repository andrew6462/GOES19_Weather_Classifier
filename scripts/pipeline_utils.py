from pathlib import Path
import math

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CMI_DIR = PROJECT_ROOT / "raw" / "goes_cmi"
RAW_ACM_DIR = PROJECT_ROOT / "raw" / "goes_acm"
RAW_GLM_DIR = PROJECT_ROOT / "raw" / "goes_glm"
RAW_IBTRACS_DIR = PROJECT_ROOT / "raw" / "ibtracs"
INDEX_DIR = PROJECT_ROOT / "index"
PROCESSED_DIR = PROJECT_ROOT / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
RAW_IMAGE_DIR = PROJECT_ROOT / "raw" / "images"


def make_dirs():
    for folder in [INDEX_DIR, PROCESSED_DIR, REPORTS_DIR, RAW_IMAGE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def open_dataset(path):
    return xr.open_dataset(path, engine="netcdf4")


def rel_path(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def to_time(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def overlap_seconds(start1, end1, start2, end2):
    if pd.isna(start1) or pd.isna(end1) or pd.isna(start2) or pd.isna(end2):
        return 0.0
    start = max(start1, start2)
    end = min(end1, end2)
    if end <= start:
        return 0.0
    return float((end - start).total_seconds())


def decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="ignore").strip()
    return value


def get_product_name(ds):
    name = ds.attrs.get("dataset_name")
    if not name:
        return None
    parts = str(name).split("_")
    if len(parts) > 1:
        return parts[1]
    return str(name)


def get_scene_center(ds):
    extent = ds.get("geospatial_lat_lon_extent")
    if extent is None:
        return None, None

    lat = extent.attrs.get("geospatial_center_latitude")
    lon = extent.attrs.get("geospatial_center_longitude")

    if lat is None or lon is None:
        return None, None

    return float(lat), float(lon)


def distance_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def make_cloud_binary(acm):
    out = np.full(acm.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(acm)
    out[np.isin(acm, [0, 1]) & valid] = 0.0
    out[np.isin(acm, [2, 3]) & valid] = 1.0
    return out
