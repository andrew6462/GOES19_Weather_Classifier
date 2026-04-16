import sys

import numpy as np
import pandas as pd

from pipeline_utils import (
    INDEX_DIR,
    RAW_ACM_DIR,
    RAW_CMI_DIR,
    RAW_GLM_DIR,
    RAW_IBTRACS_DIR,
    decode_text,
    get_product_name,
    get_scene_center,
    make_dirs,
    open_dataset,
    rel_path,
    to_time,
)


def index_cmi_file(path):
    ds = open_dataset(path)
    center_lat, center_lon = get_scene_center(ds)
    row = {
        "file_path": rel_path(path),
        "file_name": path.name,
        "source_type": "CMI",
        "product_name": get_product_name(ds),
        "dataset_name": ds.attrs.get("dataset_name"),
        "title": ds.attrs.get("title"),
        "platform_id": ds.attrs.get("platform_ID"),
        "scene_id": ds.attrs.get("scene_id"),
        "start_time": to_time(ds.attrs.get("time_coverage_start")),
        "end_time": to_time(ds.attrs.get("time_coverage_end")),
        "x_size": int(ds.sizes.get("x")) if "x" in ds.sizes else None,
        "y_size": int(ds.sizes.get("y")) if "y" in ds.sizes else None,
        "band_id": decode_text(ds.coords["band_id"].values[0]) if "band_id" in ds.coords else None,
        "band_wavelength": float(ds.coords["band_wavelength"].values[0]) if "band_wavelength" in ds.coords else None,
        "scene_center_lat": center_lat,
        "scene_center_lon": center_lon,
    }
    ds.close()
    return row


def index_acm_file(path):
    ds = open_dataset(path)
    center_lat, center_lon = get_scene_center(ds)
    row = {
        "file_path": rel_path(path),
        "file_name": path.name,
        "source_type": "ACM",
        "product_name": get_product_name(ds),
        "dataset_name": ds.attrs.get("dataset_name"),
        "title": ds.attrs.get("title"),
        "platform_id": ds.attrs.get("platform_ID"),
        "scene_id": ds.attrs.get("scene_id"),
        "start_time": to_time(ds.attrs.get("time_coverage_start")),
        "end_time": to_time(ds.attrs.get("time_coverage_end")),
        "x_size": int(ds.sizes.get("x")) if "x" in ds.sizes else None,
        "y_size": int(ds.sizes.get("y")) if "y" in ds.sizes else None,
        "has_acm": "ACM" in ds.data_vars,
        "has_bcm": "BCM" in ds.data_vars,
        "has_cloud_probabilities": "Cloud_Probabilities" in ds.data_vars,
        "has_dqf": "DQF" in ds.data_vars,
        "scene_center_lat": center_lat,
        "scene_center_lon": center_lon,
    }
    ds.close()
    return row


def index_glm_file(path):
    ds = open_dataset(path)
    row = {
        "file_path": rel_path(path),
        "file_name": path.name,
        "source_type": "GLM",
        "product_name": get_product_name(ds),
        "dataset_name": ds.attrs.get("dataset_name"),
        "title": ds.attrs.get("title"),
        "platform_id": ds.attrs.get("platform_ID"),
        "start_time": to_time(ds.attrs.get("time_coverage_start")),
        "end_time": to_time(ds.attrs.get("time_coverage_end")),
        "event_count": int(ds.sizes.get("number_of_events", 0)),
        "group_count": int(ds.sizes.get("number_of_groups", 0)),
        "flash_count": int(ds.sizes.get("number_of_flashes", 0)),
    }
    ds.close()
    return row


def index_ibtracs_file(path):
    ds = open_dataset(path)
    date_count = ds.sizes["date_time"]

    sid_values = np.array([decode_text(value) for value in ds["sid"].values], dtype=object)
    name_values = np.array([decode_text(value) for value in ds["name"].values], dtype=object)
    season_values = ds["season"].values.astype(np.float32)

    sid_flat = np.repeat(sid_values, date_count)
    name_flat = np.repeat(name_values, date_count)
    season_flat = np.repeat(season_values, date_count)

    basin_flat = np.vectorize(decode_text, otypes=[object])(ds["basin"].values.reshape(-1))
    time_flat = ds["time"].values.reshape(-1)
    lat_flat = ds["lat"].values.reshape(-1).astype(np.float32)
    lon_flat = ds["lon"].values.reshape(-1).astype(np.float32)

    valid_mask = (~np.isnat(time_flat)) & np.isfinite(lat_flat) & np.isfinite(lon_flat)

    obs_data = {
        "file_path": np.full(valid_mask.sum(), rel_path(path), dtype=object),
        "sid": sid_flat[valid_mask],
        "name": name_flat[valid_mask],
        "season": season_flat[valid_mask],
        "basin": basin_flat[valid_mask],
        "time": pd.to_datetime(time_flat[valid_mask], utc=True),
        "lat": lat_flat[valid_mask],
        "lon": lon_flat[valid_mask],
    }

    if "wmo_wind" in ds:
        obs_data["wmo_wind"] = ds["wmo_wind"].values.reshape(-1)[valid_mask]
    if "wmo_pres" in ds:
        obs_data["wmo_pres"] = ds["wmo_pres"].values.reshape(-1)[valid_mask]
    if "usa_status" in ds:
        usa_status_flat = np.vectorize(decode_text, otypes=[object])(ds["usa_status"].values.reshape(-1))
        obs_data["usa_status"] = usa_status_flat[valid_mask]
    if "usa_wind" in ds:
        obs_data["usa_wind"] = ds["usa_wind"].values.reshape(-1)[valid_mask]

    obs_df = pd.DataFrame(obs_data)
    obs_df["season"] = obs_df["season"].where(np.isfinite(obs_df["season"]), np.nan)
    if "wmo_wind" in obs_df.columns:
        obs_df["wmo_wind"] = obs_df["wmo_wind"].where(np.isfinite(obs_df["wmo_wind"]), np.nan)
    if "wmo_pres" in obs_df.columns:
        obs_df["wmo_pres"] = obs_df["wmo_pres"].where(np.isfinite(obs_df["wmo_pres"]), np.nan)
    if "usa_wind" in obs_df.columns:
        obs_df["usa_wind"] = obs_df["usa_wind"].where(np.isfinite(obs_df["usa_wind"]), np.nan)

    storms_df = (
        obs_df.groupby(["sid", "name", "season"], dropna=False)
        .agg(
            file_path=("file_path", "first"),
            observation_count=("time", "count"),
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
        .reset_index()
    )

    ds.close()
    return storms_df, obs_df


def main():
    make_dirs()

    cmi_df = pd.DataFrame(index_cmi_file(path) for path in sorted(RAW_CMI_DIR.glob("**/*.nc")))
    acm_df = pd.DataFrame(index_acm_file(path) for path in sorted(RAW_ACM_DIR.glob("**/*.nc")))
    glm_df = pd.DataFrame(index_glm_file(path) for path in sorted(RAW_GLM_DIR.glob("**/*.nc")))

    ibtracs_paths = sorted(RAW_IBTRACS_DIR.glob("**/*.nc"))
    if not ibtracs_paths:
        raise RuntimeError("No IBTrACS files found.")
    storms_df, storm_obs_df = index_ibtracs_file(ibtracs_paths[0])

    cmi_df.to_csv(INDEX_DIR / "cmi_index.csv", index=False)
    acm_df.to_csv(INDEX_DIR / "acm_index.csv", index=False)
    glm_df.to_csv(INDEX_DIR / "glm_index.csv", index=False)
    storms_df.to_csv(INDEX_DIR / "ibtracs_storms.csv", index=False)
    storm_obs_df.to_csv(INDEX_DIR / "ibtracs_observations.csv.gz", index=False)

    inventory_rows = []
    for name, df in {
        "cmi_index": cmi_df,
        "acm_index": acm_df,
        "glm_index": glm_df,
        "ibtracs_storms": storms_df,
        "ibtracs_observations": storm_obs_df,
    }.items():
        inventory_rows.append(
            {
                "table_name": name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": ", ".join(df.columns),
            }
        )

    inventory_df = pd.DataFrame(inventory_rows)
    inventory_df.to_csv(INDEX_DIR / "inventory_summary.csv", index=False)

    print("Wrote index files to", INDEX_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
