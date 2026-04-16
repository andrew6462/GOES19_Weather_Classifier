import numpy as np
import pandas as pd

from pipeline_utils import INDEX_DIR, PROCESSED_DIR, distance_km, overlap_seconds


def nearest_storm_info(storm_obs_df, start_time, end_time, center_time, center_lat, center_lon):
    close_storms = storm_obs_df[
        (storm_obs_df["time"] >= start_time - pd.Timedelta(hours=6))
        & (storm_obs_df["time"] <= end_time + pd.Timedelta(hours=6))
    ].copy()

    if close_storms.empty:
        return False, None, None, None, None, None

    close_storms["time_delta_minutes"] = (
        close_storms["time"] - center_time
    ).abs().dt.total_seconds() / 60.0

    if pd.notna(center_lat) and pd.notna(center_lon):
        close_storms["distance_to_scene_km"] = close_storms.apply(
            lambda storm: distance_km(
                float(center_lat),
                float(center_lon),
                float(storm["lat"]),
                float(storm["lon"]),
            ),
            axis=1,
        )
    else:
        close_storms["distance_to_scene_km"] = np.nan

    close_storms = close_storms.sort_values("time_delta_minutes")
    nearest = close_storms.iloc[0]
    storm_overlap = bool(
        ((close_storms["time"] >= start_time) & (close_storms["time"] <= end_time)).any()
    )

    return (
        storm_overlap,
        nearest["sid"],
        nearest["name"],
        nearest["basin"],
        float(nearest["time_delta_minutes"]),
        nearest["distance_to_scene_km"],
    )


def main():
    cmi_df = pd.read_csv(INDEX_DIR / "cmi_index.csv")
    acm_df = pd.read_csv(INDEX_DIR / "acm_index.csv")
    glm_df = pd.read_csv(INDEX_DIR / "glm_index.csv")
    storm_obs_df = pd.read_csv(INDEX_DIR / "ibtracs_observations.csv.gz")

    cmi_df["start_time"] = pd.to_datetime(cmi_df["start_time"], utc=True, format="mixed")
    cmi_df["end_time"] = pd.to_datetime(cmi_df["end_time"], utc=True, format="mixed")
    acm_df["start_time"] = pd.to_datetime(acm_df["start_time"], utc=True, format="mixed")
    acm_df["end_time"] = pd.to_datetime(acm_df["end_time"], utc=True, format="mixed")
    glm_df["start_time"] = pd.to_datetime(glm_df["start_time"], utc=True, format="mixed")
    glm_df["end_time"] = pd.to_datetime(glm_df["end_time"], utc=True, format="mixed")
    storm_obs_df["time"] = pd.to_datetime(storm_obs_df["time"], utc=True, format="mixed")

    rows = []
    for cmi_row in cmi_df.itertuples(index=False):
        candidates = acm_df[
            (acm_df["platform_id"] == cmi_row.platform_id)
            & (acm_df["scene_id"] == cmi_row.scene_id)
            & (acm_df["x_size"] == cmi_row.x_size)
            & (acm_df["y_size"] == cmi_row.y_size)
        ]

        for acm_row in candidates.itertuples(index=False):
            overlap = overlap_seconds(
                cmi_row.start_time,
                cmi_row.end_time,
                acm_row.start_time,
                acm_row.end_time,
            )
            if overlap <= 0:
                continue

            start_time = max(cmi_row.start_time, acm_row.start_time)
            end_time = min(cmi_row.end_time, acm_row.end_time)
            center_time = start_time + (end_time - start_time) / 2
            scene_key = cmi_row.file_name

            glm_candidates = glm_df[glm_df["platform_id"] == cmi_row.platform_id]
            glm_matches = glm_candidates[
                glm_candidates.apply(
                    lambda row: overlap_seconds(
                        start_time,
                        end_time,
                        row["start_time"],
                        row["end_time"],
                    ) > 0,
                    axis=1,
                )
            ]

            (
                storm_overlap,
                nearest_storm_id,
                nearest_storm_name,
                nearest_storm_basin,
                nearest_storm_time_delta_minutes,
                nearest_storm_distance_km,
            ) = nearest_storm_info(
                storm_obs_df,
                start_time,
                end_time,
                center_time,
                cmi_row.scene_center_lat,
                cmi_row.scene_center_lon,
            )

            rows.append(
                {
                    "scene_id": cmi_row.scene_id,
                    "scene_key": scene_key,
                    "platform_id": cmi_row.platform_id,
                    "cmi_path": cmi_row.file_path,
                    "acm_path": acm_row.file_path,
                    "cmi_file_name": cmi_row.file_name,
                    "acm_file_name": acm_row.file_name,
                    "x_size": int(cmi_row.x_size),
                    "y_size": int(cmi_row.y_size),
                    "start_time": start_time,
                    "end_time": end_time,
                    "scene_center_time": center_time,
                    "overlap_seconds": overlap,
                    "cmi_band_id": cmi_row.band_id,
                    "cmi_band_wavelength": cmi_row.band_wavelength,
                    "scene_center_lat": cmi_row.scene_center_lat,
                    "scene_center_lon": cmi_row.scene_center_lon,
                    "matched_glm_files": len(glm_matches),
                    "glm_flash_count_sum": int(glm_matches["flash_count"].sum()) if len(glm_matches) else 0,
                    "glm_group_count_sum": int(glm_matches["group_count"].sum()) if len(glm_matches) else 0,
                    "glm_event_count_sum": int(glm_matches["event_count"].sum()) if len(glm_matches) else 0,
                    "storm_overlap": storm_overlap,
                    "nearest_storm_id": nearest_storm_id,
                    "nearest_storm_name": nearest_storm_name,
                    "nearest_storm_basin": nearest_storm_basin,
                    "nearest_storm_time_delta_minutes": nearest_storm_time_delta_minutes,
                    "nearest_storm_distance_km": nearest_storm_distance_km,
                }
            )

    matches_df = pd.DataFrame(rows)
    if len(matches_df) == 0:
        raise RuntimeError("No CMI/ACM scene matches were found.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "matched_scenes.csv"
    matches_df.to_csv(output_path, index=False)
    print(f"Wrote {len(matches_df)} matched scenes to {output_path}")
    return 0


if __name__ == "__main__":
    main()
