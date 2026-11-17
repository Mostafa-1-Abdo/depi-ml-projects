import gc
import json
import random
from pathlib import Path
from typing import Dict, List

import ijson
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# 1. Configuration & Paths
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "model_build_inputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "dataset_01"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
RANDOM_SAMPLES_PER_STOP = 15  # Number of random target stops sampled per origin stop


# ==========================================
# 2. Leakage-Free Route Splitting
# ==========================================
def get_route_splits(route_ids: List[str]) -> Dict[str, str]:
    """Splits route_ids into 70% Train, 15% Validation, and 15% Test."""
    train_ids, temp_ids = train_test_split(
        route_ids, test_size=0.30, random_state=RANDOM_SEED
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=RANDOM_SEED
    )

    split_map = {}
    for r_id in train_ids:
        split_map[r_id] = "train"
    for r_id in val_ids:
        split_map[r_id] = "val"
    for r_id in test_ids:
        split_map[r_id] = "test"

    return split_map


# ==========================================
# 3. Vectorized Geospatial Distance Features
# ==========================================
def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes Haversine distance, Manhattan distance, and coordinate deltas."""
    earth_radius_km = 6371.0

    lat1 = np.radians(df["from_lat"].values)
    lon1 = np.radians(df["from_lng"].values)
    lat2 = np.radians(df["to_lat"].values)
    lon2 = np.radians(df["to_lng"].values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine distance
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["haversine_km"] = (earth_radius_km * c).astype(np.float32)

    # Manhattan distance approximation in km
    lat_dist = np.abs(dlat) * earth_radius_km
    lon_dist = np.abs(dlon) * earth_radius_km * np.cos((lat1 + lat2) / 2.0)
    df["manhattan_km"] = (lat_dist + lon_dist).astype(np.float32)

    df["delta_lat"] = (df["to_lat"] - df["from_lat"]).astype(np.float32)
    df["delta_lng"] = (df["to_lng"] - df["from_lng"]).astype(np.float32)

    return df


# ==========================================
# 4. Tier 2 Dataset Builder (Service Time - DL)
# ==========================================
def build_tier2_datasets(
    route_data: dict, package_data_path: Path, split_map: Dict[str, str]
) -> None:
    print("\n[1/2] Loading package_data.json and building Tier 2 datasets...")
    with open(package_data_path, "r", encoding="utf-8") as f:
        package_data = json.load(f)

    records = {"train": [], "val": [], "test": []}

    for route_id, r_info in tqdm(route_data.items(), desc="Processing Tier 2"):
        split = split_map[route_id]
        station_code = r_info.get("station_code", "UNKNOWN")
        stops = r_info.get("stops", {})
        route_pkgs = package_data.get(route_id, {})

        for stop_id, s_info in stops.items():
            stop_pkgs = route_pkgs.get(stop_id, {})

            # Skip stops with no packages (e.g., depot/station stops)
            if not stop_pkgs:
                continue

            pkg_count = len(stop_pkgs)
            total_vol = 0.0
            max_vol = 0.0
            has_tw = 0
            total_service_time = 0.0

            for _, p_info in stop_pkgs.items():
                dims = p_info.get("dimensions", {})
                vol = (
                    dims.get("depth_cm", 0.0)
                    * dims.get("height_cm", 0.0)
                    * dims.get("width_cm", 0.0)
                )
                total_vol += vol
                if vol > max_vol:
                    max_vol = vol

                tw = p_info.get("time_window", {})
                if tw and (tw.get("start_time_utc") or tw.get("end_time_utc")):
                    has_tw = 1

                total_service_time += p_info.get(
                    "planned_service_time_seconds", 0.0)

            records[split].append(
                {
                    "station_code": station_code,
                    "stop_type": s_info.get("type", "Dropoff"),
                    "zone_id": str(s_info.get("zone_id", "NONE")),
                    "lat": s_info.get("lat", 0.0),
                    "lng": s_info.get("lng", 0.0),
                    "package_count": pkg_count,
                    "total_volume_cm3": total_vol,
                    "max_package_volume_cm3": max_vol,
                    "has_time_window": has_tw,
                    "target_service_time": total_service_time,
                }
            )

    del package_data
    gc.collect()

    for split in ["train", "val", "test"]:
        df = pd.DataFrame(records[split])
        for col in ["station_code", "stop_type", "zone_id"]:
            df[col] = df[col].astype("category")
        for col in [
            "lat",
            "lng",
            "total_volume_cm3",
            "max_package_volume_cm3",
            "target_service_time",
        ]:
            df[col] = df[col].astype(np.float32)
        for col in ["package_count", "has_time_window"]:
            df[col] = df[col].astype(np.int32)

        out_path = PROCESSED_DIR / f"tier2_{split}.parquet"
        df.to_parquet(out_path, engine="pyarrow",
                      compression="snappy", index=False)
        print(f" -> Saved {out_path.name}: {len(df):,} rows")


# ==========================================
# 5. Tier 1 Dataset Builder (Travel Time - ML)
# ==========================================
def build_tier1_datasets(
    route_data: dict,
    actual_sequences: dict,
    travel_times_path: Path,
    split_map: Dict[str, str],
) -> None:
    print("\n[2/2] Streaming travel_times.json and building Tier 1 datasets...")
    rng = random.Random(RANDOM_SEED)
    records = {"train": [], "val": [], "test": []}

    with open(travel_times_path, "rb") as f:
        # Stream route-by-route using ijson to keep RAM usage minimal
        parser = ijson.kvitems(f, "")

        for route_id, tt_matrix in tqdm(
            parser, total=len(route_data), desc="Streaming Tier 1"
        ):
            if route_id not in route_data:
                continue

            split = split_map[route_id]
            r_info = route_data[route_id]
            stops = r_info.get("stops", {})
            station_code = r_info.get("station_code", "UNKNOWN")
            capacity = float(r_info.get("executor_capacity_cm3", 0.0))

            # Extract departure hour and day of week
            date_str = r_info.get("date_YYYY_MM_DD", "2021-01-01")
            dep_time = r_info.get("departure_time_utc", "00:00:00")
            dt = pd.to_datetime(f"{date_str} {dep_time}", errors="coerce")
            dep_hour = dt.hour if pd.notnull(dt) else 0
            day_of_week = dt.dayofweek if pd.notnull(dt) else 0

            # 1. Include all actual observed transitions
            seq_dict = actual_sequences.get(route_id, {}).get("actual", {})
            ordered_stops = sorted(seq_dict.keys(), key=lambda s: seq_dict[s])

            selected_pairs = set()
            for i in range(len(ordered_stops) - 1):
                selected_pairs.add((ordered_stops[i], ordered_stops[i + 1]))

            # 2. Add random stop pairs per origin stop to cover non-visited edges
            all_stop_ids = list(stops.keys())
            for from_stop in all_stop_ids:
                candidates = [s for s in all_stop_ids if s != from_stop]
                sampled = rng.sample(
                    candidates, min(RANDOM_SAMPLES_PER_STOP, len(candidates))
                )
                for to_stop in sampled:
                    selected_pairs.add((from_stop, to_stop))

            # 3. Extract features and target transit time
            for from_stop, to_stop in selected_pairs:
                transit_sec = tt_matrix.get(from_stop, {}).get(to_stop)
                if transit_sec is None:
                    continue

                s_from = stops[from_stop]
                s_to = stops[to_stop]
                z_from = str(s_from.get("zone_id", "NONE"))
                z_to = str(s_to.get("zone_id", "NONE"))

                records[split].append(
                    {
                        "station_code": station_code,
                        "departure_hour": dep_hour,
                        "day_of_week": day_of_week,
                        "executor_capacity_cm3": capacity,
                        "from_type": s_from.get("type", "Dropoff"),
                        "to_type": s_to.get("type", "Dropoff"),
                        "from_zone": z_from,
                        "to_zone": z_to,
                        "is_same_zone": 1 if (z_from == z_to and z_from != "NONE") else 0,
                        "from_lat": s_from.get("lat", 0.0),
                        "from_lng": s_from.get("lng", 0.0),
                        "to_lat": s_to.get("lat", 0.0),
                        "to_lng": s_to.get("lng", 0.0),
                        "target_transit_time": float(transit_sec),
                    }
                )

    for split in ["train", "val", "test"]:
        print(f"Processing and compressing tier1_{split}...")
        df = pd.DataFrame(records[split])
        records[split] = []
        gc.collect()

        df = add_distance_features(df)

        for col in ["station_code", "from_type", "to_type", "from_zone", "to_zone"]:
            df[col] = df[col].astype("category")
        for col in ["departure_hour", "day_of_week", "is_same_zone"]:
            df[col] = df[col].astype(np.int32)
        for col in [
            "executor_capacity_cm3",
            "from_lat",
            "from_lng",
            "to_lat",
            "to_lng",
            "target_transit_time",
        ]:
            df[col] = df[col].astype(np.float32)

        out_path = PROCESSED_DIR / f"tier1_{split}.parquet"
        df.to_parquet(out_path, engine="pyarrow",
                      compression="snappy", index=False)
        print(f" -> Saved {out_path.name}: {len(df):,} rows")
        del df
        gc.collect()


# ==========================================
# 6. Main Execution Pipeline
# ==========================================
def main() -> None:
    route_data_path = RAW_DIR / "route_data.json"
    actual_seq_path = RAW_DIR / "actual_sequences.json"
    package_data_path = RAW_DIR / "package_data.json"
    travel_times_path = RAW_DIR / "travel_times.json"

    print("Loading route_data.json and actual_sequences.json...")
    with open(route_data_path, "r", encoding="utf-8") as f:
        route_data = json.load(f)

    with open(actual_seq_path, "r", encoding="utf-8") as f:
        actual_sequences = json.load(f)

    all_route_ids = list(route_data.keys())
    print(f"Total routes found: {len(all_route_ids):,}")

    split_map = get_route_splits(all_route_ids)

    build_tier2_datasets(route_data, package_data_path, split_map)
    build_tier1_datasets(route_data, actual_sequences,
                         travel_times_path, split_map)

    print("\nAll Parquet datasets generated successfully in data/processed/!")


if __name__ == "__main__":
    main()
