from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "dataset_01"


def verify_tier1() -> None:
    print("=== Verifying Tier 1 Datasets (Travel Time) ===")
    splits = {
        s: pd.read_parquet(PROCESSED_DIR / f"tier1_{s}.parquet")
        for s in ["train", "val", "test"]
    }

  # 1. Schema check
    base_cols = list(splits["train"].columns)
    base_dtypes = {col: str(dt) for col, dt in splits["train"].dtypes.items()}
    for s in ["val", "test"]:
        assert list(
            splits[s].columns) == base_cols, f"Column mismatch in tier1_{s}"
        curr_dtypes = {col: str(dt) for col, dt in splits[s].dtypes.items()}
        assert curr_dtypes == base_dtypes, f"Dtype mismatch in tier1_{s}: {curr_dtypes} vs {base_dtypes}"
    print("[PASS] Schema and dtypes match across train/val/test.")

    # 2. Null & Inf check
    for s, df in splits.items():
        nulls = df.isna().sum().sum()
        num_cols = df.select_dtypes(include=[np.number]).columns
        infs = np.isinf(df[num_cols]).sum().sum()
        assert nulls == 0, f"Found {nulls} NaNs in tier1_{s}"
        assert infs == 0, f"Found {infs} Infs in tier1_{s}"
    print("[PASS] Zero NaNs and zero Infs across all splits.")

    # 3. Physical & Coordinate bounds check
    for s, df in splits.items():
        assert (df["target_transit_time"] >= 0).all(
        ), f"Negative transit time in tier1_{s}"
        assert (df["haversine_km"] >= 0).all(
        ), f"Negative distance in tier1_{s}"
        assert (df["from_lat"] != 0).all() and (
            df["to_lat"] != 0).all(), f"Zero lat found in tier1_{s}"
        assert (df["from_lng"] != 0).all() and (
            df["to_lng"] != 0).all(), f"Zero lng found in tier1_{s}"
    print("[PASS] Physical ranges and coordinates are valid (no 0.0 fallback).")

    # 4. Station coverage check
    train_stations = set(splits["train"]["station_code"].unique())
    for s in ["val", "test"]:
        unseen = set(splits[s]["station_code"].unique()) - train_stations
        assert len(unseen) == 0, f"Unseen stations in tier1_{s}: {unseen}"
    print(
        f"[PASS] All {len(train_stations)} stations are covered in train split.")


def verify_tier2() -> None:
    print("\n=== Verifying Tier 2 Datasets (Service Time) ===")
    splits = {
        s: pd.read_parquet(PROCESSED_DIR / f"tier2_{s}.parquet")
        for s in ["train", "val", "test"]
    }

    # 1. Schema check
    base_cols = list(splits["train"].columns)
    for s in ["val", "test"]:
        assert list(
            splits[s].columns) == base_cols, f"Column mismatch in tier2_{s}"
    print("[PASS] Schema matches across train/val/test.")

    # 2. Null & Range check
    for s, df in splits.items():
        assert df.isna().sum().sum() == 0, f"Found NaNs in tier2_{s}"
        assert (df["package_count"] > 0).all(
        ), f"Zero/negative package_count in tier2_{s}"
        assert (df["target_service_time"] >= 0).all(
        ), f"Negative service time in tier2_{s}"
        assert (df["total_volume_cm3"] >= 0).all(
        ), f"Negative volume in tier2_{s}"
    print("[PASS] Zero NaNs and valid package/service ranges.")


if __name__ == "__main__":
    verify_tier1()
    verify_tier2()
    print("\nALL DATA QUALITY CHECKS PASSED SUCCESSFULLY!")
