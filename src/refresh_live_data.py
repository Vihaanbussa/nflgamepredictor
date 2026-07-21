"""Refresh 2026 data and rebuild features for games that have not been played."""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from src.build_expected_starters import main as build_expected_starters
from src.build_upcoming_features import main as build_upcoming_features
from src.collect_nfl_data import PLAY_BY_PLAY_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
STATUS_PATH = PROCESSED_DATA_DIR / "live_refresh_status.json"
TARGET_SEASON = 2026
REFRESH_LOCK = threading.Lock()


def replace_season(path: Path, new_data: pl.DataFrame) -> None:
    """Preserve historical rows and replace the current-season portion."""

    if path.exists():
        historical = pl.read_parquet(path).filter(pl.col("season") != TARGET_SEASON)
        combined = pl.concat([historical, new_data], how="diagonal_relaxed")
    else:
        combined = new_data
    combined.write_parquet(path)


def read_status() -> dict:
    if os.getenv("NFL_AUTO_REFRESH", "1") == "0":
        return {
            "state": "disabled",
            "message": "Automatic refresh is disabled; using cached local data.",
            "last_refresh": None,
            "latest_completed_week": None,
        }
    if not STATUS_PATH.exists():
        return {"state": "never_refreshed", "last_refresh": None}
    return json.loads(STATUS_PATH.read_text())


def write_status(state: str, message: str, completed_week: int | None = None) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps({
        "state": state,
        "message": message,
        "last_refresh": datetime.now(timezone.utc).isoformat(),
        "latest_completed_week": completed_week,
    }, indent=2))


def is_stale(max_age_hours: float = 12.0) -> bool:
    status = read_status()
    last_refresh = status.get("last_refresh")
    if not last_refresh:
        return True
    age = datetime.now(timezone.utc) - datetime.fromisoformat(last_refresh)
    return age.total_seconds() >= max_age_hours * 3600


def refresh_now() -> dict:
    """Download current-season data and rebuild remaining-game features."""

    with REFRESH_LOCK:
        write_status("refreshing", "Downloading the latest 2026 NFL data.")
        try:
            schedules = nfl.load_schedules(list(range(2018, TARGET_SEASON + 1)))
            schedules.write_parquet(RAW_DATA_DIR / "schedules.parquet")

            current_team = nfl.load_team_stats([TARGET_SEASON], summary_level="week")
            current_player = nfl.load_player_stats([TARGET_SEASON], summary_level="week")
            current_pbp = nfl.load_pbp([TARGET_SEASON]).select(PLAY_BY_PLAY_COLUMNS)
            current_injuries = nfl.load_injuries([TARGET_SEASON])
            replace_season(RAW_DATA_DIR / "team_stats.parquet", current_team)
            replace_season(RAW_DATA_DIR / "player_stats.parquet", current_player)
            replace_season(RAW_DATA_DIR / "play_by_play.parquet", current_pbp)
            replace_season(RAW_DATA_DIR / "injuries.parquet", current_injuries)

            nfl.load_depth_charts([TARGET_SEASON]).write_parquet(
                RAW_DATA_DIR / "depth_charts_2026.parquet"
            )
            nfl.load_rosters([TARGET_SEASON]).write_parquet(
                RAW_DATA_DIR / "rosters_2026.parquet"
            )

            build_expected_starters()
            build_upcoming_features()

            completed = schedules.filter(
                (pl.col("season") == TARGET_SEASON)
                & (pl.col("game_type") == "REG")
                & pl.col("home_score").is_not_null()
            )
            latest_week = completed.get_column("week").max() if completed.height else None
            write_status(
                "ready",
                "Live data and remaining-game features are current.",
                latest_week,
            )
        except Exception as exc:
            write_status(
                "using_cached_data",
                f"Live refresh failed; using the last saved data: {exc}",
            )
        return read_status()


def refresh_if_stale() -> dict:
    if os.getenv("NFL_AUTO_REFRESH", "1") == "0":
        return {"state": "disabled", "message": "Automatic refresh is disabled."}
    max_age = float(os.getenv("NFL_REFRESH_HOURS", "12"))
    return refresh_now() if is_stale(max_age) else read_status()
