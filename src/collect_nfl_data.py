from pathlib import Path
import nflreadpy as nfl
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1] #sets the project root directory to the parent of the current file's parent directory
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" #allows me to access the raw data directory by joining the project root with the "data/raw" path

TARGET_SEASON = 2026
HISTORICAL_SEASONS = list(range(2018, TARGET_SEASON + 1))
SCHEDULE_SEASONS = list(range(2018, 2027))  # 2018-2026 seasons
DRAFT_SEASONS = list(range(2018, 2027))

PLAY_BY_PLAY_COLUMNS = [
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "play_type",
    "pass_attempt",
    "rush_attempt",
    "complete_pass",
    "first_down",
    "epa",
    "success",
    "down",
    "qtr",
    "yardline_100",
    "touchdown",
    "drive",
    "drive_ended_with_score",
    "score_differential",
    "game_seconds_remaining",
    "qb_kneel",
    "qb_spike",
    "no_huddle",
]

def save_dataset(
    dataframe: pl.DataFrame,
    filename: str,
) -> None:

    output_path = RAW_DATA_DIR / filename
    dataframe.write_parquet(output_path)

    print(
        f"Saved {filename}: "
        f"{dataframe.height:,} rows, "
        f"{dataframe.width:,} columns"
    )
    print(f"Location: {output_path}")


def download_schedules() -> None:

    print("\nDownloading schedules...")

    schedules = nfl.load_schedules(SCHEDULE_SEASONS)

    save_dataset(
        dataframe=schedules,
        filename="schedules.parquet",
    )


def download_team_stats() -> None:

    print("\nDownloading weekly team statistics...")

    team_stats = nfl.load_team_stats(
        seasons=HISTORICAL_SEASONS,
        summary_level="week",
    )

    save_dataset(
        dataframe=team_stats,
        filename="team_stats.parquet",
    )


def download_player_stats() -> None:

    print("\nDownloading weekly player statistics...")

    player_stats = nfl.load_player_stats(
        seasons=HISTORICAL_SEASONS,
        summary_level="week",
    )

    save_dataset(
        dataframe=player_stats,
        filename="player_stats.parquet",
    )


def download_play_by_play() -> None:

    print("\nDownloading play-by-play data...")

    season_frames = []
    for season in HISTORICAL_SEASONS:
        print(f"Downloading {season} play-by-play...")
        season_frames.append(
            nfl.load_pbp(seasons=[season]).select(PLAY_BY_PLAY_COLUMNS)
        )

    play_by_play = pl.concat(season_frames, how="diagonal_relaxed")
    save_dataset(play_by_play, "play_by_play.parquet")


def download_injuries() -> None:

    print("\nDownloading weekly injury reports...")

    injuries = nfl.load_injuries(
        seasons=HISTORICAL_SEASONS,
    )

    save_dataset(
        dataframe=injuries,
        filename="injuries.parquet",
    )


def download_depth_charts() -> None:

    print("\nDownloading 2026 depth charts...")

    depth_charts = nfl.load_depth_charts(
        seasons=[2026],
    )

    save_dataset(
        dataframe=depth_charts,
        filename=f"depth_charts_{TARGET_SEASON}.parquet",
    )


def download_rosters_2026() -> None:

    print("\nDownloading current 2026 rosters...")

    rosters = nfl.load_rosters(
        seasons=[2026],
    )

    save_dataset(
        dataframe=rosters,
        filename=f"rosters_{TARGET_SEASON}.parquet",
    )


def download_draft_picks() -> None:

    print("\nDownloading 2018-2026 draft picks...")

    draft_picks = nfl.load_draft_picks(
        seasons=DRAFT_SEASONS,
    )

    save_dataset(
        dataframe=draft_picks,
        filename="draft_picks.parquet",
    )


def main() -> None:

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting NFL data download...")
    print(f"Historical seasons: {HISTORICAL_SEASONS}")
    print(f"Schedule seasons: {SCHEDULE_SEASONS}")

    download_schedules()
    download_team_stats()
    download_player_stats()
    download_play_by_play()
    download_injuries()
    download_depth_charts()
    download_rosters_2026()
    download_draft_picks()

    print("\nAll NFL data downloaded successfully.")


if __name__ == "__main__":
    main()
