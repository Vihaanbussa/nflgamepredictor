"""Attach current expected starting quarterbacks to 2026 NFL games."""

from pathlib import Path

import polars as pl

from src.build_features import (
    create_difference_features,
    join_draft_features,
    prepare_draft_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEASON = 2026


def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load schedules, depth charts, and the current draft class."""

    schedules = pl.read_parquet(
        RAW_DATA_DIR / "schedules.parquet"
    )

    depth_charts = pl.read_parquet(
        RAW_DATA_DIR / "depth_charts_2026.parquet"
    )

    draft_picks = pl.read_parquet(
        RAW_DATA_DIR / "draft_picks.parquet"
    )

    return schedules, depth_charts, draft_picks


def find_qb_candidates(
    depth_charts: pl.DataFrame,
    draft_picks: pl.DataFrame,
) -> pl.DataFrame:
    """Return every QB from each team's latest depth-chart snapshot."""

    quarterbacks = depth_charts.filter(pl.col("pos_abb") == "QB")

    latest_dates = quarterbacks.group_by("team").agg(
        pl.col("dt").max().alias("latest_dt")
    )

    quarterbacks = (
        quarterbacks
        .join(latest_dates, on="team", how="inner")
        .filter(pl.col("dt") == pl.col("latest_dt"))
    )

    drafted_qbs = (
        draft_picks
        .filter(
            (pl.col("season") == SEASON)
            & (pl.col("position") == "QB")
        )
        .select(
            "gsis_id",
            pl.col("round").alias("draft_round"),
            pl.col("pick").alias("draft_pick"),
        )
    )

    return (
        quarterbacks
        .join(drafted_qbs, on="gsis_id", how="left")
        .select(
            "team",
            "player_name",
            pl.col("gsis_id").alias("player_id"),
            pl.col("pos_rank").alias("depth_rank"),
            "draft_round",
            "draft_pick",
            pl.col("dt").alias("depth_chart_as_of"),
        )
        .sort(["team", "depth_rank"])
    )


def find_expected_starters(
    candidates: pl.DataFrame,
) -> pl.DataFrame:
    """Select each QB1 and flag first-round rookie challengers."""

    starters = (
        candidates
        .filter(pl.col("depth_rank") == 1)
        .select(
            "team",
            pl.col("player_name").alias("expected_qb_name"),
            pl.col("player_id").alias("expected_qb_id"),
            pl.col("draft_round").alias("expected_qb_draft_round"),
            pl.col("draft_pick").alias("expected_qb_draft_pick"),
            "depth_chart_as_of",
        )
        .sort("team")
    )

    challengers = (
        candidates
        .filter(
            (pl.col("depth_rank") > 1)
            & (pl.col("draft_round") == 1)
        )
        .sort(["team", "depth_rank"])
        .unique("team", keep="first", maintain_order=True)
        .select(
            "team",
            pl.col("player_name").alias("rookie_qb_challenger_name"),
            pl.col("player_id").alias("rookie_qb_challenger_id"),
            pl.col("draft_pick").alias("rookie_qb_challenger_pick"),
        )
    )

    starters = (
        starters
        .join(challengers, on="team", how="left")
        .with_columns(
            pl.col("rookie_qb_challenger_name")
            .is_not_null()
            .cast(pl.Int8)
            .alias("qb_competition")
        )
    )

    if starters.height != 32:
        raise ValueError(
            "Expected one QB1 for each of 32 teams, "
            f"but found {starters.height}."
        )

    return starters


def attach_starters_to_games(
    schedules: pl.DataFrame,
    starters: pl.DataFrame,
    draft_picks: pl.DataFrame,
) -> pl.DataFrame:
    """Add expected quarterbacks and rookie features to upcoming games."""

    games = schedules.filter(
        (pl.col("season") == SEASON)
        & (pl.col("game_type") == "REG")
        & pl.col("home_score").is_null()
    )

    home_starters = starters.rename(
        {
            column: (
                "home_team" if column == "team" else f"home_{column}"
            )
            for column in starters.columns
        }
    )

    away_starters = starters.rename(
        {
            column: (
                "away_team" if column == "team" else f"away_{column}"
            )
            for column in starters.columns
        }
    )

    games = games.join(
        home_starters,
        on="home_team",
        how="left",
    )

    games = games.join(
        away_starters,
        on="away_team",
        how="left",
    )

    draft_features = prepare_draft_features(draft_picks)
    games, draft_features = join_draft_features(
        games,
        draft_features,
    )
    games, draft_model_features = create_difference_features(
        games,
        draft_features,
    )

    games = games.select(
        "game_id",
        "season",
        "week",
        "gameday",
        "gametime",
        "away_team",
        "away_expected_qb_name",
        "away_expected_qb_id",
        "away_expected_qb_draft_round",
        "away_expected_qb_draft_pick",
        "away_depth_chart_as_of",
        "away_rookie_qb_challenger_name",
        "away_rookie_qb_challenger_id",
        "away_rookie_qb_challenger_pick",
        "away_qb_competition",
        "home_team",
        "home_expected_qb_name",
        "home_expected_qb_id",
        "home_expected_qb_draft_round",
        "home_expected_qb_draft_pick",
        "home_depth_chart_as_of",
        "home_rookie_qb_challenger_name",
        "home_rookie_qb_challenger_id",
        "home_rookie_qb_challenger_pick",
        "home_qb_competition",
        *draft_model_features,
    ).sort(["week", "gameday", "gametime", "game_id"])

    missing_qbs = games.filter(
        pl.col("home_expected_qb_id").is_null()
        | pl.col("away_expected_qb_id").is_null()
    )

    if missing_qbs.height > 0:
        raise ValueError(
            f"{missing_qbs.height} games are missing an expected quarterback."
        )

    return games


def save_outputs(
    starters: pl.DataFrame,
    candidates: pl.DataFrame,
    games: pl.DataFrame,
) -> None:
    """Save machine-readable Parquet and easy-to-view CSV outputs."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    starter_base = PROCESSED_DATA_DIR / "expected_qb_starters_2026"
    candidates_base = PROCESSED_DATA_DIR / "qb_candidates_2026"
    games_base = PROCESSED_DATA_DIR / "upcoming_games_2026"

    starters.write_parquet(starter_base.with_suffix(".parquet"))
    starters.write_csv(starter_base.with_suffix(".csv"))
    candidates.write_parquet(candidates_base.with_suffix(".parquet"))
    candidates.write_csv(candidates_base.with_suffix(".csv"))
    games.write_parquet(games_base.with_suffix(".parquet"))
    games.write_csv(games_base.with_suffix(".csv"))

    print(f"\nSaved {starters.height} expected starters.")
    print(f"Saved {candidates.height} quarterback candidates.")
    print(f"Saved expected quarterbacks for {games.height} upcoming games.")
    print(f"Starter list: {starter_base.with_suffix('.csv')}")
    print(f"Game list: {games_base.with_suffix('.csv')}")


def main() -> None:
    """Build the expected-starter and upcoming-game datasets."""

    print("Loading schedule and depth charts...")
    schedules, depth_charts, draft_picks = load_data()

    print("Finding current quarterback candidates...")
    candidates = find_qb_candidates(depth_charts, draft_picks)

    print("Selecting each team's latest QB1...")
    starters = find_expected_starters(candidates)

    print("Attaching expected quarterbacks to 2026 games...")
    games = attach_starters_to_games(
        schedules,
        starters,
        draft_picks,
    )

    save_outputs(starters, candidates, games)

    print("\nCurrent expected starters:")
    print(starters.select("team", "expected_qb_name"))


if __name__ == "__main__":
    main()
