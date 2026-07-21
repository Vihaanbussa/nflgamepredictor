"""Build model-ready features for upcoming 2026 NFL games."""

from pathlib import Path

import polars as pl

from src.build_features import (
    FORM_RAW_FEATURES,
    INJURY_FEATURES,
    PBP_RAW_FEATURES,
    QB_RAW_FEATURES,
    RAW_FEATURES,
    create_difference_features,
    create_matchup_features,
    create_sum_features,
    prepare_qb_stats,
    prepare_pbp_stats,
    prepare_team_form,
    prepare_team_stats,
)
from src.train import (
    MONEYLINE_FEATURE_COLUMNS,
    SPREAD_FEATURE_COLUMNS,
    TOTAL_FEATURE_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ROLLING_WINDOW = 5
TARGET_SEASON = 2026


def latest_snapshot(
    data: pl.DataFrame,
    key_column: str,
    raw_features: list[str],
) -> pl.DataFrame:
    """Average each team/player's latest five completed performances."""

    recent_rows = (
        data
        .drop_nulls(key_column)
        .sort([key_column, "season", "week"])
        .group_by(key_column, maintain_order=True)
        .tail(ROLLING_WINDOW)
    )

    return recent_rows.group_by(key_column).agg(
        [
            pl.col(feature).mean().alias(f"recent_{feature}")
            for feature in raw_features
        ]
    )


def join_snapshot(
    games: pl.DataFrame,
    snapshot: pl.DataFrame,
    snapshot_key: str,
    home_key: str,
    away_key: str,
    raw_features: list[str],
) -> pl.DataFrame:
    """Join a latest-form snapshot to both sides of each matchup."""

    recent_features = [f"recent_{feature}" for feature in raw_features]

    home = snapshot.rename(
        {
            snapshot_key: home_key,
            **{
                feature: f"home_{feature}"
                for feature in recent_features
            },
        }
    )

    away = snapshot.rename(
        {
            snapshot_key: away_key,
            **{
                feature: f"away_{feature}"
                for feature in recent_features
            },
        }
    )

    games = games.join(home, on=home_key, how="left")
    games = games.join(away, on=away_key, how="left")
    games, _ = create_difference_features(games, recent_features)
    games, _ = create_sum_features(games, recent_features)

    return games


def current_elo_ratings(schedules: pl.DataFrame) -> dict[str, float]:
    """Return ratings after completed games and 2026 offseason regression."""

    completed = (
        schedules
        .filter(
            (pl.col("game_type") == "REG")
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
            & (pl.col("season") <= TARGET_SEASON)
        )
        .sort(["season", "week", "gameday", "game_id"])
    )

    ratings: dict[str, float] = {}
    previous_season: int | None = None

    for game in completed.iter_rows(named=True):
        season = game["season"]

        if previous_season is not None and season != previous_season:
            ratings = {
                team: 1500.0 + 0.67 * (rating - 1500.0)
                for team, rating in ratings.items()
            }

        previous_season = season
        home = game["home_team"]
        away = game["away_team"]
        home_elo = ratings.get(home, 1500.0)
        away_elo = ratings.get(away, 1500.0)
        expected_home = 1.0 / (
            1.0 + 10.0 ** ((away_elo - (home_elo + 55.0)) / 400.0)
        )

        if game["home_score"] > game["away_score"]:
            actual_home = 1.0
        elif game["home_score"] < game["away_score"]:
            actual_home = 0.0
        else:
            actual_home = 0.5

        adjustment = 20.0 * (actual_home - expected_home)
        ratings[home] = home_elo + adjustment
        ratings[away] = away_elo - adjustment

    latest_season = completed.get_column("season").max()
    if latest_season is not None and latest_season >= TARGET_SEASON:
        return ratings

    return {
        team: 1500.0 + 0.67 * (rating - 1500.0)
        for team, rating in ratings.items()
    }


def main() -> None:
    """Create the full upcoming-game feature matrix."""

    schedules = pl.read_parquet(RAW_DATA_DIR / "schedules.parquet")
    team_stats = pl.read_parquet(RAW_DATA_DIR / "team_stats.parquet")
    player_stats = pl.read_parquet(RAW_DATA_DIR / "player_stats.parquet")
    play_by_play = pl.read_parquet(RAW_DATA_DIR / "play_by_play.parquet")
    games = pl.read_parquet(
        PROCESSED_DATA_DIR / "upcoming_games_2026.parquet"
    )

    market_columns = schedules.select(
        "game_id",
        "home_rest",
        "away_rest",
        "spread_line",
        "home_spread_odds",
        "away_spread_odds",
        "total_line",
        "over_odds",
        "under_odds",
        "home_moneyline",
        "away_moneyline",
        "location",
        "roof",
        "temp",
        "wind",
    )

    games = (
        games
        .drop("rest_diff")
        .join(market_columns, on="game_id", how="left")
        .with_columns(
            (pl.col("home_rest") - pl.col("away_rest")).alias("rest_diff"),
            (pl.col("location") == "Neutral")
            .cast(pl.Int8)
            .alias("neutral_site"),
            (pl.col("roof") == "outdoors")
            .cast(pl.Int8)
            .alias("is_outdoors"),
        )
    )

    print("Adding latest team efficiency...")
    prepared_team_stats = prepare_team_stats(team_stats)
    team_snapshot = latest_snapshot(
        prepared_team_stats,
        "team",
        RAW_FEATURES,
    )
    games = join_snapshot(
        games,
        team_snapshot,
        "team",
        "home_team",
        "away_team",
        RAW_FEATURES,
    )

    print("Adding latest scoring form...")
    prepared_form = prepare_team_form(schedules)
    form_snapshot = latest_snapshot(
        prepared_form,
        "team",
        FORM_RAW_FEATURES,
    )

    games = join_snapshot(
        games,
        form_snapshot,
        "team",
        "home_team",
        "away_team",
        FORM_RAW_FEATURES,
    )

    print("Adding offense-versus-defense matchups...")
    games = create_matchup_features(games)

    print("Adding play-by-play efficiency and pace...")
    prepared_pbp = prepare_pbp_stats(play_by_play)
    pbp_snapshot = latest_snapshot(
        prepared_pbp,
        "team",
        PBP_RAW_FEATURES,
    )
    games = join_snapshot(
        games,
        pbp_snapshot,
        "team",
        "home_team",
        "away_team",
        PBP_RAW_FEATURES,
    )

    print("Adding expected-quarterback form...")
    prepared_qbs = prepare_qb_stats(player_stats)
    qb_snapshot = latest_snapshot(
        prepared_qbs,
        "player_id",
        QB_RAW_FEATURES,
    )
    games = join_snapshot(
        games,
        qb_snapshot,
        "player_id",
        "home_expected_qb_id",
        "away_expected_qb_id",
        QB_RAW_FEATURES,
    )

    # Pregame 2026 injury reports are not published yet. Zero difference is
    # neutral, and the models' imputers handle unavailable player form.
    games = games.with_columns(
        [
            pl.lit(0.0).alias(f"{feature}_{suffix}")
            for feature in INJURY_FEATURES
            for suffix in ["diff", "sum"]
        ]
    )

    ratings = current_elo_ratings(schedules)
    rating_table = pl.DataFrame(
        {
            "team": list(ratings.keys()),
            "current_elo": list(ratings.values()),
        }
    )
    games = (
        games
        .join(
            rating_table.rename(
                {"team": "home_team", "current_elo": "home_elo"}
            ),
            on="home_team",
            how="left",
        )
        .join(
            rating_table.rename(
                {"team": "away_team", "current_elo": "away_elo"}
            ),
            on="away_team",
            how="left",
        )
        .with_columns(
            (pl.col("home_elo") - pl.col("away_elo")).alias("elo_diff")
        )
    )

    required_columns = set(
        MONEYLINE_FEATURE_COLUMNS
        + SPREAD_FEATURE_COLUMNS
        + TOTAL_FEATURE_COLUMNS
    )
    missing_columns = sorted(required_columns - set(games.columns))

    if missing_columns:
        raise ValueError(f"Upcoming features are missing: {missing_columns}")

    output_path = PROCESSED_DATA_DIR / "upcoming_features_2026.parquet"
    games.write_parquet(output_path)
    games.write_csv(output_path.with_suffix(".csv"))

    print(f"Saved features for {games.height} games: {output_path}")


if __name__ == "__main__":
    main()
