"""Generate moneyline, spread, and over/under predictions."""

from pathlib import Path

import joblib
import numpy as np
import polars as pl

from src.train_score_models import probability_above


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def predict_probability(
    games: pl.DataFrame,
    artifact_name: str,
) -> np.ndarray:
    """Load a saved classifier and return positive-class probabilities."""

    artifact = joblib.load(MODELS_DIR / artifact_name)
    features = games.select(artifact["feature_columns"]).to_numpy()
    return artifact["model"].predict_proba(features)[:, 1]


def predict_score_model(
    games: pl.DataFrame,
    artifact_name: str,
    clip_nonnegative: bool = True,
) -> tuple[np.ndarray, dict]:
    """Return score predictions and their saved model metadata."""

    artifact = joblib.load(MODELS_DIR / artifact_name)
    features = games.select(artifact["feature_columns"]).to_numpy()
    prediction = artifact["model"].predict(features)
    if clip_nonnegative:
        prediction = np.maximum(prediction, 0.0)
    return prediction, artifact


def american_implied_probability(odds: pl.Expr) -> pl.Expr:
    """Convert American odds into their break-even implied probability."""

    return (
        pl.when(odds < 0)
        .then((-odds) / ((-odds) + 100.0))
        .otherwise(100.0 / (odds + 100.0))
    )


def main() -> None:
    """Generate and save predictions for every scheduled 2026 game."""

    games = pl.read_parquet(
        PROCESSED_DATA_DIR / "upcoming_features_2026.parquet"
    )

    predicted_home, _ = predict_score_model(games, "home_score_model.joblib")
    predicted_away, _ = predict_score_model(games, "away_score_model.joblib")
    predicted_residual, residual_artifact = predict_score_model(
        games,
        "total_residual_model.joblib",
        clip_nonnegative=False,
    )
    simulation = joblib.load(MODELS_DIR / "score_simulation.joblib")
    covariance = simulation["residual_covariance"]
    margin_std = float(
        np.sqrt(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    )
    predicted_margin = predicted_home - predicted_away
    score_total = predicted_home + predicted_away
    home_win_probability = probability_above(
        predicted_margin,
        np.zeros(games.height),
        margin_std,
    )

    total_lines = games.get_column("total_line").to_numpy()
    has_total_line = ~np.isnan(total_lines)
    over_probability = probability_above(
        predicted_residual,
        np.zeros(games.height),
        float(residual_artifact["residual_std"]),
    )
    over_probability_values = [
        float(probability) if available else None
        for probability, available in zip(over_probability, has_total_line)
    ]
    predicted_total = np.where(
        has_total_line,
        total_lines + predicted_residual,
        score_total,
    )

    games = games.with_columns(
        pl.Series(
            "home_win_probability",
            home_win_probability,
        ),
        pl.Series(
            "home_cover_probability",
            predict_probability(games, "spread_model.joblib"),
        ),
        pl.Series(
            "over_probability",
            over_probability_values,
        ),
        pl.Series("predicted_home_score", predicted_home),
        pl.Series("predicted_away_score", predicted_away),
        pl.Series("predicted_total", predicted_total),
        pl.Series("predicted_margin", predicted_margin),
    )

    games = games.with_columns(
        (1.0 - pl.col("home_win_probability")).alias(
            "away_win_probability"
        ),
        (1.0 - pl.col("home_cover_probability")).alias(
            "away_cover_probability"
        ),
        (1.0 - pl.col("over_probability")).alias("under_probability"),
        american_implied_probability(pl.col("home_moneyline")).alias(
            "home_raw_implied_probability"
        ),
        american_implied_probability(pl.col("away_moneyline")).alias(
            "away_raw_implied_probability"
        ),
    )

    games = games.with_columns(
        pl.when(pl.col("home_win_probability") >= 0.5)
        .then(pl.col("home_team"))
        .otherwise(pl.col("away_team"))
        .alias("moneyline_pick"),

        pl.when(pl.col("spread_line").is_null())
        .then(None)
        .when(pl.col("home_cover_probability") >= 0.5)
        .then(pl.col("home_team"))
        .otherwise(pl.col("away_team"))
        .alias("spread_pick"),

        pl.when(pl.col("spread_line").is_null())
        .then(None)
        .when(pl.col("home_cover_probability") >= 0.5)
        .then(-pl.col("spread_line"))
        .otherwise(pl.col("spread_line"))
        .alias("spread_pick_line"),

        pl.when(pl.col("total_line").is_null())
        .then(None)
        .when(pl.col("over_probability") >= 0.5)
        .then(pl.lit("OVER"))
        .otherwise(pl.lit("UNDER"))
        .alias("total_pick"),
    )

    games = games.with_columns(
        (
            pl.col("home_raw_implied_probability")
            / (
                pl.col("home_raw_implied_probability")
                + pl.col("away_raw_implied_probability")
            )
        ).alias("home_no_vig_implied_probability")
    )

    games = games.with_columns(
        (
            pl.col("home_win_probability")
            - pl.col("home_no_vig_implied_probability")
        ).alias("home_moneyline_probability_edge")
    )

    output_columns = [
        "game_id",
        "week",
        "gameday",
        "gametime",
        "away_team",
        "away_expected_qb_name",
        "away_rookie_qb_challenger_name",
        "home_team",
        "home_expected_qb_name",
        "home_rookie_qb_challenger_name",
        "predicted_away_score",
        "predicted_home_score",
        "predicted_total",
        "predicted_margin",
        "moneyline_pick",
        "away_win_probability",
        "home_win_probability",
        "away_moneyline",
        "home_moneyline",
        "home_no_vig_implied_probability",
        "home_moneyline_probability_edge",
        "spread_pick",
        "spread_pick_line",
        "away_cover_probability",
        "home_cover_probability",
        "spread_line",
        "away_spread_odds",
        "home_spread_odds",
        "total_pick",
        "under_probability",
        "over_probability",
        "total_line",
        "under_odds",
        "over_odds",
    ]

    predictions = games.select(output_columns).sort(
        ["week", "gameday", "gametime", "game_id"]
    )

    output_path = PROCESSED_DATA_DIR / "predictions_2026.parquet"
    predictions.write_parquet(output_path)
    predictions.write_csv(output_path.with_suffix(".csv"))

    available_lines = predictions.filter(
        pl.col("spread_line").is_not_null()
        & pl.col("total_line").is_not_null()
    )

    print(f"Saved predictions for {predictions.height} games: {output_path}")
    print(f"Games currently carrying spread and total lines: {available_lines.height}")
    print("\nNext games with all three market predictions:")
    print(
        available_lines.select(
            "week",
            "away_team",
            "home_team",
            "moneyline_pick",
            "spread_pick",
            "spread_pick_line",
            "total_pick",
            "total_line",
        ).head(16)
    )


if __name__ == "__main__":
    main()
