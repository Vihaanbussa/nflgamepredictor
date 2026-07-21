"""Reusable custom-line prediction logic for web clients."""

from pathlib import Path

import joblib
import numpy as np
import polars as pl

from src.predict import predict_probability, predict_score_model
from src.train_score_models import probability_above


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def american_profit(odds: float) -> float:
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def expected_value(probability: float, odds: float) -> float:
    return probability * american_profit(odds) - (1.0 - probability)


def no_vig_home_probability(home_odds: float, away_odds: float) -> float:
    def implied(odds: float) -> float:
        return (
            abs(odds) / (abs(odds) + 100.0)
            if odds < 0
            else 100.0 / (odds + 100.0)
        )

    home = implied(home_odds)
    away = implied(away_odds)
    return home / (home + away)


def update_custom_lines(game: pl.DataFrame, **inputs: float) -> pl.DataFrame:
    """Replace market inputs and update their dependent features."""

    stored_spread = -inputs["home_spread"]
    total_line = inputs["total_line"]
    return game.with_columns(
        pl.lit(stored_spread).alias("spread_line"),
        pl.lit(total_line).alias("total_line"),
        pl.lit(inputs["home_moneyline"]).alias("home_moneyline"),
        pl.lit(inputs["away_moneyline"]).alias("away_moneyline"),
        pl.lit(inputs["home_spread_odds"]).alias("home_spread_odds"),
        pl.lit(inputs["away_spread_odds"]).alias("away_spread_odds"),
        pl.lit(inputs["over_odds"]).alias("over_odds"),
        pl.lit(inputs["under_odds"]).alias("under_odds"),
        (pl.col("expected_margin") - stored_spread).alias("margin_vs_spread"),
        (pl.col("expected_total") - total_line).alias("total_vs_line"),
    )


def evaluate_game(game: pl.DataFrame, inputs: dict[str, float]) -> dict:
    """Run moneyline, spread, and total models on one custom line set."""

    custom = update_custom_lines(game, **inputs)
    home_team = custom.item(0, "home_team")
    away_team = custom.item(0, "away_team")
    predicted_home, _ = predict_score_model(custom, "home_score_model.joblib")
    predicted_away, _ = predict_score_model(custom, "away_score_model.joblib")
    predicted_residual, residual_artifact = predict_score_model(
        custom, "total_residual_model.joblib", clip_nonnegative=False
    )

    covariance = joblib.load(
        MODELS_DIR / "score_simulation.joblib"
    )["residual_covariance"]
    margin_std = float(
        np.sqrt(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    )
    home_probability = float(probability_above(
        predicted_home - predicted_away, np.array([0.0]), margin_std
    )[0])
    home_cover_probability = float(
        predict_probability(custom, "spread_model.joblib")[0]
    )
    over_probability = float(probability_above(
        predicted_residual,
        np.array([0.0]),
        float(residual_artifact["residual_std"]),
    )[0])

    home_ml_ev = expected_value(home_probability, inputs["home_moneyline"])
    away_ml_ev = expected_value(1.0 - home_probability, inputs["away_moneyline"])
    home_spread_ev = expected_value(
        home_cover_probability, inputs["home_spread_odds"]
    )
    away_spread_ev = expected_value(
        1.0 - home_cover_probability, inputs["away_spread_odds"]
    )
    over_ev = expected_value(over_probability, inputs["over_odds"])
    under_ev = expected_value(1.0 - over_probability, inputs["under_odds"])

    return {
        "game_id": custom.item(0, "game_id"),
        "home_team": home_team,
        "away_team": away_team,
        "predicted_home": float(predicted_home[0]),
        "predicted_away": float(predicted_away[0]),
        "predicted_total": inputs["total_line"] + float(predicted_residual[0]),
        "moneyline_pick": home_team if home_probability >= 0.5 else away_team,
        "moneyline_probability": max(home_probability, 1.0 - home_probability),
        "moneyline_value": home_team if home_ml_ev >= away_ml_ev else away_team,
        "moneyline_best_ev": max(home_ml_ev, away_ml_ev),
        "market_home_probability": no_vig_home_probability(
            inputs["home_moneyline"], inputs["away_moneyline"]
        ),
        "spread_pick": home_team if home_cover_probability >= 0.5 else away_team,
        "spread_probability": max(home_cover_probability, 1.0 - home_cover_probability),
        "spread_value": home_team if home_spread_ev >= away_spread_ev else away_team,
        "spread_best_ev": max(home_spread_ev, away_spread_ev),
        "total_pick": "OVER" if over_probability >= 0.5 else "UNDER",
        "total_probability": max(over_probability, 1.0 - over_probability),
        "total_value": "OVER" if over_ev >= under_ev else "UNDER",
        "total_best_ev": max(over_ev, under_ev),
    }
