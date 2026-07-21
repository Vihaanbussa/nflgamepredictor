"""Simple form for testing custom betting lines on 2026 NFL games."""

from pathlib import Path

import joblib
import numpy as np
import polars as pl
import streamlit as st

from src.predict import predict_probability, predict_score_model
from src.train_score_models import probability_above


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "upcoming_features_2026.parquet"
MODELS_DIR = PROJECT_ROOT / "models"


def default_number(value: float | int | None, fallback: float) -> float:
    """Use a posted line when available and a sensible fallback otherwise."""

    return fallback if value is None else float(value)


def american_profit(odds: float) -> float:
    """Return profit per $1 risked for American odds."""

    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def expected_value(probability: float, odds: float) -> float:
    """Expected profit per $1 risked."""

    return probability * american_profit(odds) - (1.0 - probability)


def no_vig_home_probability(home_odds: float, away_odds: float) -> float:
    """Remove the sportsbook margin from two opposing moneylines."""

    def implied(odds: float) -> float:
        return (
            abs(odds) / (abs(odds) + 100.0)
            if odds < 0
            else 100.0 / (odds + 100.0)
        )

    home = implied(home_odds)
    away = implied(away_odds)
    return home / (home + away)


def update_custom_lines(
    game: pl.DataFrame,
    home_spread: float,
    total_line: float,
    home_moneyline: float,
    away_moneyline: float,
    home_spread_odds: float,
    away_spread_odds: float,
    over_odds: float,
    under_odds: float,
) -> pl.DataFrame:
    """Replace posted lines and recompute every feature that depends on them."""

    # nflverse stores a positive number when the home team is favored.
    stored_spread = -home_spread
    return game.with_columns(
        pl.lit(stored_spread).alias("spread_line"),
        pl.lit(total_line).alias("total_line"),
        pl.lit(home_moneyline).alias("home_moneyline"),
        pl.lit(away_moneyline).alias("away_moneyline"),
        pl.lit(home_spread_odds).alias("home_spread_odds"),
        pl.lit(away_spread_odds).alias("away_spread_odds"),
        pl.lit(over_odds).alias("over_odds"),
        pl.lit(under_odds).alias("under_odds"),
        (pl.col("expected_margin") - stored_spread).alias("margin_vs_spread"),
        (pl.col("expected_total") - total_line).alias("total_vs_line"),
    )


def evaluate_game(game: pl.DataFrame, inputs: dict[str, float]) -> dict:
    """Run all three market models against the user's custom lines."""

    custom = update_custom_lines(game, **inputs)
    home_team = custom.item(0, "home_team")
    away_team = custom.item(0, "away_team")

    predicted_home, _ = predict_score_model(custom, "home_score_model.joblib")
    predicted_away, _ = predict_score_model(custom, "away_score_model.joblib")
    predicted_residual, residual_artifact = predict_score_model(
        custom,
        "total_residual_model.joblib",
        clip_nonnegative=False,
    )

    covariance = joblib.load(
        MODELS_DIR / "score_simulation.joblib"
    )["residual_covariance"]
    margin_std = float(
        np.sqrt(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    )
    home_probability = float(
        probability_above(
            predicted_home - predicted_away,
            np.array([0.0]),
            margin_std,
        )[0]
    )
    home_cover_probability = float(
        predict_probability(custom, "spread_model.joblib")[0]
    )
    over_probability = float(
        probability_above(
            predicted_residual,
            np.array([0.0]),
            float(residual_artifact["residual_std"]),
        )[0]
    )

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
        "home_team": home_team,
        "away_team": away_team,
        "predicted_home": float(predicted_home[0]),
        "predicted_away": float(predicted_away[0]),
        "predicted_total": inputs["total_line"] + float(predicted_residual[0]),
        "moneyline_pick": home_team if home_probability >= 0.5 else away_team,
        "moneyline_probability": max(home_probability, 1.0 - home_probability),
        "moneyline_value": (
            home_team if home_ml_ev >= away_ml_ev else away_team
        ),
        "moneyline_best_ev": max(home_ml_ev, away_ml_ev),
        "market_home_probability": no_vig_home_probability(
            inputs["home_moneyline"], inputs["away_moneyline"]
        ),
        "spread_pick": home_team if home_cover_probability >= 0.5 else away_team,
        "spread_probability": max(
            home_cover_probability, 1.0 - home_cover_probability
        ),
        "spread_value": home_team if home_spread_ev >= away_spread_ev else away_team,
        "spread_best_ev": max(home_spread_ev, away_spread_ev),
        "total_pick": "OVER" if over_probability >= 0.5 else "UNDER",
        "total_probability": max(over_probability, 1.0 - over_probability),
        "total_value": "OVER" if over_ev >= under_ev else "UNDER",
        "total_best_ev": max(over_ev, under_ev),
    }


def value_text(selection: str, ev: float) -> str:
    """Show whether the model believes the entered odds offer positive value."""

    if ev <= 0:
        return "No positive expected-value side at the entered odds"
    return f"Best value: {selection} ({ev:+.1%} expected return per $1)"


def main() -> None:
    st.set_page_config(page_title="NFL Custom Line Predictor", layout="wide")
    st.title("NFL Custom Line Predictor")
    st.write(
        "Choose one or two scheduled 2026 games, enter your own betting lines, "
        "and run the predictor. Home spreads use normal notation, so -3.5 means "
        "the home team is favored by 3.5 points."
    )

    if not FEATURES_PATH.exists():
        st.error("Run `python -m src.build_upcoming_features` first.")
        st.stop()

    games = pl.read_parquet(FEATURES_PATH).sort(
        ["week", "gameday", "gametime", "game_id"]
    )
    game_labels = {
        (
            f"Week {row['week']}: {row['away_team']} at {row['home_team']} "
            f"({row['gameday']})"
        ): row["game_id"]
        for row in games.iter_rows(named=True)
    }
    selected_labels = st.multiselect(
        "Select up to two games",
        options=list(game_labels),
        default=list(game_labels)[:1],
        max_selections=2,
    )

    if not selected_labels:
        st.info("Select at least one game.")
        st.stop()

    submitted_inputs = []
    with st.form("custom_lines"):
        for index, label in enumerate(selected_labels):
            game_id = game_labels[label]
            row = games.filter(pl.col("game_id") == game_id).to_dicts()[0]
            st.subheader(label)

            moneyline_columns = st.columns(2)
            home_moneyline = moneyline_columns[0].number_input(
                f"{row['home_team']} moneyline",
                value=default_number(row["home_moneyline"], -110),
                step=5.0,
                key=f"home_ml_{game_id}",
            )
            away_moneyline = moneyline_columns[1].number_input(
                f"{row['away_team']} moneyline",
                value=default_number(row["away_moneyline"], -110),
                step=5.0,
                key=f"away_ml_{game_id}",
            )

            line_columns = st.columns(2)
            posted_home_spread = (
                -row["spread_line"] if row["spread_line"] is not None else 0.0
            )
            home_spread = line_columns[0].number_input(
                f"{row['home_team']} spread",
                value=float(posted_home_spread),
                step=0.5,
                key=f"spread_{game_id}",
            )
            total_line = line_columns[1].number_input(
                "Over/under total",
                value=default_number(row["total_line"], 44.5),
                step=0.5,
                min_value=1.0,
                key=f"total_{game_id}",
            )

            odds_columns = st.columns(4)
            home_spread_odds = odds_columns[0].number_input(
                "Home spread odds",
                value=default_number(row["home_spread_odds"], -110),
                step=5.0,
                key=f"home_spread_odds_{game_id}",
            )
            away_spread_odds = odds_columns[1].number_input(
                "Away spread odds",
                value=default_number(row["away_spread_odds"], -110),
                step=5.0,
                key=f"away_spread_odds_{game_id}",
            )
            over_odds = odds_columns[2].number_input(
                "Over odds",
                value=default_number(row["over_odds"], -110),
                step=5.0,
                key=f"over_odds_{game_id}",
            )
            under_odds = odds_columns[3].number_input(
                "Under odds",
                value=default_number(row["under_odds"], -110),
                step=5.0,
                key=f"under_odds_{game_id}",
            )

            submitted_inputs.append(
                (
                    game_id,
                    {
                        "home_spread": home_spread,
                        "total_line": total_line,
                        "home_moneyline": home_moneyline,
                        "away_moneyline": away_moneyline,
                        "home_spread_odds": home_spread_odds,
                        "away_spread_odds": away_spread_odds,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                    },
                )
            )
            st.divider()

        submitted = st.form_submit_button("Run predictions", type="primary")

    if not submitted:
        st.stop()

    if any(
        odds == 0
        for _, values in submitted_inputs
        for name, odds in values.items()
        if "odds" in name or "moneyline" in name
    ):
        st.error("American odds cannot be 0. Enter -110, +120, or another valid value.")
        st.stop()

    st.header("Predictions")
    for game_id, inputs in submitted_inputs:
        game = games.filter(pl.col("game_id") == game_id)
        result = evaluate_game(game, inputs)
        st.subheader(f"{result['away_team']} at {result['home_team']}")
        st.write(
            f"Projected score: {result['away_team']} {result['predicted_away']:.1f}, "
            f"{result['home_team']} {result['predicted_home']:.1f} "
            f"(total {result['predicted_total']:.1f})"
        )

        moneyline, spread, total = st.columns(3)
        moneyline.metric("Moneyline pick", result["moneyline_pick"])
        moneyline.write(f"Win probability: {result['moneyline_probability']:.1%}")
        moneyline.caption(
            value_text(result["moneyline_value"], result["moneyline_best_ev"])
        )

        spread.metric("Spread pick", result["spread_pick"])
        spread.write(f"Cover probability: {result['spread_probability']:.1%}")
        spread.caption(value_text(result["spread_value"], result["spread_best_ev"]))

        total.metric("Total pick", result["total_pick"])
        total.write(f"Probability: {result['total_probability']:.1%}")
        total.caption(value_text(result["total_value"], result["total_best_ev"]))
        st.divider()

    st.warning(
        "Experimental model output only. Historical spread and total performance "
        "is close to 50%, so do not treat these predictions as guaranteed bets."
    )


if __name__ == "__main__":
    main()
