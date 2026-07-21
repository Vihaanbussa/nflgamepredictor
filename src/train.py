"""Train moneyline, spread, and over/under probability models."""

from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_data.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

MONEYLINE_FEATURE_COLUMNS = [
    "recent_passing_yards_diff",
    "recent_rushing_yards_diff",
    "recent_passing_epa_per_play_diff",
    "recent_rushing_epa_per_carry_diff",
    "recent_offensive_yards_per_play_diff",
    "recent_passing_interceptions_diff",
    "recent_fumbles_lost_total_diff",
    "recent_turnover_margin_diff",
    "recent_sacks_suffered_diff",
    "recent_sack_rate_allowed_diff",
    "recent_def_sacks_diff",
    "recent_def_interceptions_diff",
    "recent_passing_yards_allowed_diff",
    "recent_rushing_yards_allowed_diff",
    "recent_passing_epa_allowed_per_play_diff",
    "recent_rushing_epa_allowed_per_carry_diff",
    "recent_def_qb_hit_rate_diff",
    "recent_plays_per_game_diff",
    "recent_pass_rate_diff",
    "recent_explosive_play_rate_diff",
    "recent_touchdown_rate_diff",
    "recent_explosive_play_rate_allowed_diff",
    "recent_touchdown_rate_allowed_diff",
    "recent_penalty_yards_per_play_diff",
    "recent_points_for_diff",
    "recent_points_against_diff",
    "recent_point_differential_diff",
    "recent_win_rate_diff",
    "recent_qb_passing_epa_per_play_diff",
    "recent_qb_yards_per_attempt_diff",
    "recent_qb_cpoe_diff",
    "recent_qb_interception_rate_diff",
    "injury_burden_diff",
    "out_count_diff",
    "questionable_count_diff",
    "qb_injury_burden_diff",
    "draft_pick_count_diff",
    "top_100_pick_count_diff",
    "first_round_pick_count_diff",
    "total_draft_capital_diff",
    "offensive_draft_capital_diff",
    "defensive_draft_capital_diff",
    "rookie_qb_first_round_diff",
    "rookie_qb_top_10_diff",
    "elo_diff",
    "rest_diff",
    "neutral_site",
    "home_pass_matchup",
    "away_pass_matchup",
    "home_rush_matchup",
    "away_rush_matchup",
    "expected_margin",
]

SPREAD_FEATURE_COLUMNS = [
    *MONEYLINE_FEATURE_COLUMNS,
    "spread_line",
    "margin_vs_spread",
]

TOTAL_FEATURE_COLUMNS = [
    "recent_passing_yards_sum",
    "recent_rushing_yards_sum",
    "recent_passing_epa_per_play_sum",
    "recent_rushing_epa_per_carry_sum",
    "recent_offensive_yards_per_play_sum",
    "recent_turnover_margin_sum",
    "recent_sack_rate_allowed_sum",
    "recent_passing_yards_allowed_sum",
    "recent_rushing_yards_allowed_sum",
    "recent_passing_epa_allowed_per_play_sum",
    "recent_rushing_epa_allowed_per_carry_sum",
    "recent_def_qb_hit_rate_sum",
    "recent_plays_per_game_sum",
    "recent_pass_rate_sum",
    "recent_explosive_play_rate_sum",
    "recent_touchdown_rate_sum",
    "recent_explosive_play_rate_allowed_sum",
    "recent_touchdown_rate_allowed_sum",
    "recent_penalty_yards_per_play_sum",
    "recent_points_for_sum",
    "recent_points_against_sum",
    "recent_win_rate_sum",
    "recent_qb_passing_epa_per_play_sum",
    "recent_qb_yards_per_attempt_sum",
    "recent_qb_cpoe_sum",
    "recent_qb_interception_rate_sum",
    "injury_burden_sum",
    "out_count_sum",
    "questionable_count_sum",
    "qb_injury_burden_sum",
    "total_line",
    "is_outdoors",
    "temp",
    "wind",
    "home_pass_matchup",
    "away_pass_matchup",
    "home_rush_matchup",
    "away_rush_matchup",
    "expected_total",
    "total_vs_line",
]

# Backward-compatible name used by earlier project code.
FEATURE_COLUMNS = MONEYLINE_FEATURE_COLUMNS


def create_model(c_value: float = 0.01) -> Pipeline:
    """Create a regularized logistic probability model."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, C=c_value),
            ),
        ]
    )


def candidate_models() -> dict[str, tuple[Pipeline, int | None]]:
    """Return conservative candidate models for time-based validation."""

    return {
        "logistic_C_0.001_all": (create_model(0.001), None),
        "logistic_C_0.01_all": (create_model(0.01), None),
        "logistic_C_0.1_all": (create_model(0.1), None),
        "logistic_C_0.001_recent3": (create_model(0.001), 3),
        "logistic_C_0.01_recent3": (create_model(0.01), 3),
        "logistic_C_0.001_recent5": (create_model(0.001), 5),
        "hist_gradient_small_all": (Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=150,
                        max_leaf_nodes=7,
                        min_samples_leaf=35,
                        l2_regularization=10.0,
                        early_stopping=False,
                        random_state=42,
                    ),
                ),
            ]
        ), None),
        "hist_gradient_small_recent5": (Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.04,
                        max_iter=150,
                        max_leaf_nodes=7,
                        min_samples_leaf=35,
                        l2_regularization=10.0,
                        early_stopping=False,
                        random_state=42,
                    ),
                ),
            ]
        ), 5),
        "hist_gradient_medium_all": (Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.03,
                        max_iter=175,
                        max_leaf_nodes=15,
                        min_samples_leaf=40,
                        l2_regularization=15.0,
                        early_stopping=False,
                        random_state=42,
                    ),
                ),
            ]
        ), None),
    }


def select_model_walk_forward(
    data: pl.DataFrame,
    target_column: str,
    feature_columns: list[str],
) -> tuple[str, Pipeline, int | None]:
    """Choose a model using 2022, 2023, and 2024 as unseen seasons."""

    validation_seasons = [2022, 2023, 2024]
    results: list[tuple[float, float, str, Pipeline, int | None]] = []

    print("\nWalk-forward model selection (lower log loss is better)")
    print("-" * 72)

    for model_name, (candidate, training_window) in candidate_models().items():
        fold_log_losses = []
        fold_accuracies = []

        for validation_season in validation_seasons:
            training_filter = pl.col("season") < validation_season
            if training_window is not None:
                training_filter &= (
                    pl.col("season") >= validation_season - training_window
                )

            training = prepare_split(
                data,
                target_column,
                training_filter,
            )
            validation = prepare_split(
                data,
                target_column,
                pl.col("season") == validation_season,
            )
            x_train = training.select(feature_columns).to_numpy()
            y_train = training.get_column(target_column).to_numpy()
            x_validation = validation.select(feature_columns).to_numpy()
            y_validation = validation.get_column(target_column).to_numpy()

            fold_model = clone(candidate)
            fold_model.fit(x_train, y_train)
            probabilities = fold_model.predict_proba(x_validation)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            fold_log_losses.append(log_loss(y_validation, probabilities))
            fold_accuracies.append(accuracy_score(y_validation, predictions))

        mean_log_loss = float(np.mean(fold_log_losses))
        mean_accuracy = float(np.mean(fold_accuracies))
        results.append(
            (
                mean_log_loss,
                -mean_accuracy,
                model_name,
                candidate,
                training_window,
            )
        )
        fold_text = ", ".join(
            f"{season}: {loss:.4f}"
            for season, loss in zip(validation_seasons, fold_log_losses)
        )
        print(
            f"{model_name:<24} mean={mean_log_loss:.4f} "
            f"acc={mean_accuracy:.4f} | {fold_text}"
        )

    _, _, best_name, best_model, best_window = min(
        results,
        key=lambda result: result[:2],
    )
    print(f"Selected: {best_name}")
    return best_name, best_model, best_window


def evaluate_model(
    model: Pipeline,
    features: np.ndarray,
    targets: np.ndarray,
    dataset_name: str,
) -> None:
    """Print classification and probability metrics."""

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n{dataset_name}")
    print("-" * 40)
    print("Accuracy:", round(accuracy_score(targets, predictions), 4))
    print("Log loss:", round(log_loss(targets, probabilities), 4))
    print("Brier score:", round(brier_score_loss(targets, probabilities), 4))
    print("ROC-AUC:", round(roc_auc_score(targets, probabilities), 4))


def prepare_split(
    data: pl.DataFrame,
    target_column: str,
    season_filter: pl.Expr,
) -> pl.DataFrame:
    """Select a season split and remove unavailable lines and pushes."""

    return data.filter(
        season_filter
        & pl.col(target_column).is_not_null()
    )


def train_market_model(
    data: pl.DataFrame,
    market_name: str,
    target_column: str,
    feature_columns: list[str],
    model_filename: str,
) -> Pipeline:
    """Train, evaluate, and save one betting-market model."""

    test_data = prepare_split(
        data,
        target_column,
        pl.col("season") == 2025,
    )

    x_test = test_data.select(feature_columns).to_numpy()
    y_test = test_data.get_column(target_column).to_numpy()

    print(f"\n{'=' * 60}")
    print(f"{market_name.upper()} MODEL")
    print(f"{'=' * 60}")
    print(f"Test games: {len(y_test):,}")

    selected_name, selected_model, selected_window = select_model_walk_forward(
        data,
        target_column,
        feature_columns,
    )
    training_filter = pl.col("season") <= 2024
    if selected_window is not None:
        training_filter &= pl.col("season") >= 2025 - selected_window
    training_data = prepare_split(data, target_column, training_filter)
    x_train = training_data.select(feature_columns).to_numpy()
    y_train = training_data.get_column(target_column).to_numpy()
    print(f"Final training games: {len(y_train):,}")
    final_model = clone(selected_model)
    final_model.fit(x_train, y_train)
    evaluate_model(
        final_model,
        x_test,
        y_test,
        f"2025 {market_name} evaluation",
    )

    model_path = MODELS_DIR / model_filename
    joblib.dump(
        {
            "market": market_name,
            "target_column": target_column,
            "selected_model": selected_name,
            "training_window_seasons": selected_window,
            "model": final_model,
            "feature_columns": feature_columns,
        },
        model_path,
    )
    print(f"Saved model to: {model_path}")

    return final_model


def main() -> None:
    """Train all three market models."""

    print("Loading model data...")
    data = pl.read_parquet(MODEL_DATA_PATH)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    moneyline_model = train_market_model(
        data=data,
        market_name="moneyline",
        target_column="home_win",
        feature_columns=MONEYLINE_FEATURE_COLUMNS,
        model_filename="moneyline_model.joblib",
    )

    # Preserve the original artifact name for the Streamlit starter app.
    joblib.dump(
        {
            "market": "moneyline",
            "target_column": "home_win",
            "model": moneyline_model,
            "feature_columns": MONEYLINE_FEATURE_COLUMNS,
        },
        MODELS_DIR / "logistic_regression.joblib",
    )

    train_market_model(
        data=data,
        market_name="spread",
        target_column="home_cover",
        feature_columns=SPREAD_FEATURE_COLUMNS,
        model_filename="spread_model.joblib",
    )

    train_market_model(
        data=data,
        market_name="over/under",
        target_column="over_hit",
        feature_columns=TOTAL_FEATURE_COLUMNS,
        model_filename="total_model.joblib",
    )


if __name__ == "__main__":
    main()
