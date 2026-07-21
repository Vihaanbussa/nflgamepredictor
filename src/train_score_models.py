"""Train home- and away-score models for moneyline and total simulations."""

from math import erf, sqrt
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_data.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
VALIDATION_SEASONS = [2022, 2023, 2024]

NON_FEATURE_COLUMNS = {
    "game_id", "season", "week", "gameday", "away_team", "home_team",
    "home_win", "home_cover", "over_hit", "home_score", "away_score",
    "home_margin", "game_total", "away_moneyline", "home_moneyline",
    "total_residual",
    "away_spread_odds", "home_spread_odds", "under_odds", "over_odds",
}


def score_feature_columns(data: pl.DataFrame) -> list[str]:
    """Select pregame numeric columns while excluding outcomes and identifiers."""

    return [
        column
        for column, dtype in data.schema.items()
        if column not in NON_FEATURE_COLUMNS and dtype.is_numeric()
    ]


def candidates() -> dict[str, tuple[Pipeline, int | None]]:
    """Models and optional number of recent seasons to retain."""

    def ridge(alpha: float) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha)),
        ])

    def gradient(leaves: int) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", HistGradientBoostingRegressor(
                learning_rate=0.035,
                max_iter=175,
                max_leaf_nodes=leaves,
                min_samples_leaf=40,
                l2_regularization=15.0,
                early_stopping=False,
                random_state=42,
            )),
        ])

    return {
        "ridge_10_all": (ridge(10.0), None),
        "ridge_50_all": (ridge(50.0), None),
        "ridge_100_all": (ridge(100.0), None),
        "ridge_50_recent3": (ridge(50.0), 3),
        "ridge_50_recent5": (ridge(50.0), 5),
        "gradient_7_all": (gradient(7), None),
        "gradient_15_all": (gradient(15), None),
        "gradient_7_recent5": (gradient(7), 5),
    }


def season_training_filter(validation_season: int, window: int | None) -> pl.Expr:
    condition = pl.col("season") < validation_season
    if window is not None:
        condition &= pl.col("season") >= validation_season - window
    return condition


def select_score_model(
    data: pl.DataFrame,
    features: list[str],
    target: str,
) -> tuple[str, Pipeline, int | None]:
    """Select a regressor using chronological 2022-2024 validation."""

    ranked = []
    print(f"\nWalk-forward selection for {target}")
    print("-" * 78)

    for name, (candidate, window) in candidates().items():
        fold_mae = []
        fold_rmse = []
        for season in VALIDATION_SEASONS:
            train = data.filter(season_training_filter(season, window))
            validation = data.filter(pl.col("season") == season)
            model = clone(candidate)
            model.fit(
                train.select(features).to_numpy(),
                train.get_column(target).to_numpy(),
            )
            prediction = model.predict(validation.select(features).to_numpy())
            actual = validation.get_column(target).to_numpy()
            fold_mae.append(mean_absolute_error(actual, prediction))
            fold_rmse.append(mean_squared_error(actual, prediction) ** 0.5)

        mean_mae = float(np.mean(fold_mae))
        mean_rmse = float(np.mean(fold_rmse))
        ranked.append((mean_mae, mean_rmse, name, candidate, window))
        fold_text = ", ".join(
            f"{season}: {mae:.2f}"
            for season, mae in zip(VALIDATION_SEASONS, fold_mae)
        )
        print(f"{name:<22} MAE={mean_mae:.3f} RMSE={mean_rmse:.3f} | {fold_text}")

    _, _, name, model, window = min(ranked, key=lambda row: row[:2])
    print(f"Selected: {name}")
    return name, model, window


def train_target(
    data: pl.DataFrame,
    features: list[str],
    target: str,
) -> tuple[dict, np.ndarray, pl.DataFrame]:
    """Fit one score model and return its 2025 and out-of-fold predictions."""

    name, candidate, window = select_score_model(data, features, target)
    oof_rows = []
    for season in VALIDATION_SEASONS:
        train = data.filter(season_training_filter(season, window))
        validation = data.filter(pl.col("season") == season)
        model = clone(candidate).fit(
            train.select(features).to_numpy(),
            train.get_column(target).to_numpy(),
        )
        prediction = model.predict(validation.select(features).to_numpy())
        oof_rows.append(validation.select("game_id", target).with_columns(
            pl.Series(f"predicted_{target}", prediction)
        ))

    final_filter = pl.col("season") <= 2024
    if window is not None:
        final_filter &= pl.col("season") >= 2025 - window
    final_train = data.filter(final_filter)
    final_model = clone(candidate).fit(
        final_train.select(features).to_numpy(),
        final_train.get_column(target).to_numpy(),
    )
    test = data.filter(pl.col("season") == 2025)
    test_prediction = final_model.predict(test.select(features).to_numpy())

    artifact = {
        "target": target,
        "selected_model": name,
        "training_window_seasons": window,
        "feature_columns": features,
        "model": final_model,
    }
    return artifact, test_prediction, pl.concat(oof_rows)


def probability_above(mean: np.ndarray, threshold: np.ndarray, std: float) -> np.ndarray:
    """Return normal-distribution probabilities without requiring SciPy."""

    z = (threshold - mean) / max(std, 1e-6)
    cdf = np.array([0.5 * (1.0 + erf(value / sqrt(2.0))) for value in z])
    return np.clip(1.0 - cdf, 0.01, 0.99)


def main() -> None:
    data = pl.read_parquet(MODEL_DATA_PATH).with_columns(
        (pl.col("game_total") - pl.col("total_line")).alias("total_residual")
    )
    features = score_feature_columns(data)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training score models with {len(features)} pregame features.")

    home_artifact, home_test, home_oof = train_target(
        data, features, "home_score"
    )
    away_artifact, away_test, away_oof = train_target(
        data, features, "away_score"
    )

    residuals = home_oof.join(away_oof, on="game_id").with_columns(
        (pl.col("home_score") - pl.col("predicted_home_score")).alias("home_error"),
        (pl.col("away_score") - pl.col("predicted_away_score")).alias("away_error"),
    )
    covariance = np.cov(
        residuals.select("home_error", "away_error").to_numpy(),
        rowvar=False,
    )

    joblib.dump(home_artifact, MODELS_DIR / "home_score_model.joblib")
    joblib.dump(away_artifact, MODELS_DIR / "away_score_model.joblib")
    joblib.dump(
        {"residual_covariance": covariance, "validation_seasons": VALIDATION_SEASONS},
        MODELS_DIR / "score_simulation.joblib",
    )

    total_line_data = data.filter(pl.col("total_line").is_not_null())
    total_artifact, total_residual_test, total_oof = train_target(
        total_line_data,
        features,
        "total_residual",
    )
    total_oof = total_oof.with_columns(
        (
            pl.col("total_residual")
            - pl.col("predicted_total_residual")
        ).alias("residual_error")
    )
    total_residual_std = float(
        total_oof.get_column("residual_error").std()
    )
    total_artifact["residual_std"] = total_residual_std
    joblib.dump(total_artifact, MODELS_DIR / "total_residual_model.joblib")

    test = data.filter(pl.col("season") == 2025)
    actual_home = test.get_column("home_score").to_numpy()
    actual_away = test.get_column("away_score").to_numpy()
    predicted_margin = home_test - away_test
    predicted_total = home_test + away_test
    actual_margin = actual_home - actual_away
    actual_total = actual_home + actual_away

    margin_std = sqrt(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    total_std = sqrt(covariance[0, 0] + covariance[1, 1] + 2 * covariance[0, 1])
    home_probability = probability_above(
        predicted_margin, np.zeros(len(test)), margin_std
    )
    total_lines = test.get_column("total_line").to_numpy()
    over_probability = probability_above(predicted_total, total_lines, total_std)
    residual_test_rows = total_line_data.filter(pl.col("season") == 2025)
    residual_actual = residual_test_rows.get_column("total_residual").to_numpy()
    residual_probability = probability_above(
        total_residual_test,
        np.zeros(len(total_residual_test)),
        total_residual_std,
    )

    winner_accuracy = np.mean((predicted_margin > 0) == (actual_margin > 0))
    total_mask = ~np.isnan(total_lines) & (actual_total != total_lines)
    total_accuracy = np.mean(
        (over_probability[total_mask] >= 0.5)
        == (actual_total[total_mask] > total_lines[total_mask])
    )
    confident = total_mask & (np.abs(over_probability - 0.5) >= 0.04)
    confident_accuracy = (
        np.mean(
            (over_probability[confident] >= 0.5)
            == (actual_total[confident] > total_lines[confident])
        )
        if confident.any() else float("nan")
    )
    residual_accuracy = np.mean(
        (residual_probability >= 0.5) == (residual_actual > 0)
    )
    residual_confident = np.abs(residual_probability - 0.5) >= 0.04
    residual_confident_accuracy = (
        np.mean(
            (residual_probability[residual_confident] >= 0.5)
            == (residual_actual[residual_confident] > 0)
        )
        if residual_confident.any() else float("nan")
    )

    print("\n2025 SCORE-MODEL EVALUATION")
    print("-" * 48)
    print(f"Home score MAE: {mean_absolute_error(actual_home, home_test):.3f}")
    print(f"Away score MAE: {mean_absolute_error(actual_away, away_test):.3f}")
    print(f"Moneyline accuracy: {winner_accuracy:.4f}")
    print(f"Over/under accuracy (all posted lines): {total_accuracy:.4f}")
    print(
        f"Over/under accuracy (54%+ confidence): {confident_accuracy:.4f} "
        f"on {confident.sum()} games"
    )
    print(f"Total-residual accuracy (all posted lines): {residual_accuracy:.4f}")
    print(
        f"Total-residual accuracy (54%+ confidence): "
        f"{residual_confident_accuracy:.4f} on {residual_confident.sum()} games"
    )
    print(f"Residual margin SD: {margin_std:.3f}")
    print(f"Residual total SD: {total_std:.3f}")


if __name__ == "__main__":
    main()
