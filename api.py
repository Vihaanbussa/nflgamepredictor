"""FastAPI backend for the React custom-line predictor."""

from pathlib import Path

import polars as pl
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.custom_prediction import evaluate_game
from src.refresh_live_data import is_stale, read_status, refresh_if_stale, refresh_now


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "upcoming_features_2026.parquet"

app = FastAPI(title="NFL Custom Line Predictor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    game_id: str
    home_spread: float
    total_line: float = Field(gt=0)
    home_moneyline: float
    away_moneyline: float
    home_spread_odds: float
    away_spread_odds: float
    over_odds: float
    under_odds: float

    @field_validator(
        "home_moneyline", "away_moneyline", "home_spread_odds",
        "away_spread_odds", "over_odds", "under_odds",
    )
    @classmethod
    def odds_cannot_be_zero(cls, value: float) -> float:
        if value == 0:
            raise ValueError("American odds cannot be zero")
        return value


def load_games() -> pl.DataFrame:
    if not FEATURES_PATH.exists():
        raise HTTPException(503, "Upcoming features have not been built yet.")
    return pl.read_parquet(FEATURES_PATH).sort(
        ["week", "gameday", "gametime", "game_id"]
    )


@app.get("/api/games")
def games(background_tasks: BackgroundTasks) -> dict:
    refresh_status = read_status()
    if refresh_status["state"] != "disabled" and is_stale():
        background_tasks.add_task(refresh_if_stale)
        refresh_status = {
            **refresh_status,
            "state": "refreshing",
            "message": "Checking for new completed 2026 games.",
        }
    rows = load_games().select(
        "game_id", "week", "gameday", "gametime", "away_team", "home_team",
        "away_expected_qb_name", "home_expected_qb_name", "spread_line",
        "home_spread_odds", "away_spread_odds", "total_line", "over_odds",
        "under_odds", "home_moneyline", "away_moneyline",
    ).to_dicts()
    for row in rows:
        row["gameday"] = str(row["gameday"])
        row["home_spread"] = (
            -row["spread_line"] if row["spread_line"] is not None else 0.0
        )
    return {"games": rows, "refresh": refresh_status}


@app.get("/api/refresh-status")
def refresh_status() -> dict:
    return read_status()


@app.post("/api/refresh")
def refresh(background_tasks: BackgroundTasks) -> dict:
    background_tasks.add_task(refresh_now)
    return {"state": "refreshing"}


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    games = load_games()
    game = games.filter(pl.col("game_id") == request.game_id)
    if game.height != 1:
        raise HTTPException(404, "That upcoming game was not found.")
    values = request.model_dump()
    values.pop("game_id")
    return evaluate_game(game, values)
