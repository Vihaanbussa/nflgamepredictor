# NFL Game Predictor

A machine-learning project that predicts upcoming NFL games using historical
team and player statistics, injuries, and pregame news sentiment.

React site: <https://vihaanbussa.github.io/nflgamepredictor/>

GitHub Pages hosts the static React frontend only. The prediction form needs
the FastAPI service in `api.py` to be deployed separately, with its public URL
provided as `VITE_API_BASE_URL` when the frontend is built.

## Project layout

- `data/raw/`: downloaded source data (not committed)
- `data/processed/`: model-ready feature tables (not committed)
- `models/`: trained model artifacts (not committed)
- `notebooks/`: exploration and experiments
- `src/`: data collection, feature engineering, training, and prediction code
- `tests/`: automated tests
- `frontend/`: React custom-lines website
- `api.py`: FastAPI prediction and live-refresh API
- `app.py`: original Streamlit dashboard

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app with:

```bash
streamlit run app.py
```

## React custom-lines website

The easiest way to start both React and the prediction API is:

```bash
source .venv/bin/activate
python start_react_app.py
```

Then open `http://127.0.0.1:5173`. Press Control+C in the terminal to stop both
servers.

If you prefer separate terminals, start the API first:

```bash
uvicorn api:app --reload --port 8000
```

Then run `cd frontend && npm run dev` in the second terminal.

By default the API checks for new 2026 data every
12 hours. It downloads the current schedule, completed weekly statistics,
play-by-play, injuries, depth charts, and rosters, then rebuilds features for
only the games that remain unplayed. Set `NFL_AUTO_REFRESH=0` before starting
the API to work entirely from cached local data.

## Generate 2026 predictions

After refreshing the source data, build and train all three market models:

```bash
python -m src.build_2026_roster
python -m src.build_expected_starters
python -m src.build_features
python -m src.train
python -m src.train_score_models
python -m src.build_upcoming_features
python -m src.predict
```

The output at `data/processed/predictions_2026.csv` includes straight-up
winner probabilities, moneyline odds and no-vig comparison, against-the-spread
picks, and over/under picks. Spread and total predictions remain blank when a
book line has not been posted.

## Suggested implementation order

1. Download schedules and weekly statistics in `src/collect_nfl_data.py`.
2. Create leakage-safe rolling features in `src/build_features.py`, including
   play-by-play EPA and success rate, early-down efficiency, neutral pass rate,
   red-zone touchdowns, scoring drives, pace, explosive plays, scoring form,
   and offense-versus-defense matchup features.
3. Train several regularized linear and gradient-boosted candidates in
   `src/train.py`. The script uses 2022, 2023, and 2024 as chronological
   walk-forward validation seasons, compares all-history and recent-era
   training windows, then evaluates the selected model once on 2025.
4. Train separate home-score, away-score, and total-residual models in
   `src/train_score_models.py`. Their estimated score distributions produce
   the moneyline and over/under probabilities used by the dashboard.
5. Add news collection and sentiment in `src/collect_news.py`.
6. Generate upcoming-game predictions in `src/predict.py`.
7. Display custom-line predictions in the React site under `frontend/`.
