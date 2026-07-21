"""Streamlit dashboard for NFL moneyline, spread, and total predictions."""

from pathlib import Path

import polars as pl
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "predictions_2026.parquet"
)


st.set_page_config(
    page_title="Fourth Down | NFL Predictor",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600;700&family=Fira+Sans:wght@400;500;600;700&display=swap');

    :root {
        --primary: #1E40AF;
        --secondary: #3B82F6;
        --accent: #D97706;
        --background: #F8FAFC;
        --foreground: #172554;
        --muted: #E9EEF6;
        --border: #DBEAFE;
    }

    html, body, [class*="st-"] {
        font-family: "Fira Sans", sans-serif;
    }

    h1, h2, h3, [data-testid="stMetricValue"] {
        font-family: "Fira Code", monospace;
        color: var(--foreground);
    }

    [data-testid="stAppViewContainer"] {
        background: var(--background);
    }

    [data-testid="stSidebar"] {
        background: #EFF6FF;
        border-right: 1px solid var(--border);
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    .market-label {
        color: var(--primary);
        font-family: "Fira Code", monospace;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .context-note {
        background: white;
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        color: #334155;
        padding: 12px 16px;
        margin: 12px 0 20px;
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_predictions() -> pl.DataFrame:
    """Load generated predictions from disk."""

    return pl.read_parquet(PREDICTIONS_PATH)


def probability(value: float | None) -> str:
    """Format a probability for display."""

    return "Unavailable" if value is None else f"{value:.1%}"


def american_odds(value: int | None) -> str:
    """Format American odds with an explicit positive sign."""

    if value is None:
        return "Line not posted"
    return f"{value:+d}"


def point_line(value: float | None) -> str:
    """Format a spread or total line."""

    if value is None:
        return "Line not posted"
    return f"{value:+.1f}"


def selected_probability(game: dict, market: str) -> float | None:
    """Return the probability corresponding to the displayed selection."""

    if market == "moneyline":
        return (
            game["home_win_probability"]
            if game["moneyline_pick"] == game["home_team"]
            else game["away_win_probability"]
        )

    if market == "spread":
        if game["spread_pick"] is None:
            return None
        return (
            game["home_cover_probability"]
            if game["spread_pick"] == game["home_team"]
            else game["away_cover_probability"]
        )

    if game["total_pick"] is None:
        return None
    return (
        game["over_probability"]
        if game["total_pick"] == "OVER"
        else game["under_probability"]
    )


def main() -> None:
    """Render the prediction dashboard."""

    st.title("Fourth Down")
    st.caption("NFL matchup probabilities for moneyline, spread, and totals")

    if not PREDICTIONS_PATH.exists():
        st.error(
            "Prediction data has not been generated. Run `python -m src.predict` "
            "from the project directory."
        )
        st.stop()

    predictions = load_predictions()
    available_weeks = predictions.get_column("week").unique().sort().to_list()

    with st.sidebar:
        st.header("Game filters")
        selected_week = st.selectbox(
            "Week",
            available_weeks,
            index=0,
        )
        only_posted = st.checkbox(
            "Only games with posted lines",
            value=False,
            help="Later-season spreads and totals may not be available yet.",
        )

    week_games = predictions.filter(pl.col("week") == selected_week)

    if only_posted:
        week_games = week_games.filter(
            pl.col("spread_line").is_not_null()
            & pl.col("total_line").is_not_null()
        )

    if week_games.is_empty():
        st.warning("No games match the current filters.")
        st.stop()

    game_options = {
        (
            f"{row['away_team']} at {row['home_team']} | "
            f"{row['gameday']} {row['gametime']} ET"
        ): row["game_id"]
        for row in week_games.iter_rows(named=True)
    }

    selected_label = st.selectbox("Matchup", list(game_options.keys()))
    selected_game_id = game_options[selected_label]
    game = (
        week_games
        .filter(pl.col("game_id") == selected_game_id)
        .to_dicts()[0]
    )

    st.subheader(f"{game['away_team']} at {game['home_team']}")
    st.caption(
        f"Expected QBs: {game['away_expected_qb_name']} vs. "
        f"{game['home_expected_qb_name']}"
    )

    competitions = []
    if game["away_rookie_qb_challenger_name"]:
        competitions.append(
            f"{game['away_team']}: {game['away_rookie_qb_challenger_name']}"
        )
    if game["home_rookie_qb_challenger_name"]:
        competitions.append(
            f"{game['home_team']}: {game['home_rookie_qb_challenger_name']}"
        )

    if competitions:
        st.markdown(
            '<div class="context-note"><strong>QB competition tracked:</strong> '
            + "; ".join(competitions)
            + ". Predictions currently use the listed depth-chart QB1.</div>",
            unsafe_allow_html=True,
        )

    away_score_column, total_score_column, home_score_column = st.columns(3)
    with away_score_column:
        st.metric(
            f"Projected {game['away_team']} points",
            f"{game['predicted_away_score']:.1f}",
        )
    with total_score_column:
        st.metric("Projected game total", f"{game['predicted_total']:.1f}")
    with home_score_column:
        st.metric(
            f"Projected {game['home_team']} points",
            f"{game['predicted_home_score']:.1f}",
        )

    moneyline_column, spread_column, total_column = st.columns(3)

    with moneyline_column:
        st.markdown('<div class="market-label">Moneyline</div>', unsafe_allow_html=True)
        moneyline_probability = selected_probability(game, "moneyline")
        selected_odds = (
            game["home_moneyline"]
            if game["moneyline_pick"] == game["home_team"]
            else game["away_moneyline"]
        )
        st.metric("Predicted winner", game["moneyline_pick"])
        st.metric("Win probability", probability(moneyline_probability))
        st.caption(f"Posted odds: {american_odds(selected_odds)}")

    with spread_column:
        st.markdown('<div class="market-label">Against the spread</div>', unsafe_allow_html=True)
        spread_probability = selected_probability(game, "spread")
        spread_selection = (
            "Line not posted"
            if game["spread_pick"] is None
            else f"{game['spread_pick']} {point_line(game['spread_pick_line'])}"
        )
        st.metric("Predicted cover", spread_selection)
        st.metric("Cover probability", probability(spread_probability))
        st.caption(
            "Stored line is from the home-team perspective; the displayed pick "
            "uses standard betting notation."
        )

    with total_column:
        st.markdown('<div class="market-label">Game total</div>', unsafe_allow_html=True)
        total_probability = selected_probability(game, "total")
        total_selection = (
            "Line not posted"
            if game["total_pick"] is None
            else f"{game['total_pick']} {game['total_line']:.1f}"
        )
        st.metric("Predicted total side", total_selection)
        st.metric("Probability", probability(total_probability))
        st.caption(
            f"Over {american_odds(game['over_odds'])} | "
            f"Under {american_odds(game['under_odds'])}"
        )

    st.divider()
    st.subheader(f"Week {selected_week} board")

    board = week_games.with_columns(
        pl.concat_str(
            [pl.col("away_team"), pl.lit(" at "), pl.col("home_team")]
        ).alias("Matchup"),
        (pl.col("home_win_probability") * 100).round(1).alias("Home win %"),
        (pl.col("home_cover_probability") * 100).round(1).alias("Home cover %"),
        (pl.col("over_probability") * 100).round(1).alias("Over %"),
        pl.col("predicted_total").round(1).alias("Projected total"),
    ).select(
        "Matchup",
        pl.col("moneyline_pick").alias("Winner"),
        "Home win %",
        pl.col("spread_pick").alias("ATS pick"),
        pl.col("spread_pick_line").alias("ATS line"),
        "Home cover %",
        pl.col("total_pick").alias("Total pick"),
        "total_line",
        "Projected total",
        "Over %",
    )

    st.dataframe(
        board.to_pandas(),
        hide_index=True,
        width="stretch",
    )

    st.info(
        "Experimental analytics only. Spread and total backtests are close to "
        "coin-flip performance, so these outputs should not be treated as "
        "guaranteed bets or financial advice."
    )


if __name__ == "__main__":
    main()
