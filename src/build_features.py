from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

ROLLING_WINDOW = 5

RAW_FEATURES = [
    "passing_yards",
    "rushing_yards",
    "passing_epa_per_play",
    "rushing_epa_per_carry",
    "offensive_yards_per_play",
    "passing_interceptions",
    "fumbles_lost_total",
    "turnover_margin",
    "sacks_suffered",
    "sack_rate_allowed",
    "def_sacks",
    "def_interceptions",
    "passing_yards_allowed",
    "rushing_yards_allowed",
    "passing_epa_allowed_per_play",
    "rushing_epa_allowed_per_carry",
    "def_qb_hit_rate",
    "plays_per_game",
    "pass_rate",
    "explosive_play_rate",
    "touchdown_rate",
    "explosive_play_rate_allowed",
    "touchdown_rate_allowed",
    "penalty_yards_per_play",
]

MATCHUP_FEATURES = [
    "home_pass_matchup",
    "away_pass_matchup",
    "home_rush_matchup",
    "away_rush_matchup",
    "expected_home_points",
    "expected_away_points",
    "expected_margin",
    "expected_total",
    "margin_vs_spread",
    "total_vs_line",
]

PBP_RAW_FEATURES = [
    "pbp_epa_per_play",
    "pbp_success_rate",
    "early_down_epa",
    "early_down_success_rate",
    "neutral_pass_rate",
    "red_zone_td_rate",
    "late_down_conversion_rate",
    "completion_rate",
    "no_huddle_rate",
    "scoring_drive_rate",
    "plays_per_drive",
    "pbp_epa_allowed_per_play",
    "pbp_success_rate_allowed",
    "early_down_epa_allowed",
    "red_zone_td_rate_allowed",
    "scoring_drive_rate_allowed",
]

FORM_RAW_FEATURES = [
    "points_for",
    "points_against",
    "point_differential",
    "win_rate",
]

QB_RAW_FEATURES = [
    "qb_passing_epa_per_play",
    "qb_yards_per_attempt",
    "qb_cpoe",
    "qb_interception_rate",
]

INJURY_FEATURES = [
    "injury_burden",
    "out_count",
    "questionable_count",
    "qb_injury_burden",
]

DRAFT_FEATURES = [
    "draft_pick_count",
    "top_100_pick_count",
    "first_round_pick_count",
    "total_draft_capital",
    "offensive_draft_capital",
    "defensive_draft_capital",
    "rookie_qb_first_round",
    "rookie_qb_top_10",
]

DRAFT_TEAM_RENAMES = {
    "CLV": "CLE",
    "GNB": "GB",
    "KAN": "KC",
    "LAR": "LA",
    "LVR": "LV",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
}


def load_data() -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    """Load schedules, statistics, injuries, and draft picks."""

    schedules = pl.read_parquet(
        RAW_DATA_DIR / "schedules.parquet"
    )

    team_stats = pl.read_parquet(
        RAW_DATA_DIR / "team_stats.parquet"
    )

    player_stats = pl.read_parquet(
        RAW_DATA_DIR / "player_stats.parquet"
    )

    injuries = pl.read_parquet(
        RAW_DATA_DIR / "injuries.parquet"
    )

    draft_picks = pl.read_parquet(
        RAW_DATA_DIR / "draft_picks.parquet"
    )

    return schedules, team_stats, player_stats, injuries, draft_picks


def prepare_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Keep completed games and create moneyline, spread, and total targets."""

    games = schedules.filter(
        (pl.col("game_type") == "REG")
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
        & (pl.col("home_score") != pl.col("away_score"))
    )

    games = games.with_columns(
        (pl.col("home_score") > pl.col("away_score"))
        .cast(pl.Int8)
        .alias("home_win"),
        (pl.col("home_score") - pl.col("away_score"))
        .cast(pl.Float64)
        .alias("home_margin"),
        (pl.col("home_score") + pl.col("away_score"))
        .cast(pl.Float64)
        .alias("game_total"),
        (pl.col("location") == "Neutral")
        .cast(pl.Int8)
        .alias("neutral_site"),
        (pl.col("roof") == "outdoors")
        .cast(pl.Int8)
        .alias("is_outdoors"),
    )

    games = games.with_columns(
        pl.when(
            pl.col("spread_line").is_null()
            | (pl.col("home_margin") == pl.col("spread_line"))
        )
        .then(None)
        .otherwise(pl.col("home_margin") > pl.col("spread_line"))
        .cast(pl.Int8)
        .alias("home_cover"),

        pl.when(
            pl.col("total_line").is_null()
            | (pl.col("game_total") == pl.col("total_line"))
        )
        .then(None)
        .otherwise(pl.col("game_total") > pl.col("total_line"))
        .cast(pl.Int8)
        .alias("over_hit"),
    )

    return games


def prepare_team_stats(
    team_stats: pl.DataFrame,
) -> pl.DataFrame:
    """Create efficiency metrics and rolling pregame averages."""

    team_stats = (
        team_stats
        .filter(pl.col("season_type") == "REG")
        .sort(["team", "season", "week"])
    )

    # Each game has two team rows. Subtracting the current team's value
    # from the game total produces the opponent's offensive statistic,
    # which is what the current team's defense allowed.
    team_stats = team_stats.with_columns(
        (
            pl.col("passing_yards").sum().over("game_id")
            - pl.col("passing_yards")
        ).alias("passing_yards_allowed"),
        (
            pl.col("rushing_yards").sum().over("game_id")
            - pl.col("rushing_yards")
        ).alias("rushing_yards_allowed"),
        (
            pl.col("passing_epa").sum().over("game_id")
            - pl.col("passing_epa")
        ).alias("opponent_passing_epa"),
        (
            pl.col("rushing_epa").sum().over("game_id")
            - pl.col("rushing_epa")
        ).alias("opponent_rushing_epa"),
        (
            pl.col("attempts").sum().over("game_id")
            - pl.col("attempts")
        ).alias("opponent_attempts"),
        (
            pl.col("sacks_suffered").sum().over("game_id")
            - pl.col("sacks_suffered")
        ).alias("opponent_sacks_suffered"),
        (
            pl.col("carries").sum().over("game_id")
            - pl.col("carries")
        ).alias("opponent_carries"),
        (
            pl.col("passing_tds").sum().over("game_id")
            - pl.col("passing_tds")
        ).alias("opponent_passing_tds"),
        (
            pl.col("rushing_tds").sum().over("game_id")
            - pl.col("rushing_tds")
        ).alias("opponent_rushing_tds"),
        (
            pl.col("passing_20").sum().over("game_id")
            - pl.col("passing_20")
        ).alias("opponent_passing_20"),
        (
            pl.col("rushing_20").sum().over("game_id")
            - pl.col("rushing_20")
        ).alias("opponent_rushing_20"),
    )

    # Turn total EPA into efficiency per play.
    team_stats = team_stats.with_columns(
        (
            pl.col("passing_epa")
            / (
                pl.col("attempts")
                + pl.col("sacks_suffered")
            )
        ).alias("passing_epa_per_play"),

        (
            pl.col("rushing_epa")
            / pl.col("carries")
        ).alias("rushing_epa_per_carry"),

        (
            (pl.col("passing_yards") + pl.col("rushing_yards"))
            / (
                pl.col("attempts")
                + pl.col("sacks_suffered")
                + pl.col("carries")
            )
        ).alias("offensive_yards_per_play"),

        (
            pl.col("def_interceptions")
            + pl.col("fumble_recovery_opp")
            - pl.col("passing_interceptions")
            - pl.col("fumbles_lost_total")
        ).alias("turnover_margin"),

        (
            pl.col("sacks_suffered")
            / (pl.col("attempts") + pl.col("sacks_suffered"))
        ).alias("sack_rate_allowed"),

        (
            pl.col("opponent_passing_epa")
            / (
                pl.col("opponent_attempts")
                + pl.col("opponent_sacks_suffered")
            )
        ).alias("passing_epa_allowed_per_play"),

        (
            pl.col("opponent_rushing_epa")
            / pl.col("opponent_carries")
        ).alias("rushing_epa_allowed_per_carry"),

        (
            pl.col("def_qb_hits")
            / (
                pl.col("opponent_attempts")
                + pl.col("opponent_sacks_suffered")
            )
        ).alias("def_qb_hit_rate"),

        (
            pl.col("attempts")
            + pl.col("sacks_suffered")
            + pl.col("carries")
        ).alias("plays_per_game"),

        (
            (pl.col("attempts") + pl.col("sacks_suffered"))
            / (pl.col("attempts") + pl.col("sacks_suffered") + pl.col("carries"))
        ).alias("pass_rate"),

        (
            (pl.col("passing_20") + pl.col("rushing_20"))
            / (pl.col("attempts") + pl.col("sacks_suffered") + pl.col("carries"))
        ).alias("explosive_play_rate"),

        (
            (pl.col("passing_tds") + pl.col("rushing_tds"))
            / (pl.col("attempts") + pl.col("sacks_suffered") + pl.col("carries"))
        ).alias("touchdown_rate"),

        (
            (pl.col("opponent_passing_20") + pl.col("opponent_rushing_20"))
            / (
                pl.col("opponent_attempts")
                + pl.col("opponent_sacks_suffered")
                + pl.col("opponent_carries")
            )
        ).alias("explosive_play_rate_allowed"),

        (
            (pl.col("opponent_passing_tds") + pl.col("opponent_rushing_tds"))
            / (
                pl.col("opponent_attempts")
                + pl.col("opponent_sacks_suffered")
                + pl.col("opponent_carries")
            )
        ).alias("touchdown_rate_allowed"),

        (
            pl.col("penalty_yards")
            / (pl.col("attempts") + pl.col("sacks_suffered") + pl.col("carries"))
        ).alias("penalty_yards_per_play"),
    )

    # Build five-game rolling averages.
    rolling_expressions = []

    for feature in RAW_FEATURES:
        expression = (
            pl.col(feature)
            .shift(1)
            .rolling_mean(
                window_size=ROLLING_WINDOW,
                min_samples=1,
            )
            .over(["team", "season"])
            .alias(f"recent_{feature}")
        )

        rolling_expressions.append(expression)

    team_stats = team_stats.with_columns(
        rolling_expressions
    )

    return team_stats


def join_team_features(
    games: pl.DataFrame,
    team_stats: pl.DataFrame,
    raw_features: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Join home and away pregame statistics to every game."""

    if raw_features is None:
        raw_features = RAW_FEATURES

    recent_features = [
        f"recent_{feature}"
        for feature in raw_features
    ]

    home_rename_map = {
        feature: f"home_{feature}"
        for feature in recent_features
    }

    away_rename_map = {
        feature: f"away_{feature}"
        for feature in recent_features
    }

    home_features = (
        team_stats
        .select(
            [
                "game_id",
                "team",
                *recent_features,
            ]
        )
        .rename(
            {
                "team": "home_team",
                **home_rename_map,
            }
        )
    )

    away_features = (
        team_stats
        .select(
            [
                "game_id",
                "team",
                *recent_features,
            ]
        )
        .rename(
            {
                "team": "away_team",
                **away_rename_map,
            }
        )
    )

    games = games.join(
        home_features,
        on=["game_id", "home_team"],
        how="left",
    )

    games = games.join(
        away_features,
        on=["game_id", "away_team"],
        how="left",
    )

    return games, recent_features


def prepare_team_form(schedules: pl.DataFrame) -> pl.DataFrame:
    """Create rolling pregame scoring and win-rate features."""

    completed = schedules.filter(
        (pl.col("game_type") == "REG")
        & pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )

    home_rows = completed.select(
        "game_id",
        "season",
        "week",
        pl.col("home_team").alias("team"),
        pl.col("home_score").cast(pl.Float64).alias("points_for"),
        pl.col("away_score").cast(pl.Float64).alias("points_against"),
    )

    away_rows = completed.select(
        "game_id",
        "season",
        "week",
        pl.col("away_team").alias("team"),
        pl.col("away_score").cast(pl.Float64).alias("points_for"),
        pl.col("home_score").cast(pl.Float64).alias("points_against"),
    )

    team_form = (
        pl.concat([home_rows, away_rows])
        .with_columns(
            (
                pl.col("points_for") - pl.col("points_against")
            ).alias("point_differential"),
            pl.when(pl.col("points_for") > pl.col("points_against"))
            .then(1.0)
            .when(pl.col("points_for") == pl.col("points_against"))
            .then(0.5)
            .otherwise(0.0)
            .alias("win_rate"),
        )
        .sort(["team", "season", "week"])
    )

    rolling_expressions = [
        (
            pl.col(feature)
            .shift(1)
            .rolling_mean(
                window_size=ROLLING_WINDOW,
                min_samples=1,
            )
            .over(["team", "season"])
            .alias(f"recent_{feature}")
        )
        for feature in FORM_RAW_FEATURES
    ]

    return team_form.with_columns(rolling_expressions)


def prepare_pbp_stats(play_by_play: pl.DataFrame) -> pl.DataFrame:
    """Aggregate play-by-play into leakage-safe rolling team form."""

    plays = play_by_play.filter(
        pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
        & (pl.col("qb_kneel").fill_null(0) == 0)
        & (pl.col("qb_spike").fill_null(0) == 0)
        & pl.col("epa").is_not_null()
    )

    drives = plays.group_by(["game_id", "posteam", "drive"]).agg(
        pl.col("drive_ended_with_score").fill_null(0).max().alias("scored"),
        pl.len().alias("drive_plays"),
        (pl.col("yardline_100") <= 20).any().alias("reached_red_zone"),
        pl.col("touchdown").fill_null(0).max().alias("drive_touchdown"),
    )

    drive_stats = drives.group_by(["game_id", "posteam"]).agg(
        pl.col("scored").mean().alias("scoring_drive_rate"),
        pl.col("drive_plays").mean().alias("plays_per_drive"),
        pl.when(pl.col("reached_red_zone"))
        .then(pl.col("drive_touchdown"))
        .otherwise(None)
        .mean()
        .alias("red_zone_td_rate"),
    )

    offense = (
        plays.group_by(["game_id", "season", "week", "posteam", "defteam"])
        .agg(
            pl.col("epa").mean().alias("pbp_epa_per_play"),
            pl.col("success").mean().alias("pbp_success_rate"),
            pl.when(pl.col("down") <= 2)
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("early_down_epa"),
            pl.when(pl.col("down") <= 2)
            .then(pl.col("success"))
            .otherwise(None)
            .mean()
            .alias("early_down_success_rate"),
            pl.when(
                (pl.col("qtr") <= 3)
                & (pl.col("score_differential").abs() <= 8)
            )
            .then(pl.col("pass_attempt"))
            .otherwise(None)
            .mean()
            .alias("neutral_pass_rate"),
            pl.when(pl.col("down") >= 3)
            .then(pl.col("first_down"))
            .otherwise(None)
            .mean()
            .alias("late_down_conversion_rate"),
            pl.when(pl.col("pass_attempt") == 1)
            .then(pl.col("complete_pass"))
            .otherwise(None)
            .mean()
            .alias("completion_rate"),
            pl.col("no_huddle").fill_null(0).mean().alias("no_huddle_rate"),
        )
        .join(drive_stats, on=["game_id", "posteam"], how="left")
        .rename({"posteam": "team", "defteam": "opponent_team"})
    )

    allowed = offense.select(
        "game_id",
        pl.col("team").alias("opponent_team"),
        pl.col("pbp_epa_per_play").alias("pbp_epa_allowed_per_play"),
        pl.col("pbp_success_rate").alias("pbp_success_rate_allowed"),
        pl.col("early_down_epa").alias("early_down_epa_allowed"),
        pl.col("red_zone_td_rate").alias("red_zone_td_rate_allowed"),
        pl.col("scoring_drive_rate").alias("scoring_drive_rate_allowed"),
    )

    team_games = (
        offense.join(allowed, on=["game_id", "opponent_team"], how="left")
        .sort(["team", "season", "week", "game_id"])
    )

    return team_games.with_columns(
        [
            pl.col(feature)
            .shift(1)
            .rolling_mean(window_size=ROLLING_WINDOW, min_samples=1)
            .over(["team", "season"])
            .alias(f"recent_{feature}")
            for feature in PBP_RAW_FEATURES
        ]
    )


def calculate_pregame_elo(schedules: pl.DataFrame) -> pl.DataFrame:
    """Calculate chronological pregame Elo ratings without leakage."""

    games = (
        schedules
        .filter(
            (pl.col("game_type") == "REG")
            & pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
        )
        .sort(["season", "week", "gameday", "game_id"])
    )

    ratings: dict[str, float] = {}
    rows: list[dict[str, str | int | float]] = []
    previous_season: int | None = None

    for game in games.iter_rows(named=True):
        season = game["season"]

        if previous_season is not None and season != previous_season:
            ratings = {
                team: 1500.0 + 0.67 * (rating - 1500.0)
                for team, rating in ratings.items()
            }

        previous_season = season
        home_team = game["home_team"]
        away_team = game["away_team"]
        home_elo = ratings.get(home_team, 1500.0)
        away_elo = ratings.get(away_team, 1500.0)

        rows.append(
            {
                "game_id": game["game_id"],
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo - away_elo,
            }
        )

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
        ratings[home_team] = home_elo + adjustment
        ratings[away_team] = away_elo - adjustment

    return pl.DataFrame(rows)


def prepare_qb_stats(
    player_stats: pl.DataFrame,
) -> pl.DataFrame:
    """Build leakage-safe rolling form features for quarterbacks."""

    qb_stats = (
        player_stats
        .filter(
            (pl.col("season_type") == "REG")
            & (pl.col("position") == "QB")
        )
        .sort(["player_id", "season", "week"])
    )

    qb_stats = qb_stats.with_columns(
        pl.when(
            (pl.col("attempts") + pl.col("sacks_suffered")) > 0
        )
        .then(
            pl.col("passing_epa")
            / (pl.col("attempts") + pl.col("sacks_suffered"))
        )
        .otherwise(None)
        .alias("qb_passing_epa_per_play"),

        pl.when(pl.col("attempts") > 0)
        .then(pl.col("passing_yards") / pl.col("attempts"))
        .otherwise(None)
        .alias("qb_yards_per_attempt"),

        pl.col("passing_cpoe").alias("qb_cpoe"),

        pl.when(pl.col("attempts") > 0)
        .then(pl.col("passing_interceptions") / pl.col("attempts"))
        .otherwise(None)
        .alias("qb_interception_rate"),
    )

    rolling_expressions = [
        (
            pl.col(feature)
            .shift(1)
            .rolling_mean(
                window_size=ROLLING_WINDOW,
                min_samples=1,
            )
            .over(["player_id", "season"])
            .alias(f"recent_{feature}")
        )
        for feature in QB_RAW_FEATURES
    ]

    return qb_stats.with_columns(rolling_expressions)


def join_qb_features(
    games: pl.DataFrame,
    qb_stats: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """Join the scheduled starting quarterbacks' recent form."""

    recent_features = [
        f"recent_{feature}"
        for feature in QB_RAW_FEATURES
    ]

    home_features = (
        qb_stats
        .select(["game_id", "player_id", *recent_features])
        .rename(
            {
                "player_id": "home_qb_id",
                **{
                    feature: f"home_{feature}"
                    for feature in recent_features
                },
            }
        )
    )

    away_features = (
        qb_stats
        .select(["game_id", "player_id", *recent_features])
        .rename(
            {
                "player_id": "away_qb_id",
                **{
                    feature: f"away_{feature}"
                    for feature in recent_features
                },
            }
        )
    )

    games = games.join(
        home_features,
        on=["game_id", "home_qb_id"],
        how="left",
    )

    games = games.join(
        away_features,
        on=["game_id", "away_qb_id"],
        how="left",
    )

    return games, recent_features


def prepare_injuries(injuries: pl.DataFrame) -> pl.DataFrame:
    """Aggregate pregame injury designations into team-week features."""

    injuries = (
        injuries
        .with_columns(
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
        )
        .filter(
            (pl.col("game_type") == "REG")
            & pl.col("report_status").is_in(
                ["Questionable", "Doubtful", "Out"]
            )
        )
    )

    injuries = injuries.with_columns(
        pl.when(pl.col("report_status") == "Out")
        .then(1.0)
        .when(pl.col("report_status") == "Doubtful")
        .then(0.75)
        .otherwise(0.35)
        .alias("status_weight"),

        pl.when(pl.col("position") == "QB")
        .then(3.0)
        .when(pl.col("position").is_in(["T", "OT", "G", "OG", "C"]))
        .then(1.5)
        .when(pl.col("position").is_in(["RB", "WR", "TE"]))
        .then(1.25)
        .otherwise(1.0)
        .alias("position_weight"),
    )

    injuries = injuries.with_columns(
        (
            pl.col("status_weight")
            * pl.col("position_weight")
        ).alias("player_injury_burden")
    )

    return injuries.group_by(["season", "week", "team"]).agg(
        pl.col("player_injury_burden").sum().alias("injury_burden"),
        (pl.col("report_status") == "Out").sum().alias("out_count"),
        (pl.col("report_status") == "Questionable")
        .sum()
        .alias("questionable_count"),
        pl.when(pl.col("position") == "QB")
        .then(pl.col("player_injury_burden"))
        .otherwise(0.0)
        .sum()
        .alias("qb_injury_burden"),
    )


def join_injury_features(
    games: pl.DataFrame,
    injuries: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """Join each team's injury burden for the upcoming game."""

    home_injuries = injuries.rename(
        {
            "team": "home_team",
            **{
                feature: f"home_{feature}"
                for feature in INJURY_FEATURES
            },
        }
    )

    away_injuries = injuries.rename(
        {
            "team": "away_team",
            **{
                feature: f"away_{feature}"
                for feature in INJURY_FEATURES
            },
        }
    )

    games = games.join(
        home_injuries,
        on=["season", "week", "home_team"],
        how="left",
    )

    games = games.join(
        away_injuries,
        on=["season", "week", "away_team"],
        how="left",
    )

    injury_columns = [
        f"{side}_{feature}"
        for side in ["home", "away"]
        for feature in INJURY_FEATURES
    ]

    games = games.with_columns(
        [pl.col(column).fill_null(0) for column in injury_columns]
    )

    return games, INJURY_FEATURES


def prepare_draft_features(
    draft_picks: pl.DataFrame,
) -> pl.DataFrame:
    """Aggregate each incoming draft class into team-season features."""

    draft_picks = draft_picks.with_columns(
        pl.col("team")
        .replace(DRAFT_TEAM_RENAMES)
        .alias("team"),
        (1.0 / pl.col("pick").cast(pl.Float64).sqrt())
        .alias("pick_value"),
    )

    return draft_picks.group_by(["season", "team"]).agg(
        pl.len().alias("draft_pick_count"),
        (pl.col("pick") <= 100).sum().alias("top_100_pick_count"),
        (pl.col("round") == 1).sum().alias("first_round_pick_count"),
        pl.col("pick_value").sum().alias("total_draft_capital"),
        pl.when(pl.col("side") == "O")
        .then(pl.col("pick_value"))
        .otherwise(0.0)
        .sum()
        .alias("offensive_draft_capital"),
        pl.when(pl.col("side") == "D")
        .then(pl.col("pick_value"))
        .otherwise(0.0)
        .sum()
        .alias("defensive_draft_capital"),
        (
            (pl.col("position") == "QB")
            & (pl.col("round") == 1)
        )
        .any()
        .cast(pl.Int8)
        .alias("rookie_qb_first_round"),
        (
            (pl.col("position") == "QB")
            & (pl.col("pick") <= 10)
        )
        .any()
        .cast(pl.Int8)
        .alias("rookie_qb_top_10"),
    )


def join_draft_features(
    games: pl.DataFrame,
    draft_features: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    """Attach home and away draft-class strength to each game."""

    home_draft = draft_features.rename(
        {
            "team": "home_team",
            **{
                feature: f"home_{feature}"
                for feature in DRAFT_FEATURES
            },
        }
    )

    away_draft = draft_features.rename(
        {
            "team": "away_team",
            **{
                feature: f"away_{feature}"
                for feature in DRAFT_FEATURES
            },
        }
    )

    games = games.join(
        home_draft,
        on=["season", "home_team"],
        how="left",
    )

    games = games.join(
        away_draft,
        on=["season", "away_team"],
        how="left",
    )

    draft_columns = [
        f"{side}_{feature}"
        for side in ["home", "away"]
        for feature in DRAFT_FEATURES
    ]

    games = games.with_columns(
        [pl.col(column).fill_null(0) for column in draft_columns]
    )

    return games, DRAFT_FEATURES


def create_difference_features(
    games: pl.DataFrame,
    recent_features: list[str],
) -> tuple[pl.DataFrame, list[str]]:
    """Calculate the home team's advantage for each feature."""

    difference_expressions = []
    model_features = []

    for feature in recent_features:
        difference_name = f"{feature}_diff"

        difference_expressions.append(
            (
                pl.col(f"home_{feature}")
                - pl.col(f"away_{feature}")
            ).alias(difference_name)
        )

        model_features.append(difference_name)

    difference_expressions.append(
        (
            pl.col("home_rest")
            - pl.col("away_rest")
        ).alias("rest_diff")
    )

    model_features.append("rest_diff")

    games = games.with_columns(
        difference_expressions
    )

    return games, model_features


def create_sum_features(
    games: pl.DataFrame,
    recent_features: list[str],
) -> tuple[pl.DataFrame, list[str]]:
    """Create combined home-plus-away features for total predictions."""

    expressions = []
    sum_features = []

    for feature in recent_features:
        sum_name = f"{feature}_sum"
        expressions.append(
            (
                pl.col(f"home_{feature}")
                + pl.col(f"away_{feature}")
            ).alias(sum_name)
        )
        sum_features.append(sum_name)

    return games.with_columns(expressions), sum_features


def create_matchup_features(games: pl.DataFrame) -> pl.DataFrame:
    """Compare each offense with the defense it is about to face."""

    games = games.with_columns(
        (
            pl.col("home_recent_passing_epa_per_play")
            - pl.col("away_recent_passing_epa_allowed_per_play")
        ).alias("home_pass_matchup"),
        (
            pl.col("away_recent_passing_epa_per_play")
            - pl.col("home_recent_passing_epa_allowed_per_play")
        ).alias("away_pass_matchup"),
        (
            pl.col("home_recent_rushing_epa_per_carry")
            - pl.col("away_recent_rushing_epa_allowed_per_carry")
        ).alias("home_rush_matchup"),
        (
            pl.col("away_recent_rushing_epa_per_carry")
            - pl.col("home_recent_rushing_epa_allowed_per_carry")
        ).alias("away_rush_matchup"),
        (
            (pl.col("home_recent_points_for") + pl.col("away_recent_points_against"))
            / 2.0
        ).alias("expected_home_points"),
        (
            (pl.col("away_recent_points_for") + pl.col("home_recent_points_against"))
            / 2.0
        ).alias("expected_away_points"),
    )

    games = games.with_columns(
        (pl.col("expected_home_points") - pl.col("expected_away_points"))
        .alias("expected_margin"),
        (pl.col("expected_home_points") + pl.col("expected_away_points"))
        .alias("expected_total"),
    )

    return games.with_columns(
        (pl.col("expected_margin") - pl.col("spread_line"))
        .alias("margin_vs_spread"),
        (pl.col("expected_total") - pl.col("total_line"))
        .alias("total_vs_line"),
    )


def save_model_data(
    games: pl.DataFrame,
    model_features: list[str],
    required_features: list[str],
) -> None:
    """Save the completed model-ready dataset."""

    model_data = (
        games
        .drop_nulls(subset=required_features)
        .select(
            [
                "game_id",
                "season",
                "week",
                "gameday",
                "away_team",
                "home_team",
                "home_win",
                "home_score",
                "away_score",
                "home_cover",
                "over_hit",
                "home_margin",
                "game_total",
                "spread_line",
                "total_line",
                "away_moneyline",
                "home_moneyline",
                "away_spread_odds",
                "home_spread_odds",
                "under_odds",
                "over_odds",
                "neutral_site",
                "is_outdoors",
                "temp",
                "wind",
                *model_features,
            ]
        )
        .sort(["season", "week", "game_id"])
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DATA_DIR
        / "model_data.parquet"
    )

    model_data.write_parquet(output_path)

    print("\nModel dataset created successfully.")
    print(f"Rows: {model_data.height:,}")
    print(f"Columns: {model_data.width:,}")
    print(f"Saved to: {output_path}")

    print("\nModel features:")

    for feature in model_features:
        print(f"- {feature}")


def main() -> None:
    """Run the complete feature-engineering pipeline."""

    print("Loading raw data...")
    schedules, team_stats, player_stats, injuries, draft_picks = load_data()

    print("Preparing completed games...")
    games = prepare_games(schedules)

    print("Calculating rolling team statistics...")
    team_stats = prepare_team_stats(team_stats)

    print("Joining home and away features...")
    games, recent_features = join_team_features(
        games,
        team_stats,
    )

    print("Calculating matchup differences...")
    games, team_model_features = create_difference_features(
        games,
        recent_features,
    )
    games, team_sum_features = create_sum_features(
        games,
        recent_features,
    )

    print("Calculating recent scoring and win form...")
    team_form = prepare_team_form(schedules)
    games, form_recent_features = join_team_features(
        games,
        team_form,
        raw_features=FORM_RAW_FEATURES,
    )
    games, form_model_features = create_difference_features(
        games,
        form_recent_features,
    )
    games, form_sum_features = create_sum_features(
        games,
        form_recent_features,
    )

    print("Calculating offense-versus-defense matchup features...")
    games = create_matchup_features(games)

    print("Calculating play-by-play efficiency and pace...")
    play_by_play = pl.read_parquet(RAW_DATA_DIR / "play_by_play.parquet")
    pbp_stats = prepare_pbp_stats(play_by_play)
    games, pbp_recent_features = join_team_features(
        games,
        pbp_stats,
        raw_features=PBP_RAW_FEATURES,
    )
    games, pbp_model_features = create_difference_features(
        games,
        pbp_recent_features,
    )
    games, pbp_sum_features = create_sum_features(
        games,
        pbp_recent_features,
    )

    print("Calculating pregame Elo ratings...")
    elo_features = calculate_pregame_elo(schedules)
    games = games.join(elo_features, on="game_id", how="left")

    print("Calculating starting-quarterback form...")
    qb_stats = prepare_qb_stats(player_stats)
    games, qb_recent_features = join_qb_features(games, qb_stats)
    games, qb_model_features = create_difference_features(
        games,
        qb_recent_features,
    )
    games, qb_sum_features = create_sum_features(
        games,
        qb_recent_features,
    )

    print("Calculating injury and availability features...")
    injury_features = prepare_injuries(injuries)
    games, injury_features = join_injury_features(
        games,
        injury_features,
    )
    games, injury_model_features = create_difference_features(
        games,
        injury_features,
    )
    games, injury_sum_features = create_sum_features(
        games,
        injury_features,
    )

    print("Calculating draft-class and rookie features...")
    draft_features = prepare_draft_features(draft_picks)
    games, draft_features = join_draft_features(
        games,
        draft_features,
    )
    games, draft_model_features = create_difference_features(
        games,
        draft_features,
    )

    model_features = list(
        dict.fromkeys(
            team_model_features
            + form_model_features
            + qb_model_features
            + injury_model_features
            + draft_model_features
            + ["elo_diff"]
            + team_sum_features
            + form_sum_features
            + qb_sum_features
            + injury_sum_features
            + MATCHUP_FEATURES
            + pbp_model_features
            + pbp_sum_features
        )
    )

    save_model_data(
        games,
        model_features,
        required_features=team_model_features,
    )


if __name__ == "__main__":
    main()
