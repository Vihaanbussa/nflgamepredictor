"""Validate the current 2026 rosters and identify incoming rookies."""

from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEASON = 2026

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


def main() -> None:
    """Build current-roster, rookie, and team-summary artifacts."""

    rosters = pl.read_parquet(
        RAW_DATA_DIR / "rosters_2026.parquet"
    )

    draft_picks = (
        pl.read_parquet(RAW_DATA_DIR / "draft_picks.parquet")
        .filter(pl.col("season") == SEASON)
        .with_columns(
            pl.col("team").replace(DRAFT_TEAM_RENAMES).alias("team")
        )
    )

    current_roster = (
        rosters
        .sort(["team", "full_name", "week"], descending=[False, False, True])
        .unique(["team", "full_name"], keep="first", maintain_order=True)
        .sort(["team", "position", "full_name"])
    )

    drafted_rookies = (
        draft_picks
        .select(
            "team",
            pl.col("pfr_player_name").alias("player_name"),
            "position",
            "round",
            "pick",
            "college",
            "gsis_id",
        )
        .sort("pick")
    )

    roster_summary = current_roster.group_by("team").agg(
        pl.len().alias("roster_player_count"),
        (pl.col("status") == "ACT").sum().alias("active_count"),
        (pl.col("rookie_year") == SEASON).sum().alias("rookie_count"),
        (
            (pl.col("rookie_year") == SEASON)
            & (pl.col("position") == "QB")
        )
        .sum()
        .alias("rookie_qb_count"),
        pl.col("years_exp").mean().alias("average_years_experience"),
    )

    draft_summary = draft_picks.group_by("team").agg(
        pl.len().alias("drafted_rookie_count"),
        (pl.col("round") == 1).sum().alias("first_round_rookie_count"),
        (pl.col("pick") <= 100).sum().alias("top_100_rookie_count"),
    )

    team_summary = (
        roster_summary
        .join(draft_summary, on="team", how="left")
        .sort("team")
    )

    if current_roster.get_column("team").n_unique() != 32:
        raise ValueError("The 2026 roster snapshot does not contain 32 teams.")

    mendoza_on_roster = current_roster.filter(
        (pl.col("team") == "LV")
        & (pl.col("full_name") == "Fernando Mendoza")
    )

    mendoza_drafted = drafted_rookies.filter(
        (pl.col("team") == "LV")
        & (pl.col("player_name") == "Fernando Mendoza")
        & (pl.col("pick") == 1)
    )

    if mendoza_on_roster.height != 1 or mendoza_drafted.height != 1:
        raise ValueError("Fernando Mendoza failed roster/draft reconciliation.")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "roster_2026": current_roster,
        "drafted_rookies_2026": drafted_rookies,
        "team_roster_summary_2026": team_summary,
    }

    for name, dataframe in outputs.items():
        base = PROCESSED_DATA_DIR / name
        dataframe.write_parquet(base.with_suffix(".parquet"))
        dataframe.write_csv(base.with_suffix(".csv"))
        print(f"Saved {name}: {dataframe.height:,} rows")

    print("\nValidated Fernando Mendoza: LV roster, QB, No. 1 pick.")


if __name__ == "__main__":
    main()
