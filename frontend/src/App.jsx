import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const apiUrl = (path) => `${API_BASE_URL}${path}`;

const fallback = (value, defaultValue) => value ?? defaultValue;

function inputsFor(game) {
  return {
    home_spread: fallback(game.home_spread, 0),
    total_line: fallback(game.total_line, 44.5),
    home_moneyline: fallback(game.home_moneyline, -110),
    away_moneyline: fallback(game.away_moneyline, -110),
    home_spread_odds: fallback(game.home_spread_odds, -110),
    away_spread_odds: fallback(game.away_spread_odds, -110),
    over_odds: fallback(game.over_odds, -110),
    under_odds: fallback(game.under_odds, -110),
  };
}

function gameLabel(game) {
  return `Week ${game.week}: ${game.away_team} at ${game.home_team} (${game.gameday})`;
}

function LineInput({ label, name, value, onChange, step = 5, min }) {
  return (
    <label>
      <span>{label}</span>
      <input
        type="number"
        name={name}
        value={value}
        step={step}
        min={min}
        onChange={(event) => onChange(name, Number(event.target.value))}
        required
      />
    </label>
  );
}

function GameLines({ game, values, onChange }) {
  return (
    <fieldset>
      <legend>{gameLabel(game)}</legend>
      <p className="quarterbacks">
        Expected QBs: {game.away_expected_qb_name} vs. {game.home_expected_qb_name}
      </p>
      <div className="input-grid">
        <LineInput label={`${game.home_team} moneyline`} name="home_moneyline" value={values.home_moneyline} onChange={onChange} />
        <LineInput label={`${game.away_team} moneyline`} name="away_moneyline" value={values.away_moneyline} onChange={onChange} />
        <LineInput label={`${game.home_team} spread`} name="home_spread" value={values.home_spread} onChange={onChange} step={0.5} />
        <LineInput label="Over/under total" name="total_line" value={values.total_line} onChange={onChange} step={0.5} min={1} />
        <LineInput label="Home spread odds" name="home_spread_odds" value={values.home_spread_odds} onChange={onChange} />
        <LineInput label="Away spread odds" name="away_spread_odds" value={values.away_spread_odds} onChange={onChange} />
        <LineInput label="Over odds" name="over_odds" value={values.over_odds} onChange={onChange} />
        <LineInput label="Under odds" name="under_odds" value={values.under_odds} onChange={onChange} />
      </div>
    </fieldset>
  );
}

function valueText(selection, expectedValue) {
  if (expectedValue <= 0) return "No positive-value side at these odds";
  return `Best value: ${selection} (${(expectedValue * 100).toFixed(1)}% expected return per $1)`;
}

function Prediction({ result }) {
  return (
    <section className="result">
      <h2>{result.away_team} at {result.home_team}</h2>
      <p>
        Projected score: {result.away_team} {result.predicted_away.toFixed(1)}, {" "}
        {result.home_team} {result.predicted_home.toFixed(1)} (total {result.predicted_total.toFixed(1)})
      </p>
      <div className="market-grid">
        <article>
          <h3>Moneyline</h3>
          <strong>{result.moneyline_pick}</strong>
          <p>{(result.moneyline_probability * 100).toFixed(1)}% win probability</p>
          <small>{valueText(result.moneyline_value, result.moneyline_best_ev)}</small>
        </article>
        <article>
          <h3>Spread</h3>
          <strong>{result.spread_pick}</strong>
          <p>{(result.spread_probability * 100).toFixed(1)}% cover probability</p>
          <small>{valueText(result.spread_value, result.spread_best_ev)}</small>
        </article>
        <article>
          <h3>Total</h3>
          <strong>{result.total_pick}</strong>
          <p>{(result.total_probability * 100).toFixed(1)}% probability</p>
          <small>{valueText(result.total_value, result.total_best_ev)}</small>
        </article>
      </div>
    </section>
  );
}

export default function App() {
  const [games, setGames] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [gameSearch, setGameSearch] = useState("");
  const [lineInputs, setLineInputs] = useState({});
  const [results, setResults] = useState([]);
  const [refresh, setRefresh] = useState(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function loadGames(preserveSelection = true) {
    let response;
    try {
      response = await fetch(apiUrl("/api/games"));
    } catch {
      throw new Error(
        "The Python API is not running on port 8000. Stop this site with Control+C, "
        + "then run `python start_react_app.py` from the project folder.",
      );
    }
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail ?? `The prediction API returned error ${response.status}.`);
    }
    const data = await response.json();
    setGames(data.games);
    setRefresh(data.refresh);
    if (!preserveSelection) {
      setSelectedId("");
      setGameSearch("");
    }
  }

  useEffect(() => {
    loadGames(false).catch((reason) => setError(reason.message)).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!refresh || !["refreshing", "never_refreshed"].includes(refresh.state)) return undefined;
    const timer = window.setInterval(async () => {
      const response = await fetch(apiUrl("/api/refresh-status"));
      const status = await response.json();
      setRefresh(status);
      if (status.state !== "refreshing") {
        window.clearInterval(timer);
        await loadGames(true);
      }
    }, 5000);
    return () => window.clearInterval(timer);
  }, [refresh?.state]);

  const gameById = useMemo(
    () => Object.fromEntries(games.map((game) => [game.game_id, game])),
    [games],
  );

  function searchForGame(value) {
    setGameSearch(value);
    const game = games.find((candidate) => gameLabel(candidate) === value);
    const gameId = game?.game_id ?? "";
    setSelectedId(gameId);
    if (gameId) {
      setLineInputs((current) => ({
        ...current,
        [gameId]: current[gameId] ?? inputsFor(game),
      }));
    }
    setResults([]);
  }

  function changeLine(gameId, name, value) {
    setLineInputs((current) => ({
      ...current,
      [gameId]: { ...current[gameId], [name]: value },
    }));
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const invalidOdds = Object.entries(lineInputs[selectedId]).some(
        ([name, value]) => (name.includes("odds") || name.includes("moneyline")) && value === 0,
      );
      if (invalidOdds) throw new Error("American odds cannot be zero.");
      const response = await fetch(apiUrl("/api/predict"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ game_id: selectedId, ...lineInputs[selectedId] }),
      });
      if (!response.ok) {
        const body = await response.json();
        throw new Error(body.detail ?? "Prediction failed.");
      }
      setResults([await response.json()]);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) return <main><p>Loading the 2026 schedule...</p></main>;

  return (
    <main>
      <h1>NFL Custom Line Predictor</h1>
      <p>Search for a remaining 2026 game, enter your own lines, and run the models.</p>
      {refresh && (
        <p className="status">
          Data status: {refresh.state.replaceAll("_", " ")}
          {refresh.latest_completed_week ? ` through Week ${refresh.latest_completed_week}` : ""}
        </p>
      )}
      {error && <p className="error" role="alert">{error}</p>}
      {error && games.length === 0 && (
        <button
          className="retry"
          type="button"
          onClick={() => {
            setLoading(true);
            setError("");
            loadGames(false).catch((reason) => setError(reason.message)).finally(() => setLoading(false));
          }}
        >
          Retry API connection
        </button>
      )}

      <form onSubmit={submit}>
        <label className="game-search">
          <span>Search games</span>
          <input
            type="search"
            list="game-options"
            value={gameSearch}
            onChange={(event) => searchForGame(event.target.value)}
            placeholder="Search by team, week, or date"
            autoComplete="off"
            required
          />
        </label>
        <datalist id="game-options">
          {games.map((game) => (
            <option key={game.game_id} value={gameLabel(game)} />
          ))}
        </datalist>

        {selectedId && (
          <GameLines
            game={gameById[selectedId]}
            values={lineInputs[selectedId] ?? inputsFor(gameById[selectedId])}
            onChange={(name, value) => changeLine(selectedId, name, value)}
          />
        )}
        <button type="submit" disabled={submitting || !selectedId}>
          {submitting ? "Running..." : "Run predictions"}
        </button>
      </form>

      {results.map((result) => <Prediction key={result.game_id} result={result} />)}
    </main>
  );
}
