"""매일 실행되는 오케스트레이션: 예측 기록 -> 만기 예측 평가 -> (표본 충분하면) 가중치 재적합 -> 요약 생성."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .backtest import (
    append_evaluations,
    append_predictions,
    build_predictions,
    evaluate_predictions,
    find_matured_predictions,
    read_csv_rows,
)
from .calibration import calibrate_horizon
from .data_fetcher import fetch_snapshots
from .feature_research import load_candidates
from .predictor import HORIZONS, feature_vector, load_weights, save_weights
from .quality_filter import QualityThresholds, filter_snapshots
from .shadow_backtest import append_shadow_rows, build_shadow_rows
from .shadow_features import compute_shadow_values
from .taxonomy_manager import load_taxonomy
from .universe import select_top_n_per_node
from .valuation import compute_node_valuation, score_undervaluation

BASE_DIR = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = BASE_DIR / "config" / "taxonomy.yaml"
WEIGHTS_PATH = BASE_DIR / "config" / "model_weights.json"
CANDIDATE_FEATURES_PATH = BASE_DIR / "config" / "candidate_features.yaml"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"
EVALUATIONS_PATH = BASE_DIR / "data" / "evaluations.csv"
SHADOW_PREDICTIONS_PATH = BASE_DIR / "data" / "shadow_predictions.csv"
SUMMARY_PATH = BASE_DIR / "data" / "daily_summary.md"

# horizon별 재적합 최소 표본 수. 1일/1주는 노이즈가 커서 표본을 더 많이 요구한다.
# 피처가 4개->6개로 늘면서 추정할 파라미터가 늘어난 만큼 기준도 함께 올렸다.
MIN_SAMPLES = {"1d": 40, "1w": 30, "1m": 25, "1y": 15}


@dataclass
class DailyResult:
    n_tracked: int
    n_predictions_logged: int
    n_evaluated: int
    calibrated_horizons: list[str]
    eval_summary: dict[str, dict] = field(default_factory=dict)
    active_features: dict[str, list[str]] = field(default_factory=dict)
    n_shadow_values_logged: int = 0


def run_daily(
    today: date | None = None,
    top_n_per_node: int = 2,
    taxonomy_path: Path = TAXONOMY_PATH,
    weights_path: Path = WEIGHTS_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    evaluations_path: Path = EVALUATIONS_PATH,
    summary_path: Path = SUMMARY_PATH,
    candidate_features_path: Path = CANDIDATE_FEATURES_PATH,
    shadow_predictions_path: Path = SHADOW_PREDICTIONS_PATH,
) -> DailyResult:
    today = today or date.today()

    taxonomy = load_taxonomy(taxonomy_path)
    thresholds = QualityThresholds()

    node_valuations = []
    quality_by_node = {}
    for node in taxonomy.nodes:
        snapshots = fetch_snapshots(node.tickers)
        node_valuations.append(compute_node_valuation(node.id, node.name, snapshots))
        quality_by_node[node.id] = filter_snapshots(snapshots, thresholds)
    node_valuations = score_undervaluation(node_valuations)

    tracked = select_top_n_per_node(node_valuations, quality_by_node, top_n=top_n_per_node)

    model = load_weights(weights_path)
    candidates = load_candidates(candidate_features_path)
    shadow_ids = {c["id"] for c in candidates.candidates if c.get("status") == "shadow"}

    new_predictions = []
    new_shadow_rows = []
    for t in tracked:
        if t.snapshot.price is None:
            continue
        features = feature_vector(
            t.node_valuation.undervaluation_score if t.node_valuation else None,
            t.snapshot.revenue_growth,
            t.snapshot.profit_margin,
            t.snapshot.peg_ratio,
            t.snapshot.momentum_5d,
            t.snapshot.momentum_21d,
        )
        new_predictions.extend(
            build_predictions(today, t.ticker, t.node_id, t.snapshot.price, features, model)
        )

        if shadow_ids:
            shadow_values = compute_shadow_values(t.snapshot, shadow_ids)
            new_shadow_rows.extend(build_shadow_rows(today, t.ticker, t.node_id, shadow_values))
    append_predictions(new_predictions, predictions_path)
    if new_shadow_rows:
        append_shadow_rows(new_shadow_rows, shadow_predictions_path)

    matured = find_matured_predictions(predictions_path, evaluations_path, today)
    matured_tickers = sorted({row["ticker"] for row in matured})
    current_snapshots = fetch_snapshots(matured_tickers) if matured_tickers else {}
    current_prices = {tkr: snap.price for tkr, snap in current_snapshots.items() if snap.price is not None}
    evals = evaluate_predictions(matured, current_prices)
    append_evaluations(evals, evaluations_path)

    calibrated_horizons = []
    eval_summary: dict[str, dict] = {}
    all_evals = read_csv_rows(evaluations_path)
    for horizon in HORIZONS:
        rows = [r for r in all_evals if r["horizon"] == horizon]
        if rows:
            errors = [float(r["error"]) for r in rows]
            realized = [float(r["realized_return"]) for r in rows]
            predicted = [float(r["predicted_return"]) for r in rows]
            hits = sum(1 for p, r in zip(predicted, realized) if (p >= 0) == (r >= 0))
            eval_summary[horizon] = {
                "n": len(rows),
                "mae": sum(abs(e) for e in errors) / len(errors),
                "direction_hit_rate": hits / len(rows),
            }
        if len(rows) >= MIN_SAMPLES[horizon]:
            features = [
                [float(r["x1"]), float(r["x2"]), float(r["x3"]), float(r["x4"]), float(r["x5"]), float(r["x6"])]
                for r in rows
            ]
            targets = [float(r["realized_return"]) for r in rows]
            if calibrate_horizon(model, horizon, features, targets, min_samples=MIN_SAMPLES[horizon]):
                calibrated_horizons.append(horizon)

    save_weights(model, weights_path)

    active_features = {
        horizon: model.meta[horizon].get("active_features", [])
        for horizon in calibrated_horizons
    }

    result = DailyResult(
        n_tracked=len(tracked),
        n_predictions_logged=len(new_predictions),
        n_evaluated=len(evals),
        calibrated_horizons=calibrated_horizons,
        eval_summary=eval_summary,
        active_features=active_features,
        n_shadow_values_logged=len(new_shadow_rows),
    )
    write_summary(result, today, summary_path)
    return result


def write_summary(result: DailyResult, today: date, summary_path: Path) -> None:
    lines = [f"# 일일 예측/평가 요약 ({today.isoformat()})", ""]
    lines.append(f"- 추적 종목: {result.n_tracked}개")
    lines.append(f"- 오늘 새로 기록한 예측: {result.n_predictions_logged}건")
    lines.append(f"- 오늘 만기 도달해 평가된 예측: {result.n_evaluated}건")
    if result.n_shadow_values_logged:
        lines.append(f"- 섀도우 피처(검증 대기 중, 실거래 예측식에는 미반영) 값 기록: {result.n_shadow_values_logged}건")
    if result.calibrated_horizons:
        lines.append(f"- 가중치 재적합됨: {', '.join(result.calibrated_horizons)}")
    else:
        lines.append("- 가중치 재적합: 없음 (표본 부족)")
    lines.append("")
    lines.append("| horizon | 누적 평가 표본 | MAE(오차) | 방향 적중률 | 최소표본 |")
    lines.append("|---|---|---|---|---|")
    for horizon in HORIZONS:
        s = result.eval_summary.get(horizon)
        min_n = MIN_SAMPLES[horizon]
        if s:
            lines.append(
                f"| {horizon} | {s['n']} | {s['mae'] * 100:.2f}%p | {s['direction_hit_rate'] * 100:.0f}% | {min_n} |"
            )
        else:
            lines.append(f"| {horizon} | 0 | - | - | {min_n} |")
    lines.append("")

    if result.active_features:
        lines.append("## Elastic Net이 유지한 피처 (재적합된 horizon만)")
        lines.append("")
        for horizon, features in result.active_features.items():
            shown = ", ".join(features) if features else "(전부 0으로 축소됨 — 유의미한 피처 없음)"
            lines.append(f"- {horizon}: {shown}")
        lines.append("")
    lines.append(
        "> 방향 적중률은 예측 부호(오를지/내릴지)와 실제 부호가 일치한 비율입니다. "
        "표본이 최소 기준 미만인 horizon은 아직 가중치가 재적합되지 않은 상태(초기값)입니다."
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
