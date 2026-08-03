"""run_daily()를 이틀치 실행으로 시뮬레이션해 예측->만기->평가 사이클 전체를 검증.

실제 yfinance 호출은 monkeypatch로 대체하고, 하루 사이 가격이 바뀌었을 때
1일(1d) 예측이 정확히 만기 판정되고 실현수익률이 올바르게 계산되는지 확인한다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import src.daily as daily_module
from src.backtest import read_csv_rows
from src.data_fetcher import TickerSnapshot


def _make_snapshot(ticker: str, price: float) -> TickerSnapshot:
    return TickerSnapshot(
        ticker=ticker,
        trailing_pe=15.0,
        forward_pe=15.0,
        price_to_sales=3.0,
        peg_ratio=1.0,
        total_revenue=1_000_000_000,
        net_income=100_000_000,
        profit_margin=0.1,
        revenue_growth=0.2,
        return_on_equity=0.15,
        market_cap=10_000_000_000,
        price=price,
        analyst_target_mean=price * 1.1,  # 섀도우 피처(analyst_upside) 계산용
    )


def _write_candidate_features(path: Path) -> None:
    path.write_text(
        """
candidates:
  - id: analyst_upside
    name: "테스트용 섀도우 피처"
    description: "targetMeanPrice/price - 1"
    rationale: "test"
    status: shadow
    source: test
    proposed_date: "2026-01-01"
research_history: []
""",
        encoding="utf-8",
    )


def _write_taxonomy(path: Path) -> None:
    path.write_text(
        """
version: 1
last_updated: "2026-01-01"
nodes:
  - id: node1
    name: "테스트 노드"
    stage: midstream
    tickers: [AAA, BBB]
review:
  pending_additions: []
  pending_removals: []
  history: []
""",
        encoding="utf-8",
    )


def test_predict_then_evaluate_cycle_across_two_days(tmp_path: Path, monkeypatch):
    taxonomy_path = tmp_path / "taxonomy.yaml"
    _write_taxonomy(taxonomy_path)

    weights_path = tmp_path / "weights.json"
    predictions_path = tmp_path / "predictions.csv"
    evaluations_path = tmp_path / "evaluations.csv"
    summary_path = tmp_path / "summary.md"
    candidate_features_path = tmp_path / "candidate_features.yaml"
    shadow_predictions_path = tmp_path / "shadow_predictions.csv"
    _write_candidate_features(candidate_features_path)

    prices = {"AAA": 100.0, "BBB": 100.0}

    def fake_fetch_snapshots(tickers):
        return {t: _make_snapshot(t, prices[t]) for t in tickers if t in prices}

    monkeypatch.setattr(daily_module, "fetch_snapshots", fake_fetch_snapshots)

    day0 = date(2026, 1, 1)
    result0 = daily_module.run_daily(
        today=day0,
        top_n_per_node=2,
        taxonomy_path=taxonomy_path,
        weights_path=weights_path,
        predictions_path=predictions_path,
        evaluations_path=evaluations_path,
        summary_path=summary_path,
        candidate_features_path=candidate_features_path,
        shadow_predictions_path=shadow_predictions_path,
    )

    assert result0.n_tracked == 2
    assert result0.n_predictions_logged == 8  # 2 tickers x 4 horizons
    assert result0.n_evaluated == 0  # 아직 어떤 horizon도 만기 전
    assert result0.n_shadow_values_logged == 8  # 2 tickers x 4 horizons, shadow 피처 1개

    shadow_rows = read_csv_rows(shadow_predictions_path)
    assert len(shadow_rows) == 8
    assert all(r["feature_id"] == "analyst_upside" for r in shadow_rows)
    assert abs(float(shadow_rows[0]["value"]) - 0.1) < 1e-9  # target=price*1.1 -> upside 10%

    # 다음 날: AAA는 +1%, BBB는 -1%
    prices["AAA"] = 101.0
    prices["BBB"] = 99.0

    day1 = date(2026, 1, 2)
    result1 = daily_module.run_daily(
        today=day1,
        top_n_per_node=2,
        taxonomy_path=taxonomy_path,
        weights_path=weights_path,
        predictions_path=predictions_path,
        evaluations_path=evaluations_path,
        summary_path=summary_path,
        candidate_features_path=candidate_features_path,
        shadow_predictions_path=shadow_predictions_path,
    )

    assert result1.n_predictions_logged == 8  # 그 날 또 새 예측 8건
    assert result1.n_evaluated == 2  # AAA, BBB의 1d 예측만 만기

    eval_rows = {r["ticker"]: r for r in read_csv_rows(evaluations_path)}
    assert eval_rows["AAA"]["horizon"] == "1d"
    assert abs(float(eval_rows["AAA"]["realized_return"]) - 0.01) < 1e-9
    assert abs(float(eval_rows["BBB"]["realized_return"]) - (-0.01)) < 1e-9
    # 초기 가중치가 전부 0이므로 예측치는 0, 오차 = 실현수익률 그대로
    assert abs(float(eval_rows["AAA"]["error"]) - 0.01) < 1e-9

    # 표본이 너무 적어(2개) 가중치 재적합은 아직 일어나지 않아야 함
    assert result1.calibrated_horizons == []

    assert summary_path.exists()
    assert "일일 예측/평가 요약" in summary_path.read_text(encoding="utf-8")
