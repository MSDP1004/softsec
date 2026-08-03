"""네트워크 없이 파이프라인 전체 흐름(수집 -> 필터링 -> 스코어링 -> 리포트)을 검증.

실제 yfinance 호출은 monkeypatch로 대체하고, 노드별로 밸류에이션이 다른
합성 데이터를 흘려서 저평가 노드 선별과 퀄리티 필터가 함께 맞물려 동작하는지
확인한다.
"""

from __future__ import annotations

from pathlib import Path

import src.pipeline as pipeline_module
from src.data_fetcher import TickerSnapshot
from src.report import render_markdown


def _snapshot(ticker, forward_pe, price_to_sales, peg_ratio, revenue, net_income):
    return TickerSnapshot(
        ticker=ticker,
        trailing_pe=forward_pe,
        forward_pe=forward_pe,
        price_to_sales=price_to_sales,
        peg_ratio=peg_ratio,
        total_revenue=revenue,
        net_income=net_income,
        profit_margin=0.1 if net_income and net_income > 0 else -0.05,
        revenue_growth=0.15,
        return_on_equity=0.12,
        market_cap=1_000_000_000,
    )


FAKE_DATA = {
    # 저평가 노드: PE/PS/PEG가 낮음, 우량주 하나 + 적자기업 하나(필터에서 걸러져야 함)
    "cheap_node": {
        "GOOD1": _snapshot("GOOD1", 10, 2, 0.8, revenue=5_000_000_000, net_income=500_000_000),
        "LOSER1": _snapshot("LOSER1", 10, 2, 0.8, revenue=0, net_income=-100_000_000),
    },
    # 고평가 노드: PE/PS/PEG가 높음
    "expensive_node": {
        "GOOD2": _snapshot("GOOD2", 50, 15, 4.0, revenue=2_000_000_000, net_income=200_000_000),
    },
    # 중간 노드
    "mid_node": {
        "GOOD3": _snapshot("GOOD3", 25, 6, 2.0, revenue=3_000_000_000, net_income=300_000_000),
    },
}


def _fake_fetch_snapshots(tickers: list[str]) -> dict[str, TickerSnapshot]:
    for node_snapshots in FAKE_DATA.values():
        if set(tickers) & set(node_snapshots):
            return {t: node_snapshots[t] for t in tickers if t in node_snapshots}
    return {}


def test_pipeline_end_to_end(tmp_path: Path, monkeypatch):
    taxonomy_yaml = tmp_path / "taxonomy.yaml"
    taxonomy_yaml.write_text(
        """
version: 1
last_updated: "2026-08-03"
nodes:
  - id: cheap_node
    name: "저평가 노드"
    stage: midstream
    tickers: [GOOD1, LOSER1]
  - id: expensive_node
    name: "고평가 노드"
    stage: midstream
    tickers: [GOOD2]
  - id: mid_node
    name: "중간 노드"
    stage: midstream
    tickers: [GOOD3]
review:
  pending_additions: []
  pending_removals: []
  history: []
""",
        encoding="utf-8",
    )
    history_csv = tmp_path / "history.csv"

    monkeypatch.setattr(pipeline_module, "fetch_snapshots", _fake_fetch_snapshots)

    result = pipeline_module.run_pipeline(
        taxonomy_path=taxonomy_yaml,
        history_path=history_csv,
        top_k_nodes=1,
    )

    # 가장 저평가된 노드(cheap_node)만 top_k=1로 선택되어야 함
    assert {c.node_id for c in result.candidates} == {"cheap_node"}

    # cheap_node 안에서도 적자기업(LOSER1)은 퀄리티 필터에서 제외되어야 함
    tickers = {c.ticker for c in result.candidates}
    assert tickers == {"GOOD1"}
    assert "LOSER1" not in tickers

    cheap = next(nv for nv in result.node_valuations if nv.node_id == "cheap_node")
    expensive = next(nv for nv in result.node_valuations if nv.node_id == "expensive_node")
    assert cheap.undervaluation_score < expensive.undervaluation_score

    # history CSV에 3개 노드가 모두 기록되어야 함
    assert history_csv.exists()
    lines = history_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 + 3  # header + 3 nodes

    # 리포트 렌더링까지 끝까지 에러 없이 동작하는지 확인
    report = render_markdown(result)
    assert "GOOD1" in report
    assert "LOSER1" not in report
