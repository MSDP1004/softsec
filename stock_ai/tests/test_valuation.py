from pathlib import Path

from src.data_fetcher import TickerSnapshot
from src.valuation import (
    NodeValuation,
    append_history,
    compute_node_valuation,
    score_undervaluation,
)


def _snapshot(ticker, forward_pe, price_to_sales, peg_ratio):
    return TickerSnapshot(
        ticker=ticker,
        trailing_pe=forward_pe,
        forward_pe=forward_pe,
        price_to_sales=price_to_sales,
        peg_ratio=peg_ratio,
        total_revenue=1,
        net_income=1,
        profit_margin=0.1,
        revenue_growth=0.1,
        return_on_equity=0.1,
        market_cap=1,
    )


def test_compute_node_valuation_uses_median():
    snapshots = {
        "A": _snapshot("A", 10, 2, 1.0),
        "B": _snapshot("B", 20, 4, 2.0),
        "C": _snapshot("C", 30, 6, 3.0),
    }
    nv = compute_node_valuation("node1", "Node 1", snapshots)
    assert nv.median_forward_pe == 20
    assert nv.median_price_to_sales == 4
    assert nv.median_peg == 2.0
    assert nv.n_tickers == 3


def test_score_undervaluation_ranks_cheaper_node_lower():
    cheap = NodeValuation(
        "cheap", "Cheap", median_forward_pe=10, median_price_to_sales=2, median_peg=1.0, n_tickers=3
    )
    expensive = NodeValuation(
        "expensive",
        "Expensive",
        median_forward_pe=40,
        median_price_to_sales=10,
        median_peg=4.0,
        n_tickers=3,
    )

    result = score_undervaluation([cheap, expensive])
    scores = {nv.node_id: nv.undervaluation_score for nv in result}

    assert scores["cheap"] < scores["expensive"]


def test_score_undervaluation_handles_missing_metrics():
    nv1 = NodeValuation(
        "a", "A", median_forward_pe=None, median_price_to_sales=None, median_peg=None, n_tickers=0
    )
    result = score_undervaluation([nv1])
    assert result[0].undervaluation_score is None


def test_append_history_writes_header_once(tmp_path: Path):
    history_path = tmp_path / "history.csv"
    nv = NodeValuation(
        "a", "A", median_forward_pe=10, median_price_to_sales=2, median_peg=1.0, n_tickers=1
    )
    nv.undervaluation_score = 0.3

    append_history([nv], history_path)
    append_history([nv], history_path)

    lines = history_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("date,node_id")
    assert len(lines) == 3  # header + 2 rows
