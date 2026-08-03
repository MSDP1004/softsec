"""yfinance 기반 가격/재무 데이터 수집."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TickerSnapshot:
    ticker: str
    trailing_pe: float | None
    forward_pe: float | None
    price_to_sales: float | None
    peg_ratio: float | None
    total_revenue: float | None
    net_income: float | None
    profit_margin: float | None
    revenue_growth: float | None
    return_on_equity: float | None
    market_cap: float | None
    price: float | None = None


def fetch_snapshot(ticker: str) -> TickerSnapshot | None:
    """단일 티커의 최신 스냅샷을 가져온다. 조회 실패 시 None."""
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        return None

    return TickerSnapshot(
        ticker=ticker,
        trailing_pe=info.get("trailingPE"),
        forward_pe=info.get("forwardPE"),
        price_to_sales=info.get("priceToSalesTrailing12Months"),
        peg_ratio=info.get("trailingPegRatio") or info.get("pegRatio"),
        total_revenue=info.get("totalRevenue"),
        net_income=info.get("netIncomeToCommon"),
        profit_margin=info.get("profitMargins"),
        revenue_growth=info.get("revenueGrowth"),
        return_on_equity=info.get("returnOnEquity"),
        market_cap=info.get("marketCap"),
        price=info.get("currentPrice") or info.get("regularMarketPrice"),
    )


def fetch_snapshots(tickers: list[str]) -> dict[str, TickerSnapshot]:
    """여러 티커의 스냅샷을 가져온다. 실패한 티커는 결과에서 제외."""
    results: dict[str, TickerSnapshot] = {}
    for ticker in tickers:
        snapshot = fetch_snapshot(ticker)
        if snapshot is not None:
            results[ticker] = snapshot
    return results
