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
    momentum_5d: float | None = None
    momentum_21d: float | None = None
    analyst_target_mean: float | None = None
    week52_low: float | None = None
    week52_high: float | None = None


def fetch_momentum(ticker: str) -> tuple[float | None, float | None]:
    """최근 5거래일/21거래일(약 1주/1개월) 수익률. 재무지표와 달리 매일 바뀐다."""
    import yfinance as yf

    try:
        hist = yf.Ticker(ticker).history(period="2mo")
    except Exception:
        return None, None

    if hist is None or hist.empty or "Close" not in hist:
        return None, None

    closes = hist["Close"].dropna()
    if len(closes) < 6:
        return None, None

    r5 = float(closes.iloc[-1] / closes.iloc[-6] - 1)
    r21 = float(closes.iloc[-1] / closes.iloc[-22] - 1) if len(closes) >= 22 else None
    return r5, r21


def fetch_snapshot(ticker: str) -> TickerSnapshot | None:
    """단일 티커의 최신 스냅샷을 가져온다. 조회 실패 시 None."""
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        return None

    momentum_5d, momentum_21d = fetch_momentum(ticker)

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
        momentum_5d=momentum_5d,
        momentum_21d=momentum_21d,
        analyst_target_mean=info.get("targetMeanPrice"),
        week52_low=info.get("fiftyTwoWeekLow"),
        week52_high=info.get("fiftyTwoWeekHigh"),
    )


def fetch_snapshots(tickers: list[str]) -> dict[str, TickerSnapshot]:
    """여러 티커의 스냅샷을 가져온다. 실패한 티커는 결과에서 제외."""
    results: dict[str, TickerSnapshot] = {}
    for ticker in tickers:
        snapshot = fetch_snapshot(ticker)
        if snapshot is not None:
            results[ticker] = snapshot
    return results
