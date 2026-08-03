from src.data_fetcher import TickerSnapshot
from src.quality_filter import QualityThresholds, passes_quality_filter


def _snapshot(**overrides):
    base = dict(
        ticker="TEST",
        trailing_pe=20.0,
        forward_pe=18.0,
        price_to_sales=5.0,
        peg_ratio=1.5,
        total_revenue=1_000_000_000,
        net_income=100_000_000,
        profit_margin=0.1,
        revenue_growth=0.2,
        return_on_equity=0.15,
        market_cap=10_000_000_000,
    )
    base.update(overrides)
    return TickerSnapshot(**base)


def test_profitable_company_passes():
    assert passes_quality_filter(_snapshot()) is True


def test_no_revenue_company_fails():
    assert passes_quality_filter(_snapshot(total_revenue=None)) is False


def test_zero_revenue_fails():
    assert passes_quality_filter(_snapshot(total_revenue=0)) is False


def test_negative_net_income_fails():
    assert passes_quality_filter(_snapshot(net_income=-50_000_000)) is False


def test_missing_net_income_fails():
    assert passes_quality_filter(_snapshot(net_income=None)) is False


def test_custom_thresholds_allow_break_even():
    thresholds = QualityThresholds(require_positive_net_income=False)
    assert passes_quality_filter(_snapshot(net_income=0), thresholds) is True
