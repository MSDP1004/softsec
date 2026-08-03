from src.data_fetcher import TickerSnapshot
from src.shadow_features import analyst_upside, compute_shadow_values, week52_position


def _snapshot(**overrides) -> TickerSnapshot:
    base = dict(
        ticker="TEST",
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
        price=100.0,
    )
    base.update(overrides)
    return TickerSnapshot(**base)


def test_analyst_upside_computes_ratio():
    snap = _snapshot(analyst_target_mean=110.0)
    assert abs(analyst_upside(snap) - 0.1) < 1e-9


def test_analyst_upside_none_when_missing_target():
    snap = _snapshot(analyst_target_mean=None)
    assert analyst_upside(snap) is None


def test_week52_position_midpoint():
    snap = _snapshot(price=150.0, week52_low=100.0, week52_high=200.0)
    assert abs(week52_position(snap) - 0.5) < 1e-9


def test_week52_position_at_high():
    snap = _snapshot(price=200.0, week52_low=100.0, week52_high=200.0)
    assert abs(week52_position(snap) - 1.0) < 1e-9


def test_week52_position_none_when_invalid_range():
    snap = _snapshot(price=150.0, week52_low=200.0, week52_high=100.0)
    assert week52_position(snap) is None


def test_compute_shadow_values_only_includes_requested_active_ids():
    snap = _snapshot(analyst_target_mean=120.0, week52_low=50.0, week52_high=150.0)
    values = compute_shadow_values(snap, active_ids={"analyst_upside"})
    assert set(values.keys()) == {"analyst_upside"}


def test_compute_shadow_values_skips_none_results():
    snap = _snapshot(analyst_target_mean=None, week52_low=50.0, week52_high=150.0)
    values = compute_shadow_values(snap, active_ids={"analyst_upside", "week52_position"})
    assert "analyst_upside" not in values
    assert "week52_position" in values


def test_compute_shadow_values_ignores_unregistered_ids():
    snap = _snapshot()
    values = compute_shadow_values(snap, active_ids={"nonexistent_feature"})
    assert values == {}
