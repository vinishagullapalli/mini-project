"""
test_replenishment.py
Unit tests for Store Replenishment Planner and inventory optimization formulas.
"""

import pytest
from src.replenishment import StoreReplenishmentPlanner, get_z_score


def test_z_score_lookup():
    """Verifies standard normal z-scores for service levels."""
    assert get_z_score(0.95) == pytest.approx(1.645, 0.01)
    assert get_z_score(0.90) == pytest.approx(1.282, 0.01)
    assert get_z_score(0.98) == pytest.approx(2.054, 0.01)
    assert get_z_score(0.99) == pytest.approx(2.326, 0.01)


def test_replenishment_trigger_and_order():
    """Tests that reorder is triggered when inventory is below ROP and rounded to case pack."""
    planner = StoreReplenishmentPlanner(
        default_lead_time_days=3,
        default_review_period_days=7,
        default_service_level=0.95,
        default_case_pack_size=6
    )

    # Forecast = 70 units/week. Lead time = 3 days => LTD = 70 * (3/7) = 30 units.
    # Protection period = (3 + 7)/7 = 10/7 ~ 1.428 weeks.
    # RMSE = 10. SS = 1.645 * 10 * sqrt(1.428) ~ 19.66 => 20 units.
    # ROP = LTD + SS = 30 + 20 = 50 units.
    # Order-Up-To = (70 * 1.428) + 20 = 100 + 20 = 120 units.
    result = planner.calculate_sku_replenishment(
        sku_id="SKU_TEST_01",
        store_id="STORE_01",
        forecasted_demand=70.0,
        forecast_rmse=10.0,
        current_on_hand=15, # well below ROP (50)
        in_transit=0,
        case_pack=6
    )

    assert result["needs_reorder"] is True
    assert result["reorder_point"] >= 45
    assert result["safety_stock"] >= 15
    # Order quantity must be a multiple of case pack (6)
    assert result["recommended_order_qty"] % 6 == 0
    assert result["recommended_order_qty"] > 0
    # Imminent stockout risk check (on hand 15 < lead time demand 30)
    assert result["imminent_stockout_risk"] is True


def test_healthy_inventory_no_reorder():
    """Tests that no order is triggered when on-hand is comfortably above ROP."""
    planner = StoreReplenishmentPlanner(
        default_lead_time_days=3,
        default_review_period_days=7,
        default_service_level=0.95
    )

    result = planner.calculate_sku_replenishment(
        sku_id="SKU_TEST_02",
        store_id="STORE_01",
        forecasted_demand=50.0,
        forecast_rmse=5.0,
        current_on_hand=100, # Much higher than ROP
        in_transit=0
    )

    assert result["needs_reorder"] is False
    assert result["recommended_order_qty"] == 0
    assert result["imminent_stockout_risk"] is False
