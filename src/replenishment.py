"""
replenishment.py
Store Replenishment Planner and Automated Purchase Order Generator.
Translates weekly SKU demand forecasts into actionable inventory decisions,
computing dynamic safety stock, reorder points (ROP), order-up-to levels,
and purchase order quantities with service level, MOQ, and perishability constraints.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

# Standard Normal Z-table values for common retail target cycle service levels
Z_SCORES = {
    0.80: 0.842,
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326
}


def get_z_score(service_level: float) -> float:
    """Returns standard normal z-score for a given service level."""
    if service_level in Z_SCORES:
        return Z_SCORES[service_level]
    # Simple rational approximation for normal quantile
    p = min(max(service_level, 0.5), 0.999)
    # Norm sin inverse approximation
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - ((c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t))
    return float(z)


class StoreReplenishmentPlanner:
    """
    Translates Random Forest SKU demand forecasts into actionable store replenishment schedules.
    """

    def __init__(
        self,
        default_lead_time_days: int = 3,
        default_review_period_days: int = 7,
        default_service_level: float = 0.95,
        default_case_pack_size: int = 6
    ):
        self.default_lead_time_days = default_lead_time_days
        self.default_review_period_days = default_review_period_days
        self.default_service_level = default_service_level
        self.default_case_pack_size = default_case_pack_size

    def calculate_sku_replenishment(
        self,
        sku_id: str,
        store_id: str,
        forecasted_demand: float,
        forecast_rmse: float,
        current_on_hand: int,
        in_transit: int = 0,
        unit_cost: float = 2.50,
        unit_price: float = 4.29,
        shelf_life_days: int = 30,
        lead_time_days: Optional[int] = None,
        service_level: Optional[float] = None,
        case_pack: Optional[int] = None,
        min_order_qty: int = 0
    ) -> Dict[str, Union[str, float, int, bool]]:
        """
        Calculates replenishment parameters for a single Store-SKU pair.
        """
        lead_time = lead_time_days or self.default_lead_time_days
        srv_level = service_level or self.default_service_level
        case_size = case_pack or self.default_case_pack_size
        review_days = self.default_review_period_days

        # Ensure forecast is non-negative
        forecast = max(0.0, float(forecasted_demand))
        rmse = max(1.0, float(forecast_rmse))

        # Z-score for target service level
        z = get_z_score(srv_level)

        # Protection period in weeks (Lead Time + Review Period)
        protection_interval_weeks = (lead_time + review_days) / 7.0
        lead_time_weeks = lead_time / 7.0

        # Expected demand during supplier lead time
        lead_time_demand = forecast * lead_time_weeks

        # Dynamic Safety Stock (SS) based on model uncertainty (RMSE) and protection interval
        safety_stock = z * rmse * math.sqrt(protection_interval_weeks)
        safety_stock = int(math.ceil(safety_stock))

        # Reorder Point (ROP) = Lead Time Demand + Safety Stock
        reorder_point = int(math.ceil(lead_time_demand + safety_stock))

        # Order-Up-To Target (S) = Expected demand during (L + R) + Safety Stock
        order_up_to_target = int(math.ceil((forecast * protection_interval_weeks) + safety_stock))

        # Current Net Inventory
        net_inventory = current_on_hand + in_transit

        # Reorder Trigger: Is net inventory at or below ROP?
        needs_reorder = net_inventory <= reorder_point

        # Compute Recommended Order Quantity (ROQ)
        if needs_reorder:
            raw_order = max(0, order_up_to_target - net_inventory)
            raw_order = max(raw_order, min_order_qty)
            # Round up to nearest case pack size
            recommended_order_qty = int(math.ceil(raw_order / case_size) * case_size) if case_size > 1 else raw_order
        else:
            recommended_order_qty = 0

        # Stockout Risk: Can current on-hand satisfy lead time demand?
        imminent_stockout = current_on_hand < lead_time_demand

        # Perishability / Spoilage Risk Check
        # If total stock (on hand + order) exceeds expected demand over shelf life window
        weeks_of_shelf_life = max(1.0, shelf_life_days / 7.0)
        max_digestible_stock = forecast * weeks_of_shelf_life
        potential_overstock = (net_inventory + recommended_order_qty) > max_digestible_stock
        perishability_risk = potential_overstock and (shelf_life_days <= 14)

        # Financial Calculations
        order_cost = round(recommended_order_qty * unit_cost, 2)
        expected_revenue = round(recommended_order_qty * unit_price, 2)
        expected_gross_margin = round(expected_revenue - order_cost, 2)

        return {
            "sku_id": sku_id,
            "store_id": store_id,
            "forecasted_weekly_demand": round(forecast, 1),
            "forecast_rmse": round(rmse, 2),
            "current_on_hand": int(current_on_hand),
            "in_transit": int(in_transit),
            "net_inventory": int(net_inventory),
            "lead_time_days": lead_time,
            "service_level": round(srv_level, 2),
            "lead_time_demand": round(lead_time_demand, 1),
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "order_up_to_target": order_up_to_target,
            "needs_reorder": needs_reorder,
            "recommended_order_qty": recommended_order_qty,
            "imminent_stockout_risk": imminent_stockout,
            "perishability_risk": perishability_risk,
            "unit_cost": unit_cost,
            "unit_price": unit_price,
            "order_total_cost": order_cost,
            "expected_revenue": expected_revenue,
            "expected_margin": expected_gross_margin
        }

    def generate_plan_for_dataset(
        self,
        predictions_df: pd.DataFrame,
        sku_catalog: pd.DataFrame,
        default_stock_ratio: float = 0.5,
        service_level: float = 0.95,
        lead_time_days: int = 3
    ) -> pd.DataFrame:
        """
        Generates an actionable replenishment schedule across all predicted store-SKU rows.
        """
        sku_lookup = sku_catalog.set_index("sku_id").to_dict(orient="index")
        plan_rows = []

        # Global or group RMSE estimate
        overall_rmse = 12.5
        if "residual" in predictions_df.columns:
            overall_rmse = float(np.std(predictions_df["residual"].dropna()))
            if overall_rmse < 1.0:
                overall_rmse = 8.0

        for _, row in predictions_df.iterrows():
            sku_id = str(row["sku_id"])
            store_id = str(row["store_id"])
            forecast_val = float(row.get("rf_forecast", row.get("target_next_week_demand", 50.0)))
            
            sku_info = sku_lookup.get(sku_id, {})
            unit_cost = float(sku_info.get("unit_cost", 2.50))
            unit_price = float(sku_info.get("base_price", 4.00))
            shelf_life = int(sku_info.get("shelf_life_days", 30))
            sku_name = sku_info.get("sku_name", sku_id)
            category = sku_info.get("category", "General")

            # Simulate current inventory on hand if not explicitly given
            # e.g., ratio of weekly forecast to test ordering logic
            simulated_on_hand = int(max(5, round(forecast_val * default_stock_ratio)))

            item_plan = self.calculate_sku_replenishment(
                sku_id=sku_id,
                store_id=store_id,
                forecasted_demand=forecast_val,
                forecast_rmse=overall_rmse,
                current_on_hand=simulated_on_hand,
                in_transit=0,
                unit_cost=unit_cost,
                unit_price=unit_price,
                shelf_life_days=shelf_life,
                lead_time_days=lead_time_days,
                service_level=service_level
            )
            item_plan["sku_name"] = sku_name
            item_plan["category"] = category
            plan_rows.append(item_plan)

        df_plan = pd.DataFrame(plan_rows)
        return df_plan
