"""
data_generator.py
Generates realistic Point-of-Sale (POS) transaction logs for multi-store, multi-SKU retail demand forecasting.
Includes temporal dynamics, seasonality, holiday shocks, promotional uplifts, and stockout effects.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Product Catalog Definition
DEFAULT_SKU_CATALOG = [
    {
        "sku_id": "SKU_DAIRY_01",
        "sku_name": "Organic Whole Milk 1 Gal",
        "category": "Dairy",
        "base_price": 4.29,
        "unit_cost": 2.50,
        "is_perishable": 1,
        "shelf_life_days": 12,
        "base_daily_demand": 45,
        "price_elasticity": 1.6
    },
    {
        "sku_id": "SKU_DAIRY_02",
        "sku_name": "Greek Yogurt Vanilla 32oz",
        "category": "Dairy",
        "base_price": 4.99,
        "unit_cost": 2.90,
        "is_perishable": 1,
        "shelf_life_days": 21,
        "base_daily_demand": 30,
        "price_elasticity": 1.4
    },
    {
        "sku_id": "SKU_BEV_01",
        "sku_name": "Cold-Pressed Orange Juice 59oz",
        "category": "Beverages",
        "base_price": 4.79,
        "unit_cost": 2.80,
        "is_perishable": 1,
        "shelf_life_days": 25,
        "base_daily_demand": 35,
        "price_elasticity": 1.8
    },
    {
        "sku_id": "SKU_BEV_02",
        "sku_name": "Sparkling Mineral Water 12-pk",
        "category": "Beverages",
        "base_price": 6.99,
        "unit_cost": 3.80,
        "is_perishable": 0,
        "shelf_life_days": 365,
        "base_daily_demand": 50,
        "price_elasticity": 2.1
    },
    {
        "sku_id": "SKU_BAKE_01",
        "sku_name": "Artisanal Sourdough Loaf",
        "category": "Bakery",
        "base_price": 3.99,
        "unit_cost": 1.80,
        "is_perishable": 1,
        "shelf_life_days": 5,
        "base_daily_demand": 40,
        "price_elasticity": 1.3
    },
    {
        "sku_id": "SKU_SNACK_01",
        "sku_name": "Organic Roasted Sea Salt Almonds 16oz",
        "category": "Snacks",
        "base_price": 7.99,
        "unit_cost": 4.50,
        "is_perishable": 0,
        "shelf_life_days": 180,
        "base_daily_demand": 25,
        "price_elasticity": 1.9
    },
    {
        "sku_id": "SKU_PROD_01",
        "sku_name": "Hass Avocados Bag 4-ct",
        "category": "Produce",
        "base_price": 4.49,
        "unit_cost": 2.60,
        "is_perishable": 1,
        "shelf_life_days": 7,
        "base_daily_demand": 55,
        "price_elasticity": 2.0
    },
    {
        "sku_id": "SKU_PROD_02",
        "sku_name": "Gala Apples Bag 3-lb",
        "category": "Produce",
        "base_price": 5.29,
        "unit_cost": 3.00,
        "is_perishable": 1,
        "shelf_life_days": 14,
        "base_daily_demand": 38,
        "price_elasticity": 1.5
    },
    {
        "sku_id": "SKU_PANTRY_01",
        "sku_name": "Extra Virgin Olive Oil 500ml",
        "category": "Pantry",
        "base_price": 9.99,
        "unit_cost": 5.80,
        "is_perishable": 0,
        "shelf_life_days": 540,
        "base_daily_demand": 20,
        "price_elasticity": 1.2
    },
    {
        "sku_id": "SKU_PANTRY_02",
        "sku_name": "Bronze-Cut Penne Rigate 16oz",
        "category": "Pantry",
        "base_price": 2.49,
        "unit_cost": 1.10,
        "is_perishable": 0,
        "shelf_life_days": 720,
        "base_daily_demand": 60,
        "price_elasticity": 1.7
    }
]

# Store Network Definition
DEFAULT_STORE_CATALOG = [
    {"store_id": "STORE_01", "store_name": "Metro Downtown Supermarket", "store_type": "Supermarket", "traffic_factor": 1.25, "location": "Urban"},
    {"store_id": "STORE_02", "store_name": "Suburban Center Hypermarket", "store_type": "Hypermarket", "traffic_factor": 1.50, "location": "Suburban"},
    {"store_id": "STORE_03", "store_name": "Transit Station Express Mart", "store_type": "Express", "traffic_factor": 0.75, "location": "Urban Transit"},
    {"store_id": "STORE_04", "store_name": "Westside Community Grocery", "store_type": "Supermarket", "traffic_factor": 1.05, "location": "Suburban"}
]

US_HOLIDAYS = [
    # Fixed or approximate annual holiday dates (month, day, name, multiplier)
    (1, 1, "New Year's Day", 1.45),
    (2, 14, "Valentine's Day", 1.25),
    (5, 27, "Memorial Day", 1.35),
    (7, 4, "Independence Day", 1.50),
    (9, 2, "Labor Day", 1.35),
    (10, 31, "Halloween", 1.30),
    (11, 26, "Thanksgiving Week", 1.75),
    (11, 28, "Black Friday", 1.85),
    (12, 24, "Christmas Eve", 1.90),
    (12, 25, "Christmas Day", 0.30), # Store reduced hours or closed
    (12, 31, "New Year's Eve", 1.60)
]


def is_holiday(date: datetime):
    """Checks if a given date matches a holiday window."""
    for m, d, name, mult in US_HOLIDAYS:
        if date.month == m and abs(date.day - d) <= 1:
            return 1, name, mult
    return 0, "Regular", 1.0


def generate_pos_transactions(
    start_date: str = "2024-01-01",
    num_days: int = 730,  # 2 full years (104 weeks)
    random_seed: int = 42,
    output_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Simulates high-fidelity Point-of-Sale transaction records across stores and SKUs.
    Outputs transaction log records and daily aggregates.
    """
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    records = []
    transaction_id_counter = 1000000

    print(f"Generating realistic POS transaction logs from {start_date} for {num_days} days...")

    # Precompute scheduled promotions: each store/SKU pair gets random promo weeks
    promo_schedule = {}
    for store in DEFAULT_STORE_CATALOG:
        for sku in DEFAULT_SKU_CATALOG:
            key = (store["store_id"], sku["sku_id"])
            # Randomly select ~8-12 promotional weeks out of 104 weeks
            promo_weeks = set(np.random.choice(range(1, 105), size=10, replace=False))
            promo_schedule[key] = promo_weeks

    for day_idx in range(num_days):
        current_date = start_dt + timedelta(days=day_idx)
        current_week = (day_idx // 7) + 1
        day_of_week = current_date.weekday() # 0 = Monday, 6 = Sunday
        day_of_year = current_date.timetuple().tm_yday
        is_wknd = 1 if day_of_week in [4, 5, 6] else 0 # Fri, Sat, Sun uplift
        
        # Holiday check
        holiday_flag, holiday_name, holiday_mult = is_holiday(current_date)
        
        # Day of week multiplier (weekend shopping pattern)
        dow_multipliers = [0.85, 0.88, 0.92, 0.98, 1.20, 1.45, 1.30]
        dow_mult = dow_multipliers[day_of_week]
        
        # Annual seasonality (peaking in summer for beverages, winter for pantry/comfort food)
        annual_cycle = np.sin(2 * np.pi * day_of_year / 365.25)

        for store in DEFAULT_STORE_CATALOG:
            store_id = store["store_id"]
            store_traffic = store["traffic_factor"]

            for sku in DEFAULT_SKU_CATALOG:
                sku_id = sku["sku_id"]
                base_demand = sku["base_daily_demand"]
                base_price = sku["base_price"]
                unit_cost = sku["unit_cost"]
                elasticity = sku["price_elasticity"]
                category = sku["category"]

                # Category seasonal impact
                if category == "Beverages":
                    cat_seasonality = 1.0 + 0.25 * annual_cycle  # Peaks in summer
                elif category in ["Pantry", "Bakery"]:
                    cat_seasonality = 1.0 - 0.15 * annual_cycle  # Higher in autumn/winter
                else:
                    cat_seasonality = 1.0 + 0.08 * np.cos(2 * np.pi * day_of_year / 365.25)

                # Promotion impact
                is_promo = 1 if current_week in promo_schedule[(store_id, sku_id)] else 0
                discount_rate = np.random.choice([0.15, 0.20, 0.25, 0.30]) if is_promo else 0.0
                actual_unit_price = round(base_price * (1.0 - discount_rate), 2)

                # Price elasticity demand uplift
                promo_multiplier = (1.0 + (discount_rate * elasticity)) if is_promo else 1.0

                # Mean expected demand for the day
                expected_demand = (
                    base_demand
                    * store_traffic
                    * dow_mult
                    * holiday_mult
                    * cat_seasonality
                    * promo_multiplier
                )

                # Add stochastic Poisson noise
                actual_demand = int(np.random.poisson(max(0.5, expected_demand)))

                # Occasional stockout event (1% probability on non-promos, 3% on high promos)
                stockout_prob = 0.03 if is_promo else 0.01
                had_stockout = np.random.rand() < stockout_prob
                if had_stockout:
                    # Stockout truncates realized demand to a fraction
                    actual_demand = int(actual_demand * np.random.uniform(0.1, 0.5))

                if actual_demand <= 0:
                    continue

                # Generate transactional log entries (simulating individual basket purchases)
                # Split total daily demand into random basket sizes (1 to 4 items per transaction)
                remaining_demand = actual_demand
                while remaining_demand > 0:
                    basket_qty = min(remaining_demand, int(np.random.choice([1, 2, 3, 4], p=[0.55, 0.30, 0.10, 0.05])))
                    remaining_demand -= basket_qty
                    transaction_id_counter += 1

                    # Transaction time between 08:00 and 21:59
                    txn_hour = int(np.random.choice(range(8, 22), p=[
                        0.03, 0.05, 0.07, 0.08, 0.10, 0.10, 0.08, 0.08, 0.09, 0.12, 0.10, 0.06, 0.03, 0.01
                    ]))
                    txn_minute = np.random.randint(0, 60)
                    txn_timestamp = f"{current_date.strftime('%Y-%m-%d')} {txn_hour:02d}:{txn_minute:02d}:00"

                    records.append({
                        "transaction_id": f"TXN_{transaction_id_counter}",
                        "timestamp": txn_timestamp,
                        "date": current_date.strftime("%Y-%m-%d"),
                        "store_id": store_id,
                        "store_name": store["store_name"],
                        "store_type": store["store_type"],
                        "sku_id": sku_id,
                        "sku_name": sku["sku_name"],
                        "category": category,
                        "unit_price": actual_unit_price,
                        "unit_cost": unit_cost,
                        "quantity": basket_qty,
                        "total_sales": round(actual_unit_price * basket_qty, 2),
                        "total_cost": round(unit_cost * basket_qty, 2),
                        "is_perishable": sku.get("is_perishable", 0),
                        "shelf_life_days": sku.get("shelf_life_days", 30),
                        "is_promoted": is_promo,
                        "discount_pct": round(discount_rate * 100, 1),
                        "is_holiday": holiday_flag,
                        "holiday_name": holiday_name,
                        "is_weekend": is_wknd,
                        "stockout_flag": 1 if had_stockout else 0
                    })

    df_pos = pd.DataFrame(records)
    output_path = os.path.join(output_dir, "pos_transactions.csv")
    df_pos.to_csv(output_path, index=False)
    print(f"POS transaction logs generated successfully! Total transactions: {len(df_pos):,} saved to {output_path}")

    # Also save metadata catalogs for lookup
    pd.DataFrame(DEFAULT_SKU_CATALOG).to_csv(os.path.join(output_dir, "sku_catalog.csv"), index=False)
    pd.DataFrame(DEFAULT_STORE_CATALOG).to_csv(os.path.join(output_dir, "store_catalog.csv"), index=False)

    return df_pos


if __name__ == "__main__":
    generate_pos_transactions()
