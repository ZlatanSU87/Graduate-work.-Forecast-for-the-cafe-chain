"""
predictor.py — интерфейс прогноза для Telegram-бота кафе.

Не загружает ML-модели. Читает заранее посчитанные CSV-прогнозы
и карту цен. Это делает бота лёгким (<50 МБ ОЗУ) и быстрым
(старт <1 сек, ответ <10 мс).

Артефакты в artifacts/:
    forecast_dish.csv   [date, Блюдо_canonical, Торговое предприятие, forecast]
    forecast_total.csv  [date, Торговое предприятие, forecast_qty]
    price_map.pkl       Series[(Блюдо_canonical, Торговое предприятие) → price]
    meta.json           {last_train_date, horizon, enterprises, target, log_target}
"""
"""
predictor.py — интерфейс прогноза для Telegram-бота кафе.

Читает артефакты, не загружает ML-модели. Лёгкий (<50 МБ) и быстрый.

Артефакты в artifacts/:
    forecast_dish.csv           [date, Блюдо_canonical, Торговое предприятие, forecast]
    forecast_total.csv          [date, Торговое предприятие, forecast_qty]
    price_map.pkl               Series[(Блюдо_canonical, Торговое предприятие) → price]
    meta.json                   {..., avg_price_by_enterprise, median_qty_last_7_days}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Optional

import joblib
import numpy as np
import pandas as pd


@dataclass
class TotalPrediction:
    target_date:   date
    enterprise:    str
    total_qty:     float
    total_revenue: float
    dishes:        pd.DataFrame

    def pretty_qty(self) -> str:
        return f"{self.total_qty:,.0f}".replace(",", " ")

    def pretty_revenue(self) -> str:
        return f"{self.total_revenue:,.0f}".replace(",", " ")


class CafePredictor:
    """Прогноз на горизонт по кэшированным артефактам."""

    def __init__(self, artifacts_dir: str):
        self.dir = artifacts_dir
        self._load()

    # ---------------- загрузка ----------------

    def _load(self):
        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Не найден {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.last_train_date   = pd.Timestamp(self.meta["last_train_date"]).normalize()
        self.horizon           = int(self.meta["horizon"])
        self.enterprises       = list(self.meta.get("enterprises", []))
        self.avg_price_map     = self.meta.get("avg_price_by_enterprise", {})
        self.median_qty_map    = self.meta.get("median_qty_last_7_days", {})
        self.forecast_end_date = (self.last_train_date
                                  + pd.Timedelta(days=self.horizon)).date()

        self.forecast_dish = pd.read_csv(
            os.path.join(self.dir, "forecast_dish.csv"), parse_dates=["date"])
        self.forecast_total = pd.read_csv(
            os.path.join(self.dir, "forecast_total.csv"), parse_dates=["date"])
        self.price_map = joblib.load(os.path.join(self.dir, "price_map.pkl"))

        self.forecast_dish["date"]  = self.forecast_dish["date"].dt.normalize()
        self.forecast_total["date"] = self.forecast_total["date"].dt.normalize()

        self._dish_price_cache: dict[str, float] = {}

    # ---------------- утилиты ----------------

    def _check_date(self, target_date) -> pd.Timestamp:
        d = pd.Timestamp(target_date).normalize()
        if d <= self.last_train_date:
            raise ValueError(f"Дата {d.date()} должна быть позже "
                             f"{self.last_train_date.date()}")
        if d > pd.Timestamp(self.forecast_end_date):
            raise ValueError(f"Дата {d.date()} вне горизонта "
                             f"(до {self.forecast_end_date})")
        return d

    def _get_price(self, dish: str, enterprise: Optional[str] = None) -> float:
        idx = self.price_map.index
        if enterprise is not None and (dish, enterprise) in idx:
            return float(self.price_map[(dish, enterprise)])
        if dish not in self._dish_price_cache:
            sub = self.price_map[idx.get_level_values("Блюдо_canonical") == dish]
            self._dish_price_cache[dish] = (
                float(sub.median()) if len(sub) else float(self.price_map.median()))
        return self._dish_price_cache[dish]

    def _get_avg_price(self, enterprise: Optional[str] = None) -> float:
        if enterprise is not None:
            return float(self.avg_price_map.get(enterprise, 110.0))
        if not self.avg_price_map:
            return 110.0
        return float(np.mean(list(self.avg_price_map.values())))

    # ---------------- публичный API ----------------

    def available_enterprises(self) -> list[str]:
        return sorted(self.enterprises)

    def horizon_info(self) -> dict:
        return {
            "last_train_date":   str(self.last_train_date.date()),
            "forecast_end_date": str(self.forecast_end_date),
            "horizon_days":      self.horizon,
        }
    
    def mape_info(self) -> dict:
        """Возвращает MAPE-метрики из meta.json для отображения в боте."""
        return {
            "mape_median":     self.meta.get("mape_median", None),
            "mape_mean":       self.meta.get("mape_mean", None),
            "mape_by_horizon": self.meta.get("mape_by_horizon", {}),
            "coverage":        self.meta.get("coverage", {}),
            "distribution":    self.meta.get("mape_distribution", {}),
        }
        
    def predict_dishes(self, target_date, enterprise=None,
                        top_n=10) -> pd.DataFrame:
        d = self._check_date(target_date)
        fc = self.forecast_dish[self.forecast_dish["date"] == d]
        if enterprise:
            fc = fc[fc["Торговое предприятие"] == enterprise]
        if fc.empty:
            return pd.DataFrame(columns=["Блюдо_canonical", "forecast",
                                          "price", "revenue"])

        agg = fc.groupby("Блюдо_canonical", as_index=False)["forecast"].sum()

        if enterprise:
            agg["price"] = agg["Блюдо_canonical"].map(
                lambda x: self._get_price(x, enterprise))
        else:
            ent_map = (fc.groupby("Блюдо_canonical")["Торговое предприятие"]
                         .apply(list).to_dict())
            agg["price"] = agg["Блюдо_canonical"].map(
                lambda x: float(np.mean([self._get_price(x, e)
                                          for e in ent_map.get(x, [])]))
                          if ent_map.get(x) else float(self.price_map.median()))

        agg["revenue"] = agg["forecast"] * agg["price"]
        return (agg.sort_values("forecast", ascending=False)
                   .head(top_n).reset_index(drop=True))

    def predict_total(self, target_date, enterprise=None,
                       top_n=10) -> TotalPrediction:
        """
        Итог по дню.

        Схема:
          1) total_qty = sum(forecast_qty) из total-модели
          2) Если total_qty < 0.7 × median_qty_last_7_days →
             заменяем на median_qty_last_7_days (защита от схлопывания)
          3) total_revenue = total_qty × avg_price_by_enterprise
          4) dishes масштабируются пропорционально к total_qty
             и total_revenue
        """
        d = self._check_date(target_date)

        # 1) total_qty из модели
        ft = self.forecast_total[self.forecast_total["date"] == d]
        if enterprise:
            ft = ft[ft["Торговое предприятие"] == enterprise]
        total_qty = float(ft["forecast_qty"].sum()) if len(ft) else 0.0

        # 2) Fallback на median_last7 (если модель схлопнулась)
        if enterprise and enterprise in self.median_qty_map:
            med = float(self.median_qty_map[enterprise])
            if total_qty > 0 and med > 0 and total_qty < 0.7 * med:
                total_qty = med
        elif not enterprise and self.median_qty_map:
            med_sum = sum(float(v) for v in self.median_qty_map.values())
            if total_qty > 0 and med_sum > 0 and total_qty < 0.7 * med_sum:
                total_qty = med_sum

        # 3) Avg price и total_revenue
        avg_price = self._get_avg_price(enterprise)
        total_revenue = total_qty * avg_price

        # 4) Dish breakdown
        dishes = self.predict_dishes(target_date, enterprise,
                                       top_n=len(self.forecast_dish))
        dish_qty_sum = float(dishes["forecast"].sum())
        if dish_qty_sum > 0 and total_qty > 0:
            dishes["forecast"] = dishes["forecast"] * (total_qty / dish_qty_sum)
            dishes["revenue"]  = (dishes["forecast"] / total_qty) * total_revenue
            dishes["price"]    = (dishes["revenue"] /
                                    dishes["forecast"].clip(lower=1e-6))
        elif total_qty == 0:
            dishes["revenue"] = 0.0
            dishes["price"]   = 0.0

        return TotalPrediction(
            target_date=d.date(),
            enterprise=enterprise or "вся сеть",
            total_qty=total_qty,
            total_revenue=total_revenue,
            dishes=dishes.head(top_n).reset_index(drop=True),
        )


# ---------------- CLI-тест ----------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="artifacts")
    parser.add_argument("--date", default=None)
    parser.add_argument("--enterprise", default=None)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    p = CafePredictor(args.dir)
    print("📦 Загружено:", p.horizon_info())
    print("🏢 Предприятия:", p.available_enterprises())
    print("💰 Avg price:", {k: f"{v:.1f}" for k, v in p.avg_price_map.items()})
    print("📊 Median last 7d:", p.median_qty_map)

    d = args.date or str(p.last_train_date.date() + pd.Timedelta(days=7))

    print(f"\n=== Прогноз на {d} ===")
    r = p.predict_total(d, args.enterprise, top_n=args.top)
    print(f"Точка: {r.enterprise}")
    print(f"Всего: {r.pretty_qty()} шт.  |  {r.pretty_revenue()} ₽")
    print("\nТоп блюд:")
    print(r.dishes[["Блюдо_canonical", "forecast", "price", "revenue"]]
          .to_string(index=False))
