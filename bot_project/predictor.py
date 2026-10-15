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
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd


# ---------- вспомогательные структуры ----------

@dataclass
class TotalPrediction:
    target_date:  date
    enterprise:   str
    total_qty:    float
    total_revenue: float
    dishes:       pd.DataFrame  # [Блюдо_canonical, forecast, price, revenue]

    def pretty_qty(self) -> str:
        return f"{self.total_qty:,.0f}".replace(",", " ")

    def pretty_revenue(self) -> str:
        return f"{self.total_revenue:,.0f}".replace(",", " ")


# ---------- основной класс ----------

class CafePredictor:
    """Прогноз на 30 дней вперёд по кэшированным артефактам."""

    def __init__(self, artifacts_dir: str):
        self.dir = artifacts_dir
        self._load()

    # ---------------- загрузка ----------------

    def _load(self):
        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Не найден {meta_path}. Убедитесь, что артефакты "
                f"скопированы в {self.dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.last_train_date = pd.Timestamp(self.meta["last_train_date"]).normalize()
        self.horizon = int(self.meta["horizon"])
        self.enterprises = list(self.meta.get("enterprises", []))
        self.forecast_end_date = (self.last_train_date
                                  + pd.Timedelta(days=self.horizon)).date()

        self.forecast_dish = pd.read_csv(
            os.path.join(self.dir, "forecast_dish.csv"), parse_dates=["date"])
        self.forecast_total = pd.read_csv(
            os.path.join(self.dir, "forecast_total.csv"), parse_dates=["date"])
        self.price_map = joblib.load(os.path.join(self.dir, "price_map.pkl"))

        # гарантируем нормализованные даты
        self.forecast_dish["date"] = self.forecast_dish["date"].dt.normalize()
        self.forecast_total["date"] = self.forecast_total["date"].dt.normalize()

        # кэш цен на каждый dish для fallback'а
        self._dish_price_cache: dict[str, float] = {}

    # ---------------- утилиты ----------------

    def _check_date(self, target_date) -> pd.Timestamp:
        d = pd.Timestamp(target_date).normalize()
        if d <= self.last_train_date:
            raise ValueError(
                f"Дата {d.date()} должна быть позже "
                f"{self.last_train_date.date()} (последняя дата обучения)")
        if d > pd.Timestamp(self.forecast_end_date):
            raise ValueError(
                f"Дата {d.date()} вне горизонта. "
                f"Прогноз доступен до {self.forecast_end_date}")
        return d

    def _get_price(self, dish: str, enterprise: Optional[str] = None) -> float:
        idx = self.price_map.index
        if enterprise is not None and (dish, enterprise) in idx:
            return float(self.price_map[(dish, enterprise)])
        # fallback 1: медиана цены блюда по всем предприятиям
        if dish not in self._dish_price_cache:
            sub = self.price_map[idx.get_level_values("Блюдо_canonical") == dish]
            self._dish_price_cache[dish] = (
                float(sub.median()) if len(sub) else float(self.price_map.median()))
        return self._dish_price_cache[dish]

    # ---------------- публичный API ----------------

    def available_enterprises(self) -> list[str]:
        return sorted(self.enterprises)

    def horizon_info(self) -> dict:
        return {
            "last_train_date":  str(self.last_train_date.date()),
            "forecast_end_date": str(self.forecast_end_date),
            "horizon_days":     self.horizon,
        }

    def predict_dishes(
        self,
        target_date,
        enterprise: Optional[str] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Возвращает DataFrame с топ-N блюд:
            Блюдо_canonical, forecast, price, revenue
        Если enterprise=None — суммирует по всей сети.
        """
        d = self._check_date(target_date)
        fc = self.forecast_dish[self.forecast_dish["date"] == d]
        if enterprise:
            fc = fc[fc["Торговое предприятие"] == enterprise]
        if fc.empty:
            return pd.DataFrame(
                columns=["Блюдо_canonical", "forecast", "price", "revenue"])

        # суммируем прогнозы по блюду (если несколько предприятий)
        agg = fc.groupby("Блюдо_canonical", as_index=False)["forecast"].sum()

        # цена: при enterprise=... — точно по предприятию;
        # при enterprise=None — средняя цена по тем предприятиям, где продаётся
        if enterprise:
            agg["price"] = agg["Блюдо_canonical"].map(
                lambda x: self._get_price(x, enterprise))
        else:
            ent_map = (fc.groupby("Блюдо_canonical")["Торговое предприятие"]
                         .apply(list).to_dict())
            agg["price"] = agg["Блюдо_canonical"].map(
                lambda x: float(np.mean(
                    [self._get_price(x, e) for e in ent_map.get(x, [])]
                )) if ent_map.get(x) else float(self.price_map.median()))

        agg["revenue"] = agg["forecast"] * agg["price"]
        agg = (agg.sort_values("forecast", ascending=False)
                  .head(top_n)
                  .reset_index(drop=True))
        return agg

    def predict_total(
        self,
        target_date,
        enterprise: Optional[str] = None,
        top_n: int = 10,
    ) -> TotalPrediction:
        """
        Итог по дню. quantity берётся из total-модели,
        revenue — сумма произведений dish-прогнозов на цены.
        """
        d = self._check_date(target_date)
        ft = self.forecast_total[self.forecast_total["date"] == d]
        if enterprise:
            ft = ft[ft["Торговое предприятие"] == enterprise]
        total_qty = float(ft["forecast_qty"].sum())

        dishes = self.predict_dishes(target_date, enterprise,
                                     top_n=len(self.forecast_dish))
        total_revenue = float(dishes["revenue"].sum())

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
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD (по умолчанию — середина горизонта)")
    parser.add_argument("--enterprise", default=None)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    p = CafePredictor(args.dir)
    print("📦 Загружено:", p.horizon_info())
    print("🏢 Предприятия:", p.available_enterprises())

    if args.date:
        d = args.date
    else:
        d = str(p.last_train_date.date() + pd.Timedelta(days=7))

    print(f"\n=== Прогноз на {d} ===")
    r = p.predict_total(d, args.enterprise, top_n=args.top)
    print(f"Точка: {r.enterprise}")
    print(f"Всего: {r.pretty_qty()} шт.  |  {r.pretty_revenue()} ₽")
    print("\nТоп блюд:")
    print(r.dishes.to_string(index=False))
