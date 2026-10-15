"""
bot.py — Telegram-бот прогноза продаж для сети кафе.

Запуск:
    BOT_TOKEN=<token> python bot.py
или через .env-файл (см. .env.example).

Артефакты в artifacts/:
    forecast_dish.csv, forecast_total.csv, price_map.pkl, meta.json
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from predictor import CafePredictor


# ---------- конфигурация ----------
load_dotenv()  # подхватит .env, если есть

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
BOT_TOKEN = os.environ.get("BOT_TOKEN")

if not BOT_TOKEN:
    raise RuntimeError(
        "Не задан BOT_TOKEN. Передайте его через переменную окружения:\n"
        "    BOT_TOKEN=1234:ABC python bot.py\n"
        "или создайте .env с строкой BOT_TOKEN=1234:ABC"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("cafe-bot")


# ---------- загрузка артефактов ----------
log.info(f"Загружаю артефакты из {ARTIFACTS_DIR}")
predictor = CafePredictor(str(ARTIFACTS_DIR))
_h = predictor.horizon_info()
log.info(f"   горизонт: {_h['last_train_date']} → {_h['forecast_end_date']}")
log.info(f"   предприятия: {predictor.available_enterprises()}")


# ---------- утилиты форматирования ----------
def fmt_money(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") + " ₽"


def fmt_qty(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") + " шт."


def fmt_date(d) -> str:
    return d.strftime("%d.%m.%Y")


def parse_date(token: str) -> "datetime.date":
    """Принимает '25.09.2025', '2025-09-25', '25/09/2025', '25-09-2025'."""
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Не понял дату: «{token}». Форматы: 25.09.2025 или 2025-09-25"
    )


def default_date():
    """Первая доступная дата прогноза (день после последнего обучения)."""
    return (predictor.last_train_date + timedelta(days=1)).date()


def fmt_enterprise(e: Optional[str]) -> str:
    return e if e else "вся сеть"


# ---------- клавиатуры ----------
def main_keyboard() -> InlineKeyboardMarkup:
    rows = []
    for ent in predictor.available_enterprises():
        rows.append([InlineKeyboardButton(
            text=f"🍽 Прогноз: {ent}",
            callback_data=f"fc:{ent}",
        )])
    rows.append([InlineKeyboardButton(
        text="🌐 Прогноз по всей сети",
        callback_data="fc:__all__",
    )])
    rows.append([InlineKeyboardButton(
        text="📅 Доступные даты",
        callback_data="info:horizon",
    )])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# ---------- рендер ответа ----------
def render_forecast(target_date, enterprise: Optional[str], top_n: int = 10) -> str:
    r = predictor.predict_total(target_date, enterprise, top_n=top_n)

    header = (
        f"📅 Прогноз на <b>{fmt_date(r.target_date)}</b>\n"
        f"🏢 <b>{r.enterprise}</b>\n\n"
        f"📦 Всего: <b>{fmt_qty(r.total_qty)}</b>\n"
        f"💵 Выручка: <b>{fmt_money(r.total_revenue)}</b>\n\n"
        f"🍽 <b>Топ-{top_n} блюд:</b>\n"
    )

    lines = []
    for i, row in r.dishes.head(top_n).iterrows():
        lines.append(
            f"{i + 1}. {row['Блюдо_canonical']} — "
            f"{row['forecast']:.0f} шт · {fmt_money(row['revenue'])}"
        )
    return header + "\n".join(lines)


# ---------- dispatcher ----------
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(msg: Message):
    text = (
        "👋 Привет! Я бот прогноза продаж сети кафе.\n\n"
        "<b>Что я умею:</b>\n"
        "/forecast <i>дата</i> [предприятие] — прогноз\n"
        "/network <i>дата</i> — прогноз по всей сети\n"
        "/top <i>дата</i> <i>предприятие</i> [N] — топ-N блюд\n"
        "/dates — какие даты доступны\n\n"
        "<b>Пример:</b>\n"
        "<code>/forecast 25.09.2025 Тетрис</code>\n\n"
        f"Доступные предприятия: {', '.join(predictor.available_enterprises())}"
    )
    await msg.answer(text, reply_markup=main_keyboard())


@dp.message(Command("dates"))
async def cmd_dates(msg: Message):
    h = predictor.horizon_info()
    await msg.answer(
        f"📅 Прогноз построен с <b>{fmt_date(predictor.last_train_date.date() + timedelta(days=1))}</b> "
        f"по <b>{fmt_date(datetime.strptime(h['forecast_end_date'], '%Y-%m-%d').date())}</b>.\n\n"
        f"Всего дней: <b>{h['horizon_days']}</b>.\n"
        f"Последняя дата обучения: {fmt_date(datetime.strptime(h['last_train_date'], '%Y-%m-%d').date())}"
    )


async def _send_forecast(msg: Message, d, enterprise, top_n: int = 10):
    try:
        text = render_forecast(d, enterprise, top_n=top_n)
    except ValueError as e:
        await msg.answer(f"⚠️ {e}")
        return
    except Exception as e:
        log.exception("Ошибка прогноза")
        await msg.answer(f"⚠️ Не удалось построить прогноз: {e}")
        return
    await msg.answer(text, reply_markup=main_keyboard())


@dp.message(Command("forecast"))
async def cmd_forecast(msg: Message, command: CommandObject):
    args = (command.args or "").split()
    if len(args) < 1:
        await msg.answer(
            "Использование: <code>/forecast &lt;дата&gt; [предприятие]</code>\n"
            "Пример: <code>/forecast 25.09.2025 Тетрис</code>"
        )
        return
    try:
        d = parse_date(args[0])
    except ValueError as e:
        await msg.answer(str(e))
        return
    enterprise = " ".join(args[1:]) if len(args) > 1 else None
    if enterprise and enterprise not in predictor.available_enterprises():
        await msg.answer(
            f"❓ Не знаю предприятие <b>{enterprise}</b>.\n"
            f"Доступны: {', '.join(predictor.available_enterprises())}"
        )
        return
    await _send_forecast(msg, d, enterprise)


@dp.message(Command("network"))
async def cmd_network(msg: Message, command: CommandObject):
    args = (command.args or "").split()
    if not args:
        await msg.answer(
            "Использование: <code>/network &lt;дата&gt;</code>\n"
            "Пример: <code>/network 25.09.2025</code>"
        )
        return
    try:
        d = parse_date(args[0])
    except ValueError as e:
        await msg.answer(str(e))
        return
    await _send_forecast(msg, d, None)


@dp.message(Command("top"))
async def cmd_top(msg: Message, command: CommandObject):
    args = (command.args or "").split()
    if len(args) < 2:
        await msg.answer(
            "Использование: <code>/top &lt;дата&gt; &lt;предприятие&gt; [N]</code>\n"
            "Пример: <code>/top 25.09.2025 Тетрис 5</code>"
        )
        return
    try:
        d = parse_date(args[0])
    except ValueError as e:
        await msg.answer(str(e))
        return
    enterprise = args[1]
    if enterprise not in predictor.available_enterprises():
        await msg.answer(
            f"❓ Не знаю предприятие <b>{enterprise}</b>.\n"
            f"Доступны: {', '.join(predictor.available_enterprises())}"
        )
        return
    try:
        top_n = int(args[2]) if len(args) > 2 else 10
        top_n = max(1, min(top_n, 30))
    except ValueError:
        top_n = 10
    await _send_forecast(msg, d, enterprise, top_n=top_n)


# ---------- callbacks ----------
@dp.callback_query(F.data == "info:horizon")
async def cb_horizon(call: CallbackQuery):
    await call.answer()
    h = predictor.horizon_info()
    await call.message.answer(
        f"📅 Прогноз доступен с <b>{h['last_train_date']}</b> "
        f"по <b>{h['forecast_end_date']}</b>."
    )


@dp.callback_query(F.data.startswith("fc:"))
async def cb_forecast(call: CallbackQuery):
    await call.answer()
    ent = call.data.split(":", 1)[1]
    enterprise = None if ent == "__all__" else ent
    d = default_date()
    try:
        text = render_forecast(d, enterprise, top_n=10)
    except Exception as e:
        log.exception("Ошибка прогноза по кнопке")
        await call.message.answer(f"⚠️ {e}")
        return
    await call.message.answer(text)


# ---------- main ----------
async def main():
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        log.info("Бот запущен, polling…")
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
