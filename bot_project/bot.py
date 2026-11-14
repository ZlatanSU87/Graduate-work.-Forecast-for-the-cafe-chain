import asyncio
import json
import logging
import os
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from predictor import CafePredictor

load_dotenv()

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
BOT_TOKEN = os.environ.get("BOT_TOKEN")
SA_JSON_PATH = Path(__file__).parent / "service_account.json" # <--- Читаем из файла

if not BOT_TOKEN:
    raise RuntimeError("Не задан BOT_TOKEN.")
if not SA_JSON_PATH.exists():
    raise RuntimeError("Не найден файл service_account.json!")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("cafe-bot")

def download_artifacts_from_drive():
    try:
        # ⚠️ ИЗМЕНЕНИЕ: Читаем JSON из файла
        with open(SA_JSON_PATH, 'r', encoding='utf-8') as f:
            sa_json = json.load(f)

        creds = service_account.Credentials.from_service_account_info(
            sa_json, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)

        # Ищем папку artifacts
        res = service.files().list(
            q="name='artifacts' and mimeType='application/vnd.google-apps.folder' and trashed=false",
            fields="files(id)"
        ).execute()
        folders = res.get('files', [])
        if not folders:
            log.error("Папка artifacts не найдена на Диске.")
            return False
        folder_id = folders[0]['id']

        files_to_get = ['forecast_dish.csv', 'forecast_total.csv', 'price_map.pkl', 'meta.json']
        
        for fname in files_to_get:
            f_res = service.files().list(
                q=f"name='{fname}' and '{folder_id}' in parents and trashed=false",
                fields="files(id)"
            ).execute().get('files', [])
            
            if not f_res:
                continue
                
            file_id = f_res[0]['id']
            req = service.files().get_media(fileId=file_id)
            
            local_path = ARTIFACTS_DIR / fname
            with io.FileIO(local_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, req)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
        return True
    except Exception as e:
        log.error(f"Ошибка скачивания артефактов: {e}")
        return False
        
log.info("Скачиваем актуальные артефакты с Google Drive...")
if not download_artifacts_from_drive():
    log.warning("Не удалось скачать свежие артефакты.")

try:
    predictor = CafePredictor(str(ARTIFACTS_DIR))
    log.info(f"   горизонт: {predictor.horizon_info()['last_train_date']} -> {predictor.horizon_info()['forecast_end_date']}")
except Exception as e:
    log.error(f"Ошибка загрузки предиктора: {e}")
    predictor = None

def fmt_money(v: float) -> str: return f"{v:,.0f}".replace(",", " ") + " ₽"
def fmt_qty(v: float) -> str: return f"{v:,.0f}".replace(",", " ") + " шт."
def fmt_date(d) -> str: return d.strftime("%d.%m.%Y")
def parse_date(token: str) -> "datetime.date":
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try: return datetime.strptime(token, fmt).date()
        except ValueError: continue
    raise ValueError(f"Не понял дату: «{token}».")
def default_date(): return (predictor.last_train_date + timedelta(days=1)).date()

def main_keyboard() -> InlineKeyboardMarkup:
    rows = []
    for ent in predictor.available_enterprises():
        rows.append([InlineKeyboardButton(text=f"🍽 Прогноз: {ent}", callback_data=f"fc:{ent}")])
    rows.append([InlineKeyboardButton(text="🌐 Прогноз по всей сети", callback_data="fc:__all__")])
    rows.append([InlineKeyboardButton(text="📅 Доступные даты", callback_data="info:horizon")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

def render_forecast(target_date, enterprise: Optional[str], top_n: int = 10) -> str:
    r = predictor.predict_total(target_date, enterprise, top_n=top_n)
    header = (
        f"📅 Прогноз на <b>{fmt_date(r.target_date)}</b>\\n"
        f"🏢 <b>{r.enterprise}</b>\\n\\n"
        f"📦 Общий объём: <b>{fmt_qty(r.total_qty)}</b>\\n"
        f"💵 Общая выручка: <b>{fmt_money(r.total_revenue)}</b>\\n\\n"
        f"<i>Топ-{top_n} ключевых блюд (по выручке, без модификаторов):</i>\\n"
    )
    lines = []
    for i, row in r.dishes.head(top_n).iterrows():
        lines.append(f"{i + 1}. {row['Блюдо_canonical']} — {row['forecast']:.0f} шт · {fmt_money(row['revenue'])}")
    return header + "\\n".join(lines)

async def update_artifacts_loop():
    global predictor
    while True:
        await asyncio.sleep(3600)
        log.info("Проверяем обновления...")
        if download_artifacts_from_drive():
            try:
                predictor = CafePredictor(str(ARTIFACTS_DIR))
                log.info("✅ Предиктор перезагружен.")
            except Exception as e:
                log.error(f"Ошибка: {e}")

dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(msg: Message):
    mape = predictor.mape_info()
    mape_text = ""
    if mape and mape.get("mape_median"):
        mape_text = (f"\\n\\n📊 <b>Точность:</b> Медиана {mape['mape_median']:.1f}%. В ±10%: {mape.get('coverage', {}).get('within_10pct', 0):.0f}% дней.")
    text = (
        "👋 Привет! Я бот прогноза продаж.\\n\\n"
        "/forecast <i>дата</i> [предприятие] — прогноз\\n"
        "/accuracy — точность\\n\\n"
        f"Предприятия: {', '.join(predictor.available_enterprises())}" + mape_text
    )
    await msg.answer(text, reply_markup=main_keyboard())

@dp.message(Command("accuracy"))
async def cmd_accuracy(msg: Message):
    mape = predictor.mape_info()
    if not mape.get("mape_median"):
        await msg.answer("⚠️ Метрики не найдены.")
        return
    by_h = mape.get("mape_by_horizon", {})
    cov = mape.get("coverage", {})
    lines = [
        "📊 <b>Точность модели</b>\\n",
        f"Медианная ошибка: <b>{mape['mape_median']:.2f}%</b>",
        f"В ±5%: {cov.get('within_5pct', 0):.0f}%",
        f"В ±10%: {cov.get('within_10pct', 0):.0f}%",
        f"В ±20%: {cov.get('within_20pct', 0):.0f}%",
        f"MAPE 1 день: {by_h.get('1d', 0):.2f}%",
        f"MAPE 30 дней: {by_h.get('30d', 0):.2f}%",
    ]
    await msg.answer("\\n".join(lines))

async def _send_forecast(msg: Message, d, enterprise, top_n: int = 10):
    try: text = render_forecast(d, enterprise, top_n=top_n)
    except Exception as e: await msg.answer(f"⚠️ {e}"); return
    await msg.answer(text, reply_markup=main_keyboard())

@dp.message(Command("forecast"))
async def cmd_forecast(msg: Message, command: CommandObject):
    args = (command.args or "").split()
    if not args: await msg.answer("Пример: /forecast 03.11.2026 Свердлова"); return
    try: d = parse_date(args[0])
    except ValueError as e: await msg.answer(str(e)); return
    ent = " ".join(args[1:]) if len(args) > 1 else None
    await _send_forecast(msg, d, ent)

@dp.callback_query(F.data.startswith("fc:"))
async def cb_forecast(call: CallbackQuery):
    await call.answer()
    ent = call.data.split(":", 1)[1]
    enterprise = None if ent == "__all__" else ent
    d = default_date()
    try: text = render_forecast(d, enterprise)
    except Exception as e: await call.message.answer(f"⚠️ {e}"); return
    await call.message.answer(text)

async def main():
    if predictor is None: return
    asyncio.create_task(update_artifacts_loop())
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        log.info("Бот запущен!")
        await dp.start_polling(bot)
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
