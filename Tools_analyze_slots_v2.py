"""
Универсальный анализ слотов вектора состояния (полная эмуляция системных вызовов)
"""
import json
import time
import numpy as np
import torch
from models.trader_model import trader_model_instance
from models.smart_broker import SmartPortfolioBroker
from core.trading_hours_scheduler import TradingScheduler

model = trader_model_instance

with open("config/settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

# Создаём брокер
scheduler = TradingScheduler()
broker = SmartPortfolioBroker(settings, scheduler)

# Эмулируем системный вызов: получаем тикеры и цены, как в _run_cycle_impl
securities = broker.moex.get_all_securities()
top_tickers = sorted(securities.items(), key=lambda x: x[1].get('volume', 0), reverse=True)[:5]
tickers = [t[0] for t in top_tickers]

# Выбираем два тикера для теста
ticker1 = tickers[0] if len(tickers) > 0 else "SBER"
ticker2 = tickers[1] if len(tickers) > 1 else "VTBR"

# Получаем цены через системный метод
price1 = broker.moex.get_price(ticker1)
price2 = broker.moex.get_price(ticker2)

sec_info1 = securities.get(ticker1, {})
sec_info2 = securities.get(ticker2, {})

# Обновляем историю цен для индикаторов
for p in [price1, price2]:
    if p:
        broker.technical_core.update_price_data(ticker1, p, 1000)

# Симулируем новости для тикеров (если их нет в поиске)
for t in [ticker1, ticker2]:
    ticker_names = broker.rl_config.get('ticker_names', {})
    keywords = ticker_names.get(t, [t.lower()])
    existing_news = broker.news_fetcher.search_news(ticker=t, limit=3, keywords=keywords)
    if not existing_news:
        # Добавляем синтетическую новость через direct_cache
        broker.news_fetcher.news_cache.append({
            'title': f'Новость компании {t}',
            'summary': f'Корпоративные новости {t} за сегодня',
            'sentiment': 0.1,
            'source': 'test',
            'ts': __import__('time').time(),
            'published': __import__('datetime').datetime.now().isoformat(),
            'category': 'general',
            'priority': 5
        })
        broker.news_fetcher.last_fetch_time = __import__('time').time()

print("=" * 70)
print("🔍 УНИВЕРСАЛЬНЫЙ АНАЛИЗ СЛОТОВ (полная эмуляция системы)")
print("=" * 70)
print(f"Тикер 1: {ticker1}, цена: {price1:.2f}₽")
print(f"Тикер 2: {ticker2}, цена: {price2:.2f}₽")

# Базовый вектор — через системный метод _create_initial_state
base_vector = broker._create_initial_state(ticker1, price1, sec_info1).cpu().numpy()
dim = len(base_vector)

# Тестовые векторы
test1 = broker._create_initial_state(ticker1, price1 * 1.05, sec_info1).cpu().numpy()  # Цена +5%
test2 = broker._create_initial_state(ticker1, price1 * 0.95, sec_info1).cpu().numpy()  # Цена -5%
test3 = broker._create_initial_state(ticker2, price2, sec_info2).cpu().numpy()          # Другой тикер

# Анализируем
slot_sources = {i: set() for i in range(dim)}
slot_always_zero = {i: True for i in range(dim)}

tests = [
    ("Цена +5%", test1),
    ("Цена -5%", test2),
    ("Другой тикер", test3),
]

for test_name, test_vector in tests:
    for i in range(dim):
        if abs(base_vector[i] - test_vector[i]) > 1e-8:
            slot_sources[i].add(test_name)
        if abs(test_vector[i]) > 1e-8:
            slot_always_zero[i] = False

# Вывод
print(f"\nРазмерность: {dim}")
print(f"\n{'Индекс':<7} {'Базовое':<14} {'Тип':<22} {'Меняется при'}")
print("-" * 70)

price_dep, ticker_dep, const_count, reserved_count = 0, 0, 0, 0

for i in range(dim):
    sources = slot_sources[i]
    is_zero = slot_always_zero[i]
    base_val = base_vector[i]

    if is_zero:
        slot_type = "🟢 РЕЗЕРВ"
        reserved_count += 1
    elif "Цена" in str(sources):
        slot_type = "💰 Цена"
        price_dep += 1
    elif "тикер" in str(sources).lower():
        slot_type = "📰 Тикер/новости"
        ticker_dep += 1
    elif not sources:
        slot_type = "⚪ КОНСТАНТА"
        const_count += 1
    else:
        slot_type = "🔵 ДРУГОЕ"
        const_count += 1

    sources_str = ", ".join(sorted(sources))[:60] if sources else "не меняется"
    print(f"[{i:<4}] {base_val:>12.6f}  {slot_type:<22} {sources_str}")

print("-" * 70)
print(f"\nСводка:")
print(f"  💰 Цена:              {price_dep} слотов")
print(f"  📰 Тикер/новости:     {ticker_dep} слотов")
print(f"  🟢 РЕЗЕРВ (всегда 0):  {reserved_count} слотов")
print(f"  ⚪ Константы:         {const_count} слотов")
print(f"  ВСЕГО:                {dim}")

reserved_slots = [i for i in range(dim) if slot_always_zero[i]]
print(f"\nРезервные слоты (всегда 0): {reserved_slots}")
print(f"Всего свободно: {len(reserved_slots)} из {dim}")
print(f"\nГотово.")