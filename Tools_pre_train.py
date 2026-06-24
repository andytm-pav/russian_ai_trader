"""
Предобучение модели (v2): Seed + История + Валидация + Макро + Режим рынка
"""
import json
import time
import torch
import numpy as np
import random
import pandas as pd
from datetime import datetime, timedelta
from models.trader_model import trader_model_instance
from core.core_technical_trader import TechnicalTraderCore
from fetchers.moex_fetcher import MoexFetcher
from utils.portfolio_manager import PortfolioManager
from utils.logger import get_logger
import logging

logger = get_logger("PRE_TRAIN")

file_handler = logging.FileHandler('logs/pre_train.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)-15s | %(levelname)-8s | %(message)s'))
logger.logger.addHandler(file_handler)

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
with open("config/settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

# 37 тикеров, отобранных брокером
TICKERS = [
    # Нефть и газ (6)
    "GAZP", "LKOH", "ROSN", "SNGS", "TATN", "NVTK",
    # Финансы (5)
    "SBER", "MOEX", "TCSG", "BSPB", "RENI",
    # Металлургия и горнодобыча (6)
    "GMKN", "CHMF", "MAGN", "NLMK", "PLZL", "ALRS",
    # Телеком и технологии (4)
    "MTSS", "RTKM", "YNDX", "VKCO",
    # Потребительский сектор и ритейл (5)
    "MGNT", "DSKY", "AFLT", "FESH", "BELU",
    # Энергетика (4)
    "IRAO", "HYDR", "FEES", "UPRO",
    # Химия и фармацевтика (2)
    "PHOR", "LIFE",
    # Транспорт и промышленность (3)
    "TRNFP", "NMTP", "DELI",
    # ОПК и смежные (2)
    "UNAC", "KMAZ"
]

TRAIN_START = "2025-06-01"
TRAIN_END   = "2026-06-23"   #
VAL_START   = "2025-03-01"
VAL_END     = "2025-05-31"   # последние 3 дня — валидация

print("=" * 70)
print("🧠 ПРЕДОБУЧЕНИЕ МОДЕЛИ (v2 — все улучшения)")
print("=" * 70)
print(f"Тикеров: {len(TICKERS)}")
print(f"Обучение: {TRAIN_START} → {TRAIN_END}")
print(f"Валидация: {VAL_START} → {VAL_END}")

# ============================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================
model = trader_model_instance
moex = MoexFetcher()
tech_core = TechnicalTraderCore()

print(f"Память до: {len(model.memory)} опытов")

# ============================================================
# ЭТАП 1: SEED EXPERIENCES (48 опытов)
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 1: SEED EXPERIENCES (48 опытов)")
print("=" * 70)

def create_base_state(ticker="SBER", price=300, rsi=50, bb_pos=0.5, momentum=0,
                      imoex=2500, imoex_change=0, brent=80, brent_change=0,
                      rvi=25, usd_rub=72, market_regime=0, has_position=False,
                      pnl_pct=0, positions_count=0, cash=10000, exposure=0,
                      hold_time=0, volume=1e9, market_cap=5e12, spread=0.01):
    """Создаёт синтетический вектор состояния"""
    news_features = model.encode_news([f"Новость {ticker}"])

    market_data = {
        'volume': volume, 'spread': spread, 'rsi': rsi,
        'volatility': 0.02, 'sma_10_ratio': 1.0, 'sma_20_ratio': 1.0,
        'bb_position': bb_pos, 'volume_ratio': 1.0, 'atr': price * 0.01,
        'market_cap': market_cap, 'lot_size': 1, 'min_step': 0.01,
        'sector': 'финансы', 'momentum': momentum,
        'imoex': imoex, 'imoex_change': imoex_change,
        'rtsi': 1100, 'rtsi_change': 0,
        'rvi': rvi, 'rvi_change': 0,
        'moexog': 4500, 'moexfn': 8000,
        'brent': brent, 'brent_change': brent_change,
        'market_liquidity_ratio': 0.8, 'market_activity_score': 1.0,
        'spread_pct': spread / price, 'market_mood': 0,
        'shares_turnover': 5e11, 'rvi_normalized': rvi / 100,
        'imoex_normalized': imoex / 4000,
        'market_cap_total': 5e13, 'liquidity_ratio': 0.8,
        'cbr_rate_normalized': 0.675, 'vix': 16, 'moexog_normalized': 0.45,
        'market_regime': market_regime
    }

    sim_portfolio = PortfolioManager()
    sim_portfolio.initial_capital = 10000
    sim_portfolio.max_positions = 10
    sim_portfolio.cash = cash
    sim_portfolio.reserved_cash = 0

    if has_position:
        sim_portfolio.positions[ticker] = {
            'qty': 10, 'avg_price': price * (1 - pnl_pct / 100),
            'buy_time': time.time() - hold_time * 3600,
            'stop_loss': price * 0.95, 'take_profit': price * 1.1,
            'strategy': 'balanced', 'lot_size': 1, 'min_step': 0.01
        }

    state = model.build_state_vector(
        ticker=ticker, price=price, momentum=momentum,
        sentiment=0, news_features=news_features,
        market_data=market_data, market_sentiment=0,
        portfolio=sim_portfolio
    )
    return state

# Загружаем seed-опыты из конфига
with open("config/seed_experiences.json", "r", encoding="utf-8") as f:
    seeds = json.load(f)

seed_params = {
    "Перепроданность на растущем": {"rsi": 25, "bb_pos": 0.1, "imoex_change": 1.5, "market_regime": 1},
    "Перекупленность + прибыль": {"rsi": 78, "bb_pos": 0.9, "has_position": True, "pnl_pct": 4.2},
    "Боковик": {"rsi": 50, "bb_pos": 0.5},
    "Идеальный вход": {"rsi": 22, "bb_pos": 0.05, "momentum": -3, "imoex_change": 2.5, "market_regime": 1},
    "Сильная перекупленность": {"rsi": 85, "bb_pos": 1.0, "has_position": True, "pnl_pct": 8},
    "Падение ускоряется": {"rsi": 28, "momentum": -3},
    "Начало тренда": {"rsi": 55, "bb_pos": 0.6, "momentum": 2},
    "Высокая волатильность": {"rsi": 48, "imoex_change": -2, "market_regime": 2, "volume": 5e7},
    "Сжатие перед движением": {"rsi": 50, "bb_pos": 0.5, "volume": 5e7},
    "Всплеск объёма": {"rsi": 55, "volume": 3e9},
    "Низкий объём при падении": {"rsi": 45, "momentum": -1, "volume": 2e8},
    "Золотой крест SMA": {"rsi": 55, "bb_pos": 0.6, "momentum": 1.5},
    "Сильный позитив": {"rsi": 55},
    "Негатив + позиция в плюсе": {"has_position": True, "pnl_pct": 2, "imoex_change": -2},
    "Разнонаправленные новости": {"rsi": 50},
    "Инсайдерская покупка": {"rsi": 40, "bb_pos": 0.3},
    "Покупка на слухах": {"rsi": 85, "bb_pos": 0.95, "has_position": True, "pnl_pct": 8},
    "Негатив по сектору": {"rsi": 50, "has_position": True, "pnl_pct": 3},
    "Дивидендные новости": {"rsi": 55, "market_regime": 1},
    "Санкции": {"rsi": 50, "has_position": True, "pnl_pct": 1},
    "Рынок растёт, нефть растёт": {"rsi": 55, "imoex": 2550, "imoex_change": 1.5, "brent": 82, "brent_change": 2, "rvi": 22, "usd_rub": 70, "market_regime": 1},
    "Рынок падает, нефть падает": {"rsi": 45, "imoex": 2400, "imoex_change": -2.5, "brent": 77, "brent_change": -3, "rvi": 30, "usd_rub": 75, "market_regime": 2},
    "Боковик, нефть стабильна": {"rsi": 50, "imoex": 2500, "imoex_change": 0, "brent": 80, "brent_change": 0, "rvi": 25, "usd_rub": 72},
    "Рубль слабеет, рынок растёт": {"rsi": 55, "imoex": 2525, "imoex_change": 1, "brent": 81, "brent_change": 1, "rvi": 24, "usd_rub": 78, "market_regime": 1},
    "RVI>30": {"rsi": 50, "imoex": 2480, "imoex_change": -1, "brent": 79, "brent_change": -1, "rvi": 32, "usd_rub": 74, "has_position": True, "pnl_pct": 2, "market_regime": 2},
    "RVI<20": {"rsi": 55, "imoex": 2510, "imoex_change": 0.5, "rvi": 18, "market_regime": 1},
    "Нефть растёт": {"rsi": 50, "imoex": 2510, "imoex_change": 0.5, "brent": 83, "brent_change": 3, "usd_rub": 68, "rvi": 22, "market_regime": 1},
    "Рынок падает, RVI высокий": {"rsi": 40, "imoex": 2350, "imoex_change": -3, "brent": 75, "brent_change": -4, "rvi": 35, "usd_rub": 77, "has_position": True, "pnl_pct": -2, "market_regime": 2},
    "Пустой портфель": {"rsi": 55, "imoex_change": 1, "cash": 9000, "positions_count": 0, "market_regime": 1},
    "Портфель заполнен": {"rsi": 40, "imoex_change": -2, "cash": 1000, "positions_count": 8, "exposure": 0.85, "has_position": True, "pnl_pct": -3, "market_regime": 2},
    "Концентрация в секторе": {"rsi": 55, "cash": 5000, "positions_count": 3, "exposure": 0.45, "has_position": True, "pnl_pct": 5},
    "Есть кэш, рынок растёт": {"rsi": 55, "imoex_change": 1, "cash": 6000, "positions_count": 3, "exposure": 0.35, "market_regime": 1},
    "Все в плюсе": {"rsi": 75, "cash": 3000, "positions_count": 5, "exposure": 0.65, "has_position": True, "pnl_pct": 6},
    "Все в минусе": {"rsi": 30, "cash": 2000, "positions_count": 5, "exposure": 0.75, "has_position": True, "pnl_pct": -4},
    "Кэш кончается": {"rsi": 50, "cash": 500, "positions_count": 7, "exposure": 0.9, "has_position": True, "pnl_pct": -2},
    "Портфель сбалансирован": {"rsi": 55, "cash": 5000, "positions_count": 3, "exposure": 0.45},
    "Высокая частота сделок": {"rsi": 50, "volume": 3e9},
    "Низкая частота сделок": {"rsi": 55, "volume": 1e9},
    "Комиссия съела прибыль": {"rsi": 50, "has_position": True, "pnl_pct": 0.5},
    "Комиссия минимальна": {"rsi": 55, "imoex_change": 0.5, "volume": 1e9},
    "Геополитический шок": {"rsi": 20, "imoex_change": -5, "brent_change": -6, "rvi": 40, "market_regime": 2, "has_position": True, "pnl_pct": -5},
    "Отскок после паники": {"rsi": 35, "imoex_change": 3, "brent_change": 4, "rvi": 28, "market_regime": 1},
    "Затяжной боковик": {"rsi": 50, "bb_pos": 0.5, "momentum": 0, "volume": 5e7},
    "Утро": {"rsi": 55, "imoex_change": 1, "market_regime": 1, "volume": 3e9},
    "Вечер": {"rsi": 65, "bb_pos": 0.8, "volume": 5e7},
    "Позиция 1 час": {"rsi": 55, "has_position": True, "pnl_pct": 1.5, "hold_time": 1},
    "Дневной горизонт": {"rsi": 65, "has_position": True, "pnl_pct": 0.8, "hold_time": 4},
    "Недельный горизонт": {"rsi": 55, "has_position": True, "pnl_pct": 2, "hold_time": 24},
    "SELL без позиции — ошибка": {"rsi": 70, "imoex_change": -1, "cash": 10000, "positions_count": 0, "exposure": 0},
    "HOLD при пустом портфеле — терпение": {"rsi": 70, "imoex_change": -1, "cash": 10000, "positions_count": 0,
                                            "exposure": 0},
}

for seed in seeds:
    params = seed_params.get(seed["desc"], {})
    state = create_base_state(**params)
    strategy_params = list(model.strategies.values())[0] if model.strategies else {}
    full_state = model._create_strategy_state(state, strategy_params)

    experience = {
        'state': full_state.cpu(),
        'action': seed["action"],
        'reward': seed["reward"],
        'next_state': full_state.cpu(),
        'done': True,
        'pnl_rub': seed["reward"] * 100,
        'timestamp': datetime.now().isoformat()
    }
    model.memory.append(experience)
    if hasattr(model, 'prioritized_buffer'):
        model.prioritized_buffer.add(experience, td_error=2.0)

model.save_memory()
print(f"Добавлено {len(seeds)} seed-опытов")

# ============================================================
# ЗАГРУЗКА СВЕЧЕЙ ДЛЯ ТИКЕРА (функция)
# ============================================================
def load_candles(ticker, start_date, end_date):
    """Загружает свечи по частям"""
    all_candles = []
    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    period_end = datetime.strptime(end_date, "%Y-%m-%d")

    while current_start < period_end:
        chunk_end = min(current_start + timedelta(days=100), period_end)

        url = f"{moex.base_url}/engines/stock/markets/shares/securities/{ticker}/candles.json"
        params = {
            'interval': '24',
            'from': current_start.strftime('%Y-%m-%d'),
            'till': chunk_end.strftime('%Y-%m-%d'),
            'iss.meta': 'off'
        }

        try:
            data = moex._make_request(url, params, timeout=15)
            if data and 'candles' in data and data['candles']['data']:
                cols = data['candles']['columns']
                rows = data['candles']['data']
                df = pd.DataFrame(rows, columns=cols)
                if 'begin' in df.columns:
                    df['datetime'] = pd.to_datetime(df['begin'])
                    df.set_index('datetime', inplace=True)
                    df.sort_index(inplace=True)
                    column_mapping = {'open': 'Open', 'close': 'Close', 'high': 'High',
                                     'low': 'Low', 'volume': 'Volume', 'value': 'Value'}
                    for old_col, new_col in column_mapping.items():
                        if old_col in df.columns:
                            df.rename(columns={old_col: new_col}, inplace=True)
                    all_candles.append(df)
        except Exception as e:
            logger.debug(f"Ошибка загрузки {ticker} {current_start.date()}: {e}")

        current_start = chunk_end + timedelta(days=1)

    if all_candles:
        candles = pd.concat(all_candles)
        candles = candles[~candles.index.duplicated()]
        candles.sort_index(inplace=True)
        return candles
    return None


def load_historical_macro(start_date, end_date):
    """Загружает исторические макро-данные (IMOEX, RTSI, RVI)"""
    macro_history = {}

    for index_name, ticker in [("IMOEX", "IMOEX"), ("RTSI", "RTSI"), ("RVI", "RVI")]:
        url = f"{moex.base_url}/history/engines/stock/markets/index/securities/{ticker}.json"
        params = {
            'from': start_date,
            'till': end_date,
            'iss.meta': 'off',
            'history.columns': 'TRADEDATE,CLOSE'
        }

        try:
            data = moex._make_request(url, params, timeout=15)
            if data and 'history' in data and data['history']['data']:
                for row in data['history']['data']:
                    date_str = row[0]
                    value = float(row[1]) if row[1] else 0
                    if date_str not in macro_history:
                        macro_history[date_str] = {}
                    macro_history[date_str][index_name] = value
        except Exception as e:
            logger.debug(f"Ошибка загрузки истории {index_name}: {e}")

    return macro_history

# ============================================================
# ФУНКЦИЯ ОБУЧЕНИЯ НА ПЕРИОДЕ
# ============================================================
def train_on_period(tickers, start_date, end_date, macro_history, learn=True):
    """Обучение или валидация на периоде"""
    total_steps = 0
    total_reward = 0.0
    final_portfolio_value = 10000.0

    for ticker_idx, ticker in enumerate(tickers):
        print(f"\nТикер [{ticker_idx + 1}/{len(tickers)}]: {ticker}")

        candles = load_candles(ticker, start_date, end_date)
        if candles is None or candles.empty:
            print(f"  Нет данных, пропускаю")
            continue

        print(f"  Загружено {len(candles)} свечей")

        portfolio = PortfolioManager()
        # Сброс портфеля после загрузки из файла — каждый тикер стартует с чистым портфелем
        portfolio.positions.clear()
        portfolio.trade_history.clear()
        portfolio.cash = settings.get("initial_capital_rub", 10000)
        portfolio.initial_capital = portfolio.cash
        portfolio.reserved_cash = 0
        portfolio.total_commission = 0
        portfolio.commission_spent_today = 0
        portfolio.max_positions = settings.get("max_positions", 10)
        portfolio.max_trades_per_hour = 999999
        portfolio.daily_commission_limit = 999999
        portfolio.settings = {
            'max_positions_per_horizon': {'balanced': 999, 'day_session': 999, 'three_days': 999, 'week': 999}}
        portfolio.training_wheels = {}

        tech_core.price_history.clear()
        tech_core.indicators_cache.clear()

        warmup = min(20, len(candles))
        for i in range(warmup):
            row = candles.iloc[i]
            tech_core.update_price_data(ticker, float(row['Close']), int(row.get('Volume', 0) or 0))

        for i in range(warmup, len(candles)):
            row = candles.iloc[i]
            current_date = row.name.strftime('%Y-%m-%d')
            price = float(row['Close'])
            volume = int(row.get('Volume', 0) or 0)

            tech_core.update_price_data(ticker, price, volume)

            news_features = model.encode_news([f"Корпоративная новость {ticker}"])

            if i == warmup:
                securities = moex.get_all_securities()
            sec_info = securities.get(ticker, {'lot_size': 1, 'min_step': 0.01, 'sector': 'other', 'market_cap': 5e12})

            indicators = tech_core.calculate_indicators(ticker)

            # Используем исторические макро-данные или получаем актуальные из MOEX
            day_macro = macro_history.get(current_date, {})
            if not day_macro:
                # Получаем актуальные макро-данные как fallback
                live_macro = moex.get_macro_data()
                default_imoex = live_macro.get('imoex', 2500.0)
                default_rtsi = live_macro.get('rtsi', 1100.0)
                default_rvi = live_macro.get('rvi', 25.0)
                default_brent = live_macro.get('brent', 80.0)
                default_usd_rub = live_macro.get('usd_rub', 72.0)
                default_cbr_rate = live_macro.get('cbr_rate', 13.5)
                default_vix = live_macro.get('vix', 16.0)
                default_moexog = live_macro.get('moexog', 4500.0)
                default_moexfn = live_macro.get('moexfn', 8000.0)
            else:
                default_imoex = 2500.0
                default_rtsi = 1100.0
                default_rvi = 25.0
                default_brent = 80.0
                default_usd_rub = 72.0
                default_cbr_rate = 13.5
                default_vix = 16.0
                default_moexog = 4500.0
                default_moexfn = 8000.0

            imoex_val = day_macro.get('IMOEX', default_imoex)
            rtsi_val = day_macro.get('RTSI', default_rtsi)
            rvi_val = day_macro.get('RVI', default_rvi)

            # Вычисляем изменение IMOEX из истории
            prev_date = (datetime.strptime(current_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            prev_imoex = macro_history.get(prev_date, {}).get('IMOEX', imoex_val)
            imoex_change = ((imoex_val - prev_imoex) / prev_imoex * 100) if prev_imoex > 0 else 0

            # Определяем режим рынка
            if abs(imoex_change) < 0.3:
                market_regime = 0  # Боковик
            elif imoex_change > 0:
                market_regime = 1  # Растущий
            else:
                market_regime = 2  # Падающий

            market_data = {
                'volume': volume,
                'spread': sec_info.get('spread', 0.01),
                'rsi': indicators.get('rsi', 50),
                'volatility': indicators.get('atr', 0) / price if price > 0 else 0.02,
                'sma_10_ratio': indicators.get('sma_10', price) / price if price > 0 else 1.0,
                'sma_20_ratio': indicators.get('sma_20', price) / price if price > 0 else 1.0,
                'bb_position': indicators.get('bb_position', 0.5),
                'volume_ratio': indicators.get('volume_ratio', 1.0),
                'atr': indicators.get('atr', 0),
                'market_cap': sec_info.get('market_cap', 5e12),
                'lot_size': sec_info.get('lot_size', 1),
                'min_step': sec_info.get('min_step', 0.01),
                'sector': sec_info.get('sector', 'other'),
                'momentum': indicators.get('momentum', 0),
                'imoex': imoex_val,
                'imoex_change': imoex_change,
                'rtsi': rtsi_val,
                'rtsi_change': 0.0,
                'rvi': rvi_val,
                'rvi_change': 0.0,
                'moexog': default_moexog,
                'moexfn': default_moexfn,
                'brent': default_brent,
                'brent_change': 0.0,
                'market_liquidity_ratio': 0.8,
                'market_activity_score': 1.0,
                'spread_pct': sec_info.get('spread', 0.01) / price if price > 0 else 0.0,
                'market_mood': 0.0,
                'shares_turnover': 5e11,
                'rvi_normalized': rvi_val / 100.0,
                'imoex_normalized': imoex_val / 4000.0,
                'market_cap_total': 5e13,
                'liquidity_ratio': 0.8,
                'cbr_rate_normalized': default_cbr_rate / 20.0,
                'vix': default_vix / 50.0,
                'moexog_normalized': default_moexog / 10000.0,
                'market_regime': market_regime
            }

            state = model.build_state_vector(
                ticker=ticker, price=price,
                momentum=indicators.get('momentum', 0),
                sentiment=0, news_features=news_features,
                market_data=market_data, market_sentiment=0,
                portfolio=portfolio
            )

            strategy_params = list(model.strategies.values())[0] if model.strategies else {}
            full_state = model._create_strategy_state(state, strategy_params)
            state_tensor = full_state.unsqueeze(0).to(model.device)

            with torch.no_grad():
                action_probs, state_value, _ = model.policy_net(state_tensor)
                if learn:
                    action = torch.multinomial(action_probs, 1).item()
                else:
                    action = action_probs.argmax().item()

            # Исполняем сделку
            if action in [3, 4]:
                qty = int(portfolio.cash * 0.1 / price)
                lot_size = sec_info.get('lot_size', 1)
                if lot_size > 1:
                    qty = (qty // lot_size) * lot_size
                if qty < lot_size:
                    qty = lot_size
                if qty > 0 and portfolio.cash >= qty * price:
                    portfolio.buy(ticker, qty, price, 'balanced')
            elif action in [5, 6] and ticker in portfolio.positions:
                pos = portfolio.positions[ticker]
                qty = pos['qty'] if action == 6 else int(pos['qty'] * 0.5)
                if qty > 0:
                    portfolio.sell(ticker, qty, price)

            # Reward — дифференцированный по режиму рынка
            if i + 1 < len(candles):
                next_price = float(candles.iloc[i + 1]['Close'])
                price_change = (next_price - price) / price

                if action in [3, 4]:  # BUY
                    if market_regime == 1:
                        reward = price_change * 100
                    elif market_regime == 2:
                        reward = price_change * 150
                    else:
                        reward = price_change * 50
                    reward -= 0.003  # комиссия 0.3%
                elif action in [5, 6] and ticker in portfolio.positions:
                    entry_price = portfolio.positions[ticker]['avg_price']
                    if entry_price > 0:
                        pnl_pct = (price - entry_price) / entry_price
                        reward = pnl_pct * 100
                    else:
                        reward = 0
                    reward -= 0.003
                elif action in [5, 6]:
                    reward = -1.0
                else:  # HOLD
                    if ticker in portfolio.positions:
                        pos = portfolio.positions[ticker]
                        entry_price = pos['avg_price']
                        if entry_price > 0:
                            pnl_pct = (price - entry_price) / entry_price
                            reward = pnl_pct * 10
                        else:
                            reward = 0
                    else:
                        reward = -0.1

                reward = max(-5.0, min(5.0, reward))
            else:
                reward = 0

            # Сохраняем опыт
            next_state = model.build_state_vector(
                ticker=ticker, price=price,
                momentum=indicators.get('momentum', 0),
                sentiment=0, news_features=news_features,
                market_data=market_data, market_sentiment=0,
                portfolio=portfolio
            )
            next_full = model._create_strategy_state(next_state, strategy_params)

            model.remember_experience(
                state=full_state, action=action, reward=reward,
                next_state=next_full, done=(i == len(candles) - 1),
                pnl_rub=reward * 100
            )

            total_steps += 1
            total_reward += reward

            # Обучение каждые 32 шага (только в режиме обучения)
            if learn and total_steps % 32 == 0 and len(model.memory) >= 32:
                model.learn_from_experience(batch_size=32)

            if total_steps % 50 == 0:
                recent_rewards = [model.memory[j]['reward'] for j in
                                  range(max(0, len(model.memory) - 50), len(model.memory)) if
                                  isinstance(model.memory[j], dict)]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                print(
                    f"  [{current_date}] Шаг {total_steps}: reward={reward:+.2f}, ср.за 50={avg_reward:+.2f}, память={len(model.memory)}, кэш={portfolio.cash:.0f}₽")

        final_portfolio_value = portfolio.cash + sum(
            pos['qty'] * (moex.get_price(ticker) or pos['avg_price'])
            for ticker, pos in portfolio.positions.items()
        )

    return total_steps, total_reward, final_portfolio_value

# ============================================================
# ЭТАП 2: ОБУЧЕНИЕ (12 месяцев)
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 2: ОБУЧЕНИЕ (12 месяцев)")
print("=" * 70)

print("Загрузка исторических макро-данных...")
macro_history = load_historical_macro(TRAIN_START, TRAIN_END)
print(f"Загружено макро-данных за {len(macro_history)} дней")

train_steps, train_reward, train_value = train_on_period(TICKERS, TRAIN_START, TRAIN_END, macro_history, learn=True)

print(f"\nОбучение завершено: {train_steps} шагов, средний reward={train_reward/train_steps:+.4f}")
print(f"Финальная стоимость портфеля: {train_value:,.0f}₽")

# ============================================================
# ЭТАП 3: ВАЛИДАЦИЯ (3 месяца, без обучения)
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 3: ВАЛИДАЦИЯ (3 месяца, без обучения)")
print("=" * 70)

print("Загрузка исторических макро-данных для валидации...")
val_macro_history = load_historical_macro(VAL_START, VAL_END)
print(f"Загружено макро-данных за {len(val_macro_history)} дней")

val_steps, val_reward, val_value = train_on_period(TICKERS, VAL_START, VAL_END, val_macro_history, learn=False)

print(f"\nВалидация завершена: {val_steps} шагов", end="")
if val_steps > 0:
    print(f", средний reward={val_reward/val_steps:+.4f}")
else:
    print()
print(f"Финальная стоимость портфеля: {val_value:,.0f}₽")
print(f"PnL за валидацию: {val_value - 10000:+,.0f}₽")

# ============================================================
# РЕЗУЛЬТИРУЮЩИЙ ЛОГ
# ============================================================
print("\n" + "=" * 70)
print("📊 РЕЗУЛЬТИРУЮЩИЙ ЛОГ ПРЕДОБУЧЕНИЯ")
print("=" * 70)

print(f"\n🧠 ПАМЯТЬ:")
print(f"   Всего опытов: {len(model.memory)}")
if len(model.memory) > 0:
    rewards = [exp['reward'] for exp in model.memory if isinstance(exp, dict)]
    actions = [exp['action'] for exp in model.memory if isinstance(exp, dict)]

    if rewards:
        print(f"   Средний reward: {np.mean(rewards):+.4f}")
        print(f"   Медианный reward: {np.median(rewards):+.4f}")
        print(f"   Мин reward: {np.min(rewards):+.4f}")
        print(f"   Макс reward: {np.max(rewards):+.4f}")
        positive = sum(1 for r in rewards if r > 0)
        print(f"   Положительных: {positive}/{len(rewards)} ({positive/len(rewards)*100:.1f}%)")

    if actions:
        action_counts = {a: actions.count(a) for a in set(actions)}
        action_names = {0: "HOLD_SHORT", 1: "HOLD", 2: "HOLD_LONG",
                       3: "BUY_SMALL", 4: "BUY_NORMAL", 5: "SELL_SMALL", 6: "SELL_ALL"}
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ДЕЙСТВИЙ В ПАМЯТИ:")
        for action_id in sorted(action_counts.keys()):
            name = action_names.get(action_id, str(action_id))
            count = action_counts[action_id]
            pct = count / len(actions) * 100
            bar = "█" * int(pct / 2)
            print(f"   {name:<15} {count:>5} ({pct:>5.1f}%) {bar}")

print(f"\n📈 ОБУЧЕНИЕ:")
print(f"   Шагов обучения: {train_steps}")
print(f"   Средний reward (обучение): {train_reward/train_steps:+.4f}" if train_steps > 0 else "   Нет шагов")
print(f"   Портфель после обучения: {train_value:,.0f}₽")

print(f"\n📉 ВАЛИДАЦИЯ:")
print(f"   Шагов валидации: {val_steps}")
print(f"   Средний reward (валидация): {val_reward/val_steps:+.4f}" if val_steps > 0 else "   Нет шагов")
print(f"   Портфель после валидации: {val_value:,.0f}₽")
print(f"   PnL: {val_value - 10000:+,.0f}₽ ({(val_value/10000 - 1)*100:+.2f}%)")

print(f"\n🎯 ВЕСА МОДЕЛИ:")
action_net_bias = model.policy_net.action_net[2].bias.detach().cpu().numpy()
action_names = ["HOLD_SHORT", "HOLD", "HOLD_LONG", "BUY_SMALL", "BUY_NORMAL", "SELL_SMALL", "SELL_ALL"]
print(f"   Bias выходного слоя:")
for i, (name, bias) in enumerate(zip(action_names, action_net_bias)):
    direction = "📈" if bias > 0 else "📉"
    print(f"   {direction} {name:<15} bias={bias:+.4f}")

print(f"\n🧪 ТЕСТОВЫЙ ПРОГОН:")
test_state = model.build_state_vector(
    ticker="SBER", price=300.0, momentum=0, sentiment=0,
    news_features=model.encode_news(["Тестовая новость"]),
    market_data={
        'volume': 1e9, 'spread': 0.01, 'rsi': 50, 'volatility': 0.02,
        'sma_10_ratio': 1.0, 'sma_20_ratio': 1.0, 'bb_position': 0.5,
        'volume_ratio': 1.0, 'atr': 3.0, 'market_cap': 5e12,
        'lot_size': 1, 'min_step': 0.01, 'sector': 'other', 'momentum': 0,
        'imoex': 2500, 'imoex_change': 0, 'rtsi': 1100, 'rtsi_change': 0,
        'rvi': 25, 'rvi_change': 0, 'moexog': 4500, 'moexfn': 8000,
        'brent': 80, 'brent_change': 0, 'market_liquidity_ratio': 0.8,
        'market_activity_score': 1.0, 'spread_pct': 0.01/300,
        'market_mood': 0, 'shares_turnover': 5e11,
        'rvi_normalized': 0.25, 'imoex_normalized': 0.625,
        'market_cap_total': 5e13, 'liquidity_ratio': 0.8,
        'cbr_rate_normalized': 0.675, 'vix': 16, 'moexog_normalized': 0.45,
        'market_regime': 0
    },
    market_sentiment=0, portfolio=PortfolioManager()
)

strategy_params = list(model.strategies.values())[0] if model.strategies else {}
test_full = model._create_strategy_state(test_state, strategy_params)
test_tensor = test_full.unsqueeze(0).to(model.device)

model.policy_net.eval()
with torch.no_grad():
    probs, value, _ = model.policy_net(test_tensor)
    probs = probs.cpu().numpy().flatten()
    print(f"   State value: {value.item():+.4f}")
    print(f"   Распределение вероятностей:")
    for i, p in enumerate(probs):
        bar = "█" * int(p * 30)
        print(f"   {action_names[i]:<15} {p:.3f} {bar}")

# ============================================================
# СОХРАНЕНИЕ
# ============================================================
print("\n" + "=" * 70)
print("СОХРАНЕНИЕ МОДЕЛИ")
print("=" * 70)

model.save_model()
model.save_memory()
print(f"Модель сохранена: {len(model.memory)} опытов")

# ============================================================
# ОЧИСТКА ПОРТФЕЛЯ ПОСЛЕ ОБУЧЕНИЯ
# ============================================================
print("\n" + "=" * 70)
print("ОЧИСТКА ПОРТФЕЛЯ ПОСЛЕ ОБУЧЕНИЯ")
print("=" * 70)

# Сбрасываем портфель к начальному состоянию
portfolio_state = {
    "total_value": 10000,
    "cash": 10000,
    "positions": {},
    "last_update": datetime.now().isoformat(),
    "initial_capital": 10000,
    "reserved_cash": 0,
    "pending_commissions": [],
    "trade_history": [],
    "daily_trades": [],
    "commission_spent_today": 0.0,
    "total_commission": 0.0,
    "total_trades": 0,
    "total_pnl": 0.0
}

with open("data/portfolio_state.json", "w", encoding="utf-8") as f:
    json.dump(portfolio_state, f, indent=2, default=str)

print("Портфель очищен: 10 000₽ кэша, 0 позиций")


print("\n" + "=" * 70)
print("✅ ПРЕДОБУЧЕНИЕ ЗАВЕРШЕНО")
print("=" * 70)