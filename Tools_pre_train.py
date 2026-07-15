"""
Предобучение модели (v3): Seed + История + Валидация + Макро + Режим рынка
+ Интеграция с новыми модулями: history_loader (хаос-метрики),
  Hawkes (per-ticker thresholds + инициализация историей),
  tickers.json (60 ликвидных+волатильных тикеров).
"""
import json
import time
import os
import torch
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from models.trader_model import trader_model_instance
from core.core_technical_trader import TechnicalTraderCore
from fetchers.moex_fetcher import MoexFetcher
from utils.portfolio_manager import PortfolioManager
from utils.logger import get_logger
from fetchers.history_loader import HistoryLoader, history_loader
from core.hawkes_signal import hawkes_signal
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

# 🆕 Тикеры из config/tickers.json (60 ликвидных+волатильных)
TICKERS = []
try:
    with open("config/tickers.json", "r", encoding="utf-8") as f:
        tickers_cfg = json.load(f)
    TICKERS = [t["ticker"] for t in tickers_cfg.get("watchlist", [])]
    logger.info(f"Загружено {len(TICKERS)} тикеров из config/tickers.json")
except Exception as e:
    logger.warning(f"Не удалось загрузить config/tickers.json ({e}), используем fallback")
    # Fallback — базовые голубые фишки
    TICKERS = [
        "GAZP", "LKOH", "ROSN", "SNGS", "TATN", "NVTK",
        "SBER", "MOEX", "TCSG", "BSPB", "RENI",
        "GMKN", "CHMF", "MAGN", "NLMK", "PLZL", "ALRS",
        "MTSS", "RTKM", "YNDX", "VKCO",
        "MGNT", "DSKY", "AFLT", "FESH", "BELU",
        "IRAO", "HYDR", "FEES", "UPRO",
        "PHOR", "LIFE", "TRNFP", "NMTP", "DELI", "UNAC", "KMAZ",
    ]

# Параметры периодов — из конфига (если есть) или fallback
pretrain_cfg = settings.get("pre_train", {})
# 🆕 v10.1: Reward-параметры из конфига (без хардкода)
reward_cfg = pretrain_cfg.get("rewards", {})
REWARD_MATCH = reward_cfg.get("match_multiplier", 2.0)
REWARD_PARTIAL = reward_cfg.get("partial_match_multiplier", 0.8)
REWARD_HOLD = reward_cfg.get("hold_match_multiplier", 1.2)
REWARD_MISMATCH = reward_cfg.get("mismatch_multiplier", -2.0)
REWARD_SELL_NO_POS = reward_cfg.get("sell_without_position_penalty", -2.0)
REWARD_CLIP_MIN = reward_cfg.get("reward_clip_min", -5.0)
REWARD_CLIP_MAX = reward_cfg.get("reward_clip_max", 5.0)

TRAIN_START = pretrain_cfg.get("train_start", "2023-06-01")
TRAIN_END   = pretrain_cfg.get("train_end",   "2024-05-31")
TRAIN2_START = pretrain_cfg.get("train2_start", "2024-06-01")
TRAIN2_END   = pretrain_cfg.get("train2_end",   "2025-05-31")
VAL_START   = pretrain_cfg.get("val_start",   "2025-06-01")
VAL_END     = pretrain_cfg.get("val_end",     "2026-06-23")
STRESS_SVO_START = pretrain_cfg.get("stress_svo_start", "2022-01-15")
STRESS_SVO_END   = pretrain_cfg.get("stress_svo_end",   "2022-04-15")
STRESS_COVID_START = pretrain_cfg.get("stress_covid_start", "2020-02-01")
STRESS_COVID_END   = pretrain_cfg.get("stress_covid_end",   "2020-04-30")

# Ограничение числа тикеров для ускорения pre_train
# (60 тикеров × 365 дней × 5 периодов = ~109500 шагов, это долго)
MAX_TICKERS_PRETRAIN = pretrain_cfg.get("max_tickers", 20)
if MAX_TICKERS_PRETRAIN > 0 and len(TICKERS) > MAX_TICKERS_PRETRAIN:
    logger.info(f"Ограничиваем pre_train до {MAX_TICKERS_PRETRAIN} тикеров (из {len(TICKERS)})")
    TICKERS = TICKERS[:MAX_TICKERS_PRETRAIN]

print("=" * 70)
print("🧠 ПРЕДОБУЧЕНИЕ МОДЕЛИ (v3 — интеграция с хаос-метриками и Хоксом)")
print("=" * 70)
print(f"Тикеров: {len(TICKERS)}")
print(f"Этап 1 (учитель): {TRAIN_START} → {TRAIN_END}")
print(f"Этап 2 (самостоятельно): {TRAIN2_START} → {TRAIN2_END}")
print(f"Валидация: {VAL_START} → {VAL_END}")
print(f"Стресс-тест СВО: {STRESS_SVO_START} → {STRESS_SVO_END}")
print(f"Стресс-тест Ковид: {STRESS_COVID_START} → {STRESS_COVID_END}")

# ============================================================
# ИНИЦИАЛИЗАЦИЯ
# ============================================================
model = trader_model_instance
moex = MoexFetcher()
tech_core = TechnicalTraderCore()

# 🆕 Инициализация history_loader и расчёт хаос-метрик
print("\n--- Загрузка истории и расчёт хаос-метрик ---")
try:
    history_data = history_loader.load_history(months_back=3)
    chaos_metrics = history_loader.get_chaos_metrics()
    print(f"✓ Хаос-метрики: {len(chaos_metrics)} тикеров")
    # 🆕 Инициализация Хокса историческими ценами + per-ticker thresholds
    if hawkes_signal and history_data:
        now_ts = time.time()
        n_init = 0
        for ticker, hist in history_data.items():
            prices = hist.get('prices', [])
            if len(prices) < 50:
                continue
            # Per-ticker threshold по волатильности
            if ticker in chaos_metrics:
                vol = chaos_metrics[ticker].get('volatility_pct', 0.5)
                hawkes_signal.set_ticker_volatility(ticker, vol)
            # 🆕 v14.1: Timestamps заканчиваются СЕЙЧАС — события свежие
            n_points = min(len(prices), 200)
            for i, p in enumerate(prices[-n_points:]):
                ts = now_ts - (n_points - 1 - i) * 3600
                hawkes_signal.update_price(ticker, p, ts)
            hawkes_signal.fit(ticker, now_ts)
            n_init += 1
        h_stats = hawkes_signal.get_stats()
        print(f"✓ Хокс инициализирован: {n_init} тикеров, "
              f"bullish_events={h_stats['total_bullish_events']}, "
              f"bearish_events={h_stats['total_bearish_events']}")
except Exception as e:
    logger.warning(f"Не удалось инициализировать history/hawkes: {e}")
    chaos_metrics = {}

print(f"\nПамять до: {len(model.memory)} опытов")

# ============================================================
# ЭТАП 0: SEED EXPERIENCES (48 опытов)
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 0: SEED EXPERIENCES")
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
        'market_regime': market_regime,
        # 🆕 CNYRUB
        'cny_rub': 11.5, 'cny_rub_change': 0.0,
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

for seed in seeds:
    params = seed.get("params", {})
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
# LOOKAHEAD: ВЫЧИСЛЕНИЕ ИДЕАЛЬНОГО ДЕЙСТВИЯ
# ============================================================
def calculate_lookahead_ideal_action(candles, current_idx, price, commission_rate=0.003):
    """
    Вычисляет идеальное действие на основе будущих цен (учитель).

    Args:
        candles: DataFrame со свечами (индекс — datetime)
        current_idx: текущий индекс в candles
        price: текущая цена
        commission_rate: ставка комиссии (0.003 = 0.3%)

    Returns:
        dict с полями:
            - ideal_action: int (0-6) или None если недостаточно данных
            - confidence: float (0-1) уверенность в идеальном действии
            - horizon_days: int горизонт, на котором вычислено действие
            - expected_pnl_pct: float ожидаемая прибыль в %
    """
    with open("config/training_wheels.json", "r", encoding="utf-8") as f:
        tw = json.load(f)

    lookahead_config = tw.get('lookahead', {})
    if not lookahead_config.get('enabled', True):
        return {'ideal_action': None, 'confidence': 0.0, 'horizon_days': 0, 'expected_pnl_pct': 0.0}

    horizons = lookahead_config.get('horizons', [1, 2, 3])
    min_price_change = lookahead_config.get('min_price_change_pct', 0.5) / 100.0
    commission_aware = lookahead_config.get('commission_aware', True)

    total_bars = len(candles)
    best_action = None
    best_confidence = 0.0
    best_horizon = 0
    best_pnl_pct = 0.0

    # Проверяем каждый горизонт
    for horizon in horizons:
        future_idx = current_idx + horizon

        # Проверяем, что будущий индекс в пределах данных
        if future_idx >= total_bars:
            continue

        future_price = float(candles.iloc[future_idx]['Close'])

        if future_price <= 0 or price <= 0:
            continue

        price_change_pct = (future_price - price) / price

        # Учитываем комиссии, если включено
        if commission_aware:
            # BUY + SELL = 2 комиссии
            net_change_pct = price_change_pct - (2 * commission_rate)
        else:
            net_change_pct = price_change_pct

        # Определяем идеальное действие
        if net_change_pct > min_price_change:
            # BUY: выбираем между BUY_SMALL и BUY_NORMAL в зависимости от силы сигнала
            if net_change_pct > min_price_change * 3:
                action = 4  # BUY_NORMAL для сильного сигнала
            else:
                action = 3  # BUY_SMALL для умеренного сигнала

            confidence = min(1.0, abs(net_change_pct) / (min_price_change * 5))
            pnl_pct = net_change_pct

        elif net_change_pct < -min_price_change:
            # SELL: выбираем между SELL_SMALL и SELL_ALL
            if net_change_pct < -min_price_change * 3:
                action = 6  # SELL_ALL для сильного падения
            else:
                action = 5  # SELL_SMALL для умеренного падения

            confidence = min(1.0, abs(net_change_pct) / (min_price_change * 5))
            pnl_pct = net_change_pct

        else:
            # HOLD: боковик
            action = 0  # HOLD
            confidence = 0.3  # Низкая уверенность в боковике
            pnl_pct = 0.0

        # Выбираем горизонт с максимальной уверенностью
        if confidence > best_confidence:
            best_confidence = confidence
            best_action = action
            best_horizon = horizon
            best_pnl_pct = pnl_pct

    # Если ни один горизонт не подошёл (конец данных)
    if best_action is None:
        return {'ideal_action': None, 'confidence': 0.0, 'horizon_days': 0, 'expected_pnl_pct': 0.0}

    return {
        'ideal_action': best_action,
        'confidence': best_confidence,
        'horizon_days': best_horizon,
        'expected_pnl_pct': best_pnl_pct
    }


# ============================================================
# ФУНКЦИЯ ОБУЧЕНИЯ НА ПЕРИОДЕ
# ============================================================
def train_on_period(tickers, start_date, end_date, macro_history, learn=True, force_teacher=False):
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
        # 🆕 v11.1: Полный сброс портфеля — убираем наследование между тикерами
        # PortfolioManager.__init__ загружает state из data/portfolio_state.json
        # Это приводило к "Усреднена позиция" — позиция от прошлого тикера наследовалась
        portfolio.positions = {}
        portfolio.trade_history = []
        portfolio.daily_trades = []
        portfolio.pending_commissions = []
        portfolio.strategy_positions = defaultdict(list)
        portfolio.cash = settings.get("initial_capital_rub", 10000)
        portfolio.initial_capital = portfolio.cash
        portfolio.reserved_cash = 0
        portfolio.commission_reserve = 0.0
        portfolio.commission_spent_today = 0.0
        portfolio.total_commission = 0.0
        portfolio.total_trades = 0
        portfolio.total_pnl = 0.0
        portfolio.max_positions = settings.get("max_positions", 10)
        portfolio.max_trades_per_hour = 999999
        portfolio.daily_commission_limit = 999999
        portfolio.settings = {
            'max_positions_per_horizon': {'balanced': 999, 'day_session': 999, 'three_days': 999, 'week': 999}}
        portfolio.training_wheels = {}


        # Загрузка параметров

        with open("config/training_wheels.json", "r", encoding="utf-8") as f:
            tw = json.load(f)
        tw_risk = tw.get('risk_params', {})
        stop_loss_pct = tw_risk.get('stop_loss_percent', 6.0) / 100
        stop_loss_hold_penalty = tw_risk.get('stop_loss_hold_penalty', 2.0)
        stop_loss_sell_bonus = tw_risk.get('stop_loss_sell_bonus', 2.0)
        max_episode_steps = tw_risk.get('max_episode_steps', 20)
        episode_reward_scale = tw_risk.get('episode_reward_scale', 1.0)
        lookahead_config = tw.get('lookahead', {})
        lookahead_bonus_weight = lookahead_config.get('bonus_weight', 0.5)
        pending_buy = {}

        tech_core.price_history.clear()
        tech_core.indicators_cache.clear()

        # 🆕 Предзагрузка live_macro один раз для тикера (избегаем NameError)
        live_macro = {}
        try:
            live_macro = moex.get_macro_data() or {}
        except Exception as e:
            logger.debug(f"get_macro_data failed for {ticker}: {e}")
            live_macro = {}

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

            # 🆕 Обновление Хокса исторической ценой (для поддержания актуального forecast)
            if hawkes_signal:
                try:
                    # Используем timestamp из даты свечи
                    candle_ts = row.name.timestamp()
                    hawkes_signal.update_price(ticker, price, candle_ts)
                    # Переобучение каждые 50 свечей
                    if i > 0 and i % 50 == 0:
                        hawkes_signal.fit(ticker, candle_ts)
                except Exception as e:
                    logger.debug(f"Hawkes update failed for {ticker}: {e}")

            news_features = model.encode_news([f"Корпоративная новость {ticker}"])

            if i == warmup:
                securities = moex.get_all_securities()
            sec_info = securities.get(ticker, {'lot_size': 1, 'min_step': 0.01, 'sector': 'other', 'market_cap': 5e12})

            indicators = tech_core.calculate_indicators(ticker)

            # Используем исторические макро-данные или получаем актуальные из MOEX
            day_macro = macro_history.get(current_date, {})
            if not day_macro:
                # live_macro уже загружен выше
                default_imoex = live_macro.get('imoex', 2500.0)
                default_rtsi = live_macro.get('rtsi', 1100.0)
                default_rvi = live_macro.get('rvi', 25.0)
                default_brent = live_macro.get('brent', 80.0)
                default_usd_rub = live_macro.get('usd_rub', 72.0)
                default_cbr_rate = live_macro.get('cbr_rate', 13.5)
                default_vix = live_macro.get('vix', 16.0)
                default_moexog = live_macro.get('moexog', 4500.0)
                default_moexfn = live_macro.get('moexfn', 8000.0)
                default_cny_rub = live_macro.get('cny_rub', 11.5)
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
                default_cny_rub = 11.5

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
                'market_regime': market_regime,
                # 🆕 CNYRUB (наш анализ: β=-0.15)
                'cny_rub': default_cny_rub,
                'cny_rub_change': 0.0,
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

            # ========== LOOKAHEAD: ВЫЧИСЛЕНИЕ ИДЕАЛЬНОГО ДЕЙСТВИЯ ==========
            lookahead_result = calculate_lookahead_ideal_action(
                candles, i, price,
                commission_rate=0.003
            )
            lookahead_ideal = lookahead_result.get('ideal_action')
            lookahead_confidence = lookahead_result.get('confidence', 0.0)
            lookahead_horizon = lookahead_result.get('horizon_days', 0)
            lookahead_expected_pnl = lookahead_result.get('expected_pnl_pct', 0.0)

            with torch.no_grad():
                action_probs, state_value, _ = model.policy_net(state_tensor)

            # Маскирование недоступных действий
            available_actions = [0, 1, 2, 3, 4]
            if ticker in portfolio.positions:
                available_actions.extend([5, 6])

            mask = torch.zeros(action_probs.shape).to(model.device)
            for a in available_actions:
                mask[0, a] = 1.0
            masked_probs = action_probs * mask
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-10)

            if force_teacher and lookahead_ideal is not None:
                if lookahead_ideal in [5, 6] and ticker not in portfolio.positions:
                    action = torch.multinomial(masked_probs, 1).item()
                else:
                    action = lookahead_ideal
            elif learn:
                # Этап 2: сэмплирование (exploration)
                action = torch.multinomial(masked_probs, 1).item()
            else:
                # 🆕 Валидация: ε-greedy + soft teacher + anti-stuck
                # Проблема: после supervised pre-training модель часто замирает на HOLD
                # Решение: 3 механизма для обеспечения торговли
                val_cfg = pretrain_cfg.get('validation', {})
                epsilon = val_cfg.get('epsilon_greedy', 0.15)  # 15% случайных действий
                teacher_guidance_prob = val_cfg.get('teacher_guidance_prob', 0.30)  # 30% следуем учителю
                anti_stuck_window = val_cfg.get('anti_stuck_window', 30)  # если за 30 шагов 0 BUY
                anti_stuck_force_buy = val_cfg.get('anti_stuck_force_buy', True)

                r = random.random()
                if r < teacher_guidance_prob and lookahead_ideal is not None:
                    # Soft teacher: следуем учителю без обучения
                    if lookahead_ideal in [5, 6] and ticker not in portfolio.positions:
                        action = torch.multinomial(masked_probs, 1).item()
                    else:
                        action = lookahead_ideal
                elif r < teacher_guidance_prob + epsilon:
                    # ε-greedy: случайное действие из доступных
                    action = random.choice(available_actions)
                else:
                    # argmax (основной режим)
                    action = masked_probs.argmax().item()

                # 🆕 Anti-stuck: если за последние N шагов 0 BUY и нет позиции — принудительно BUY
                if anti_stuck_force_buy and ticker not in portfolio.positions:
                    recent_actions = [model.memory[j].get('action') if isinstance(model.memory[j], dict) else -1
                                      for j in range(max(0, len(model.memory) - anti_stuck_window), len(model.memory))]
                    recent_buys = sum(1 for a in recent_actions if a in [3, 4])
                    if recent_buys == 0 and len(recent_actions) >= anti_stuck_window:
                        # Принудительно BUY_SMALL (action=3)
                        action = 3
                        logger.debug(f"Anti-stuck: принудительный BUY {ticker} (нет покупок {anti_stuck_window} шагов)")

            # ========== ИСПОЛНЕНИЕ СДЕЛКИ ==========

            # 🆕 v12: Конвертация действий с учётом ограничений
            # 1) Запрет докупки при уже открытой позиции
            if action in [3, 4] and ticker in portfolio.positions:
                action = 1  # BUY → HOLD
            # 2) Минимальное удержание позиции — после BUY модель должна HOLD минимум N шагов
            #    Это предотвращает pattern "купил-продал на следующем шаге"
            min_hold_steps = pretrain_cfg.get('min_hold_steps_after_buy', 3)
            if action in [5, 6] and ticker in portfolio.positions:
                pos = portfolio.positions[ticker]
                buy_step = pos.get('buy_step', None)
                if buy_step is None:
                    # buy_step не сохранён — блокируем SELL (перестраховка)
                    action = 1
                    logger.debug(f"MIN_HOLD: {ticker} buy_step не найден → SELL→HOLD")
                else:
                    steps_held = i - buy_step
                    if steps_held < min_hold_steps:
                        logger.debug(f"MIN_HOLD: {ticker} шаг={i}, buy_step={buy_step}, "
                                    f"held={steps_held} < {min_hold_steps} → SELL→HOLD")
                        action = 1  # SELL → HOLD

            buy_this_step = False
            sell_blocked_this_step = False

            if action in [3, 4]:
                # 🆕 v12: Увеличены cash_fraction — было 5%/10%, стало 10%/20%
                # При цене 500₽ и кэше 10000₽: 5% = 500₽ = 1 акция (слишком мало)
                # 10% = 1000₽ = 2 акции, 20% = 2000₽ = 4 акции
                cash_fraction = 0.10 if action == 3 else 0.20
                qty = int(portfolio.cash * cash_fraction / price)
                lot_size = sec_info.get('lot_size', 1)
                if lot_size > 1:
                    qty = (qty // lot_size) * lot_size
                if qty < lot_size:
                    qty = lot_size
                if qty > 0 and portfolio.cash >= qty * price:
                    # 🆕 v12: Передаём buy_step через kwargs — portfolio.buy() сохранит его
                    portfolio.buy(ticker, qty, price, 'balanced', buy_step=i)
                    buy_this_step = True
                    if ticker not in pending_buy:
                        pending_buy[ticker] = {
                            'entry_price': price,
                            'total_cost': 0,
                            'total_qty': 0,
                            'step': i,
                            'action': action
                        }
                    pending_buy[ticker]['total_qty'] += qty
                    pending_buy[ticker]['total_cost'] += qty * price
                    pending_buy[ticker]['entry_price'] = (
                            pending_buy[ticker]['total_cost'] / pending_buy[ticker]['total_qty']
                    )


            elif action in [5, 6] and ticker in portfolio.positions:
                # 🆕 v11: Кулдаун BUY→SELL — если покупка была в этом шаге, пропускаем продажу
                if buy_this_step:
                    sell_blocked_this_step = True
                    action = 1  # конвертируем в HOLD
                else:
                    pos = portfolio.positions[ticker]
                    qty = pos['qty'] if action == 6 else int(pos['qty'] * 0.5)
                    if qty > 0:
                        portfolio.sell(ticker, qty, price)

                    # Закрываем эпизод — назначаем reward для BUY (ТОЛЬКО НА ЭТАПЕ 1)
                    if force_teacher and ticker in pending_buy:
                        entry = pending_buy[ticker]
                        pnl_pct = (price - entry['entry_price']) / entry['entry_price'] if entry['entry_price'] > 0 else 0
                        episode_reward = pnl_pct * 100 * episode_reward_scale
                        episode_reward -= 0.006
                        episode_reward = max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, episode_reward))

                        for j in range(len(model.memory) - 1, max(0, len(model.memory) - max_episode_steps) - 1, -1):
                            exp = model.memory[j]
                            if isinstance(exp, dict) and exp.get('action') in [3, 4]:
                                sent_data = exp.get('sentiment_data', {})
                                saved_lookahead_bonus = 0.0
                                if isinstance(sent_data, dict):
                                    saved_lookahead_bonus = sent_data.get('_lookahead_bonus', 0.0)
                                if exp['action'] == 3 and episode_reward < 0:
                                    exp['reward'] = episode_reward * 0.5 + saved_lookahead_bonus
                                else:
                                    exp['reward'] = episode_reward + saved_lookahead_bonus
                                exp['done'] = True
                                break

                        pending_buy[ticker]['total_qty'] -= qty
                        if pending_buy[ticker]['total_qty'] <= 0 or ticker not in portfolio.positions:
                            del pending_buy[ticker]

            # Принудительное закрытие зависших эпизодов (ТОЛЬКО НА ЭТАПЕ 1)
            forced_sell = False
            if force_teacher:
                for tkr in list(pending_buy.keys()):
                    if tkr in portfolio.positions and (i - pending_buy[tkr]['step']) > max_episode_steps:
                        pos = portfolio.positions[tkr]
                        qty = pos['qty']
                        portfolio.sell(tkr, qty, price)
                        forced_sell = True
                        entry = pending_buy[tkr]
                        pnl_pct = (price - entry['entry_price']) / entry['entry_price'] if entry['entry_price'] > 0 else 0
                        episode_reward = pnl_pct * 100 - 0.006
                        episode_reward = max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, episode_reward))

                        for j in range(len(model.memory) - 1, max(0, len(model.memory) - max_episode_steps) - 1, -1):
                            exp = model.memory[j]
                            if isinstance(exp, dict) and exp.get('action') in [3, 4]:
                                sent_data = exp.get('sentiment_data', {})
                                saved_lookahead_bonus = 0.0
                                if isinstance(sent_data, dict):
                                    saved_lookahead_bonus = sent_data.get('_lookahead_bonus', 0.0)
                                if exp['action'] == 3 and episode_reward < 0:
                                    exp['reward'] = episode_reward * 0.5 + saved_lookahead_bonus
                                else:
                                    exp['reward'] = episode_reward + saved_lookahead_bonus
                                exp['done'] = True
                                break

                        del pending_buy[tkr]

            # ========== REWARD ==========
            if force_teacher and lookahead_ideal is not None:
                # 🆕 УМЕНЬШЕННЫЙ reward — был ±5, стал ±2
                # Это предотвращает схлопывание softmax после supervised pre-training
                # Большие reward → большие gradients → большие logits → softmax collapse
                horizon_scale = min(1.0, lookahead_horizon / 3.0)
                lookahead_match = (action == lookahead_ideal)

                if lookahead_match:
                    reward = lookahead_confidence * REWARD_MATCH * horizon_scale
                elif (lookahead_ideal in [3, 4] and action in [3, 4]) or \
                        (lookahead_ideal in [5, 6] and action in [5, 6]):
                    reward = lookahead_confidence * REWARD_PARTIAL * horizon_scale
                elif lookahead_ideal == 0 and action == 0:
                    reward = lookahead_confidence * REWARD_HOLD * horizon_scale
                else:
                    reward = REWARD_MISMATCH * lookahead_confidence * horizon_scale

                reward = max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, reward))

            elif i + 1 < len(candles):
                # Этап 2 или валидация: Рыночный reward + мягкий lookahead-бонус
                next_price = float(candles.iloc[i + 1]['Close'])
                price_change = (next_price - price) / price
                commission_rate = 0.003

                # Основной reward
                if action in [3, 4]:  # BUY
                    commission_cost = commission_rate if action == 4 else commission_rate * 0.5
                    if ticker in pending_buy:
                        reward = -commission_cost * 100
                    else:
                        if market_regime == 1:
                            regime_mult = 1.0
                        elif market_regime == 2:
                            regime_mult = 0.3
                        else:
                            regime_mult = 0.7
                        reward = price_change * 100 * regime_mult
                        reward -= commission_cost * 100

                elif action in [5, 6] and ticker in portfolio.positions:
                    if forced_sell:
                        reward = 0
                    else:
                        entry_price = portfolio.positions[ticker]['avg_price']
                        if entry_price > 0:
                            pnl_pct = (price - entry_price) / entry_price
                            reward = pnl_pct * 100
                            if market_regime == 2:
                                reward *= 1.3
                            elif market_regime == 1:
                                reward *= 0.8
                            if pnl_pct < -stop_loss_pct:
                                reward += stop_loss_sell_bonus
                        else:
                            reward = 0
                    reward -= commission_rate * 100

                elif action in [5, 6]:
                    # 🆕 SELL без позиции — штраф из конфига
                    reward = REWARD_SELL_NO_POS

                else:  # HOLD
                    if ticker in portfolio.positions:
                        pos = portfolio.positions[ticker]
                        entry_price = pos['avg_price']
                        if entry_price > 0:
                            pnl_pct = (price - entry_price) / entry_price
                            if market_regime == 2 and pnl_pct < 0:
                                reward = pnl_pct * 100 - stop_loss_hold_penalty
                            elif pnl_pct < -stop_loss_pct:
                                reward = pnl_pct * 80 - stop_loss_hold_penalty
                            else:
                                reward = pnl_pct * 80
                        else:
                            reward = 0
                    else:
                        if market_regime == 1:
                            reward = -0.05
                        else:
                            reward = 0

                # Мягкий lookahead-бонус (только на Этапе 2)
                if learn and lookahead_ideal is not None:
                    horizon_scale = min(1.0, lookahead_horizon / 3.0)
                    if action == lookahead_ideal:
                        reward += lookahead_confidence * 1.5 * horizon_scale
                    elif (lookahead_ideal in [3, 4] and action in [3, 4]) or \
                            (lookahead_ideal in [5, 6] and action in [5, 6]):
                        reward += lookahead_confidence * 0.5 * horizon_scale
                    elif (lookahead_ideal in [3, 4] and action in [5, 6]) or \
                            (lookahead_ideal in [5, 6] and action in [3, 4]):
                        reward -= lookahead_confidence * 1.0 * horizon_scale

                reward = max(-5.0, min(5.0, reward))

            else:
                reward = 0

            lookahead_match = (action == lookahead_ideal) if lookahead_ideal is not None else None

            # ========== СОХРАНЕНИЕ ОПЫТА ==========
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
                pnl_rub=reward * 100,
                sentiment_data={
                    'teacher_action': lookahead_ideal,
                    'teacher_confidence': lookahead_confidence,
                    'teacher_horizon': lookahead_horizon,
                    'forced': force_teacher,
                    'available_actions': available_actions
                }
            )

            total_steps += 1
            total_reward += reward

            # Обучение каждые 32 шага
            if learn and total_steps % 32 == 0 and len(model.memory) >= 32:
                if force_teacher and lookahead_ideal is not None:
                    # 🆕 v13: Cross-entropy loss для supervised pre-training
                    # Policy gradient с reward от учителя неизбежно коллапсирует.
                    # Cross-entropy напрямую учит модель имитировать учителя.
                    model.learn_supervised(batch_size=32)
                else:
                    model.learn_from_experience(batch_size=32)

            if total_steps % 50 == 0:
                recent_rewards = [model.memory[j]['reward'] for j in
                                  range(max(0, len(model.memory) - 50), len(model.memory)) if
                                  isinstance(model.memory[j], dict)]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0

                buys = sum(1 for e in list(model.memory)[-50:] if isinstance(e, dict) and e.get('action') in [3, 4])
                sells = sum(1 for e in list(model.memory)[-50:] if isinstance(e, dict) and e.get('action') in [5, 6])
                holds = sum(1 for e in list(model.memory)[-50:] if isinstance(e, dict) and e.get('action') in [0, 1, 2])
                pos_count = len(portfolio.positions)
                print(
                    f"  [{current_date}] Шаг {total_steps}: reward={reward:+.2f}, ср.за 50={avg_reward:+.2f}, B={buys} S={sells} H={holds} поз={pos_count} кэш={portfolio.cash:.0f}₽ память={len(model.memory)}")

                # 🆕 Диагностика: предупреждение если модель только HOLD-ит (валидация)
                if not learn and not force_teacher and holds >= 48 and buys == 0 and sells == 0:
                    print(f"  ⚠️ ПРЕДУПРЕЖДЕНИЕ: модель только HOLD (B=0 S=0 H={holds}). "
                          f"Активирован anti-stuck механизм (ε-greedy + soft teacher + force-buy).")

        final_portfolio_value = portfolio.cash + sum(
            pos['qty'] * (moex.get_price(ticker) or pos['avg_price'])
            for ticker, pos in portfolio.positions.items()
        )

    return total_steps, total_reward, final_portfolio_value

# ============================================================
# ЭТАП 1: ПРИНУДИТЕЛЬНОЕ ОБУЧЕНИЕ С УЧИТЕЛЕМ
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 1: ПРИНУДИТЕЛЬНОЕ ОБУЧЕНИЕ С УЧИТЕЛЕМ")
print("=" * 70)

print("Загрузка исторических макро-данных...")
macro_history = load_historical_macro(TRAIN_START, TRAIN_END)
print(f"Загружено макро-данных за {len(macro_history)} дней")

train_steps, train_reward, train_value = train_on_period(
    TICKERS, TRAIN_START, TRAIN_END, macro_history, learn=True, force_teacher=True
)

print(f"\nЭтап 1 (учитель) завершён: {train_steps} шагов, средний reward={train_reward/train_steps:+.4f}" if train_steps > 0 else "\nЭтап 1: нет шагов")
print(f"Финальная стоимость портфеля: {train_value:,.0f}₽")


# # ============================================================
# # ЭТАП 2: САМОСТОЯТЕЛЬНАЯ ТОРГОВЛЯ
# # ============================================================
# print("\n" + "=" * 70)
# print("ЭТАП 2: САМОСТОЯТЕЛЬНАЯ ТОРГОВЛЯ")
# print("=" * 70)
#
# # Сброс портфеля перед Этапом 2
# with open("data/portfolio_state.json", "w", encoding="utf-8") as f:
#     json.dump({
#         "total_value": 10000,
#         "cash": 10000,
#         "positions": {},
#         "last_update": datetime.now().isoformat(),
#         "initial_capital": 10000,
#         "reserved_cash": 0,
#         "pending_commissions": [],
#         "trade_history": [],
#         "daily_trades": [],
#         "commission_spent_today": 0.0,
#         "total_commission": 0.0,
#         "total_trades": 0,
#         "total_pnl": 0.0
#     }, f, indent=2, default=str)
#
# print("Портфель сброшен для Этапа 2")
#
# print("Загрузка исторических макро-данных для Этапа 2...")
# macro_history_train2 = load_historical_macro(TRAIN2_START, TRAIN2_END)
# print(f"Загружено макро-данных за {len(macro_history_train2)} дней")
#
# train2_steps, train2_reward, train2_value = train_on_period(
#     TICKERS, TRAIN2_START, TRAIN2_END, macro_history_train2, learn=True, force_teacher=False
# )

# print(f"\nЭтап 2 (самостоятельно) завершён: {train2_steps} шагов, средний reward={train2_reward/train2_steps:+.4f}" if train2_steps > 0 else "\nЭтап 2: нет шагов")
# print(f"Финальная стоимость портфеля: {train2_value:,.0f}₽")
# print(f"PnL за самостоятельную торговлю: {train2_value - 10000:+,.0f}₽")

train2_steps, train2_reward, train2_value = 0, 0.0, 10000.0
print("\n⚠ ЭТАП 2 ОТКЛЮЧЕН — переходим сразу к валидации")


# ============================================================
# ЭТАП 3: ВАЛИДАЦИЯ (out-of-sample, без обучения)
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 3: ВАЛИДАЦИЯ (out-of-sample)")
print("=" * 70)

with open("data/portfolio_state.json", "w", encoding="utf-8") as f:
    json.dump({
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
    }, f, indent=2, default=str)

print("Портфель сброшен для валидации")

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
# ЭТАП 4: СТРЕСС-ТЕСТ 1 — СВО 2022
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 4: СТРЕСС-ТЕСТ — СВО 2022")
print("=" * 70)

with open("data/portfolio_state.json", "w", encoding="utf-8") as f:
    json.dump({
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
    }, f, indent=2, default=str)

print("Портфель сброшен для стресс-теста СВО")

print("Загрузка исторических макро-данных для стресс-теста СВО...")

stress_svo_macro = load_historical_macro(STRESS_SVO_START, STRESS_SVO_END)
print(f"Загружено макро-данных за {len(stress_svo_macro)} дней")

stress_svo_steps, stress_svo_reward, stress_svo_value = train_on_period(
    TICKERS, STRESS_SVO_START, STRESS_SVO_END, stress_svo_macro, learn=False
)

print(f"\nСтресс-тест СВО завершён: {stress_svo_steps} шагов", end="")
if stress_svo_steps > 0:
    print(f", средний reward={stress_svo_reward/stress_svo_steps:+.4f}")
else:
    print()
print(f"Финальная стоимость портфеля: {stress_svo_value:,.0f}₽")
print(f"PnL за стресс-тест СВО: {stress_svo_value - 10000:+,.0f}₽")

# ============================================================
# ЭТАП 5: СТРЕСС-ТЕСТ 2 — Ковид 2020
# ============================================================
print("\n" + "=" * 70)
print("ЭТАП 5: СТРЕСС-ТЕСТ — КОВИД 2020")
print("=" * 70)

with open("data/portfolio_state.json", "w", encoding="utf-8") as f:
    json.dump({
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
    }, f, indent=2, default=str)

print("Портфель сброшен для стресс-теста Ковид")

print("Загрузка исторических макро-данных для стресс-теста Ковид...")

stress_covid_macro = load_historical_macro(STRESS_COVID_START, STRESS_COVID_END)
print(f"Загружено макро-данных за {len(stress_covid_macro)} дней")

stress_covid_steps, stress_covid_reward, stress_covid_value = train_on_period(
    TICKERS, STRESS_COVID_START, STRESS_COVID_END, stress_covid_macro, learn=False
)

print(f"\nСтресс-тест Ковид завершён: {stress_covid_steps} шагов", end="")
if stress_covid_steps > 0:
    print(f", средний reward={stress_covid_reward/stress_covid_steps:+.4f}")
else:
    print()
print(f"Финальная стоимость портфеля: {stress_covid_value:,.0f}₽")
print(f"PnL за стресс-тест Ковид: {stress_covid_value - 10000:+,.0f}₽")


# ============================================================
# РЕЗУЛЬТИРУЮЩИЙ ЛОГ
# ============================================================
print("\n" + "=" * 70)
print("📊 РЕЗУЛЬТИРУЮЩИЙ ЛОГ ПРЕДОБУЧЕНИЯ")
print("=" * 70)

# Подсчёт lookahead-статистики (если есть в памяти)
lookahead_matches = 0
lookahead_total = 0
for exp in model.memory:
    if isinstance(exp, dict):
        sent_data = exp.get('sentiment_data', {})
        if isinstance(sent_data, dict) and 'lookahead_match' in sent_data:
            lookahead_total += 1
            if sent_data['lookahead_match']:
                lookahead_matches += 1

print(f"\n🧠 ПАМЯТЬ:")
print(f"   Всего опытов: {len(model.memory)}")
if lookahead_total > 0:
    print(f"   Lookahead accuracy: {lookahead_matches}/{lookahead_total} ({lookahead_matches/lookahead_total*100:.1f}%)")
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

        # 🆕 Диагностика дисбаланса
        total = len(actions)
        max_action = max(action_counts, key=action_counts.get)
        max_pct = action_counts[max_action] / total * 100
        if max_pct > 60:
            print(f"\n   ⚠️ ПРЕДУПРЕЖДЕНИЕ: действие {action_names.get(max_action, max_action)} "
                  f"доминирует ({max_pct:.1f}%). Возможен softmax collapse.")
            print(f"   Рекомендация: увеличьте entropy_bonus_coeff или temperature в trader_model.py")
        elif max_pct > 40:
            print(f"\n   ⚠️ ВНИМАНИЕ: действие {action_names.get(max_action, max_action)} "
                  f"преобладает ({max_pct:.1f}%). Следите за распределением после обучения.")

print(f"\n📈 ЭТАП 1: ПРИНУДИТЕЛЬНОЕ ОБУЧЕНИЕ:")
print(f"   Шагов: {train_steps}")
print(f"   Средний reward: {train_reward/train_steps:+.4f}" if train_steps > 0 else "   Нет шагов")
print(f"   Портфель: {train_value:,.0f}₽")

print(f"\n📈 ЭТАП 2: САМОСТОЯТЕЛЬНАЯ ТОРГОВЛЯ:")
print(f"   Шагов: {train2_steps}")
print(f"   Средний reward: {train2_reward/train2_steps:+.4f}" if train2_steps > 0 else "   Нет шагов")
print(f"   Портфель: {train2_value:,.0f}₽")
print(f"   PnL: {train2_value - 10000:+,.0f}₽ ({(train2_value/10000 - 1)*100:+.2f}%)")

print(f"\n📉 ВАЛИДАЦИЯ (out-of-sample):")
print(f"   Шагов: {val_steps}")
print(f"   Средний reward: {val_reward/val_steps:+.4f}" if val_steps > 0 else "   Нет шагов")
print(f"   Портфель: {val_value:,.0f}₽")
print(f"   PnL: {val_value - 10000:+,.0f}₽ ({(val_value/10000 - 1)*100:+.2f}%)")

print(f"\n💥 СТРЕСС-ТЕСТ СВО 2022:")
print(f"   Шагов: {stress_svo_steps}")
print(f"   Средний reward: {stress_svo_reward/stress_svo_steps:+.4f}" if stress_svo_steps > 0 else "   Нет шагов")
print(f"   Портфель: {stress_svo_value:,.0f}₽")
print(f"   PnL: {stress_svo_value - 10000:+,.0f}₽ ({(stress_svo_value/10000 - 1)*100:+.2f}%)")

print(f"\n💥 СТРЕСС-ТЕСТ КОВИД 2020:")
print(f"   Шагов: {stress_covid_steps}")
print(f"   Средний reward: {stress_covid_reward/stress_covid_steps:+.4f}" if stress_covid_steps > 0 else "   Нет шагов")
print(f"   Портфель: {stress_covid_value:,.0f}₽")
print(f"   PnL: {stress_covid_value - 10000:+,.0f}₽ ({(stress_covid_value/10000 - 1)*100:+.2f}%)")

print(f"\n🎯 ВЕСА МОДЕЛИ:")
action_net_bias = model.policy_net.action_net[2].bias.detach().cpu().numpy()
action_names = ["HOLD_SHORT", "HOLD", "HOLD_LONG", "BUY_SMALL", "BUY_NORMAL", "SELL_SMALL", "SELL_ALL"]
print(f"   Bias выходного слоя:")
for i, (name, bias) in enumerate(zip(action_names, action_net_bias)):
    direction = "📈" if bias > 0 else "📉"
    print(f"   {direction} {name:<15} bias={bias:+.4f}")

print(f"\n🧪 ТЕСТОВЫЙ ПРОГОН (на реальных state из памяти):")
# 🆕 v8: Тестируем на 5 реальных state из памяти вместо синтетического
# + Проверяем различимость state (state_features должны быть РАЗНЫМИ для разных state)
import random as _rnd
if len(model.memory) >= 10:
    test_indices = _rnd.sample(range(len(model.memory)), min(5, len(model.memory)))
    strategy_params = list(model.strategies.values())[0] if model.strategies else {}
    model.policy_net.eval()

    all_sf_means = []
    all_sf_stds = []
    all_probs = []

    for test_idx, mem_idx in enumerate(test_indices, 1):
        exp = model.memory[mem_idx]
        if not isinstance(exp, dict):
            continue
        state_tensor = exp['state'].to(model.device)
        actual_action = exp.get('action', -1)
        actual_reward = exp.get('reward', 0)

        # 🆕 v8: Проверяем, что state в памяти разные (не дубликаты)
        state_np = state_tensor.cpu().numpy()
        state_stats = {
            'mean': float(state_np.mean()),
            'std': float(state_np.std()),
            'min': float(state_np.min()),
            'max': float(state_np.max()),
            'n_nonzero': int((state_np != 0).sum()),
            'n_unique': len(set(state_np.round(6).tolist())),
        }

        with torch.no_grad():
            probs, value, _ = model.policy_net(state_tensor.unsqueeze(0))
            probs = probs.cpu().numpy().flatten()
            state_features = model.policy_net.state_net(state_tensor.unsqueeze(0))
            sf_stats = {
                'mean': state_features.mean().item(),
                'std': state_features.std().item(),
                'min': state_features.min().item(),
                'max': state_features.max().item(),
            }

        all_sf_means.append(sf_stats['mean'])
        all_sf_stds.append(sf_stats['std'])
        all_probs.append(probs.tolist())

        action_names_local = ["HOLD_SHORT", "HOLD", "HOLD_LONG", "BUY_SMALL", "BUY_NORMAL",
                              "SELL_SMALL", "SELL_ALL"]
        chosen = action_names_local[actual_action] if 0 <= actual_action < 7 else str(actual_action)

        print(f"\n  Тест {test_idx}: action={chosen}, reward={actual_reward:+.2f}")
        print(f"   State value: {value.item():+.4f}")
        print(f"   Input state: mean={state_stats['mean']:+.4f}, std={state_stats['std']:.4f}, "
              f"nonzero={state_stats['n_nonzero']}/{len(state_np)}, unique={state_stats['n_unique']}")
        print(f"   State features: mean={sf_stats['mean']:+.4f}, std={sf_stats['std']:.4f}, "
              f"range=[{sf_stats['min']:+.3f}, {sf_stats['max']:+.3f}]")
        print(f"   Распределение вероятностей:")
        for i, p in enumerate(probs):
            bar = "█" * int(p * 30)
            marker = " ← actual" if i == actual_action else ""
            print(f"   {action_names_local[i]:<15} {p:.3f} {bar}{marker}")

    # 🆕 v8: Диагностика различимости state
    print(f"\n  📊 ДИАГНОСТИКА РАЗЛИЧИМОСТИ STATE:")
    print(f"   State features mean по 5 тестам: {[f'{m:+.4f}' for m in all_sf_means]}")
    print(f"   State features std  по 5 тестам: {[f'{s:.4f}' for s in all_sf_stds]}")
    sf_mean_range = max(all_sf_means) - min(all_sf_means)
    sf_std_range = max(all_sf_stds) - min(all_sf_stds)
    print(f"   Range of means: {sf_mean_range:.6f}")
    print(f"   Range of stds:  {sf_std_range:.6f}")

    # Проверяем, различает ли модель state
    probs_diffs = []
    for i in range(len(all_probs)):
        for j in range(i+1, len(all_probs)):
                diff = sum(abs(a - b) for a, b in zip(all_probs[i], all_probs[j])) / len(all_probs[i])
                probs_diffs.append(diff)
    avg_prob_diff = sum(probs_diffs) / len(probs_diffs) if probs_diffs else 0
    print(f"   Avg probability difference between states: {avg_prob_diff:.6f}")

    if sf_mean_range < 0.001 and sf_std_range < 0.001:
        print(f"   ❌ State_net НЕ РАЗЛИЧАЕТ state — все features одинаковые!")
        print(f"      Причина: LayerNorm + Dropout коллапсировали state_net")
        print(f"      Решение: проверьте архитектуру state_net (уберите LayerNorm)")
    elif avg_prob_diff < 0.01:
        print(f"   ⚠️ State различимы, но model выдаёт почти одинаковые probabilities")
        print(f"      Причина: policy gradient не работает или temperature слишком высокая")
    else:
        print(f"   ✓ State различимы, model выдаёт разные probabilities")
else:
    print("   Недостаточно опытов в памяти для теста")

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