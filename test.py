#!/usr/bin/env python3
"""
Диагностика состояния системы без остановки
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.smart_broker import trader_model_instance
from core.trading_hours_scheduler import TradingScheduler
import json
from datetime import datetime
import pytz


def diagnose():
    print("=" * 60)
    print("ДИАГНОСТИКА СИСТЕМЫ")
    print("=" * 60)

    # 1. Проверка scheduler
    scheduler = TradingScheduler()
    now = datetime.now(pytz.timezone('Europe/Moscow'))

    print(f"\n1. ТЕКУЩЕЕ ВРЕМЯ: {now.strftime('%H:%M:%S')}")
    print(f"   is_trading_day: {scheduler.is_trading_day(now)}")
    print(f"   is_trading_time: {scheduler.is_trading_time(now)}")

    # 2. Проверка периода MOEX
    period = scheduler.get_current_moex_period()
    print(f"   current_period: {period}")
    print(
        f"   can_place_orders: {period in ['auction_open', 'continuous_trading', 'auction_close', 'evening_auction_open', 'evening_continuous']}")

    # 3. Проверка флага trading_enabled в модели
    if hasattr(trader_model_instance, 'trading_enabled'):
        print(f"\n2. trading_enabled в модели: {trader_model_instance.trading_enabled}")
    else:
        print("\n2. trading_enabled отсутствует в модели")

    # 4. Проверка pending_experiences
    if hasattr(trader_model_instance, 'pending_experiences'):
        print(f"\n3. pending_experiences: {len(trader_model_instance.pending_experiences)}")

    # 5. Проверка памяти модели
    if hasattr(trader_model_instance, 'memory'):
        print(f"\n4. memory size: {len(trader_model_instance.memory)}")

    # 6. Проверка стратегий
    if hasattr(trader_model_instance, 'strategy_performance'):
        print("\n5. Статистика стратегий:")
        for strat, perf in trader_model_instance.strategy_performance.items():
            print(f"   {strat}: trades={perf['total_trades']}, win_rate={perf['win_rate']:.2%}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    diagnose()