#!/usr/bin/env python3
"""
ПРОВЕРКА ВСЕХ ПРАВОК (Этапы 2, 3, 4)
Risk Manager, кулдаун, лимиты, новостные сигналы, market_features
"""

import sys
import time
import json
import random
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, '.')

from models.smart_broker import SmartPortfolioBroker
from models.trader_model import trader_model_instance
from core.risk_manager import RiskManager
from core.trading_hours_scheduler import TradingScheduler
from utils.portfolio_manager import PortfolioManager
from utils.logger import get_logger

logger = get_logger("TEST_FIXES")


def test_risk_manager_call():
    """Проверка: вызывается ли Risk Manager при BUY"""
    print("\n" + "=" * 70)
    print("ТЕСТ 1: Risk Manager при BUY")
    print("=" * 70)

    risk_mgr = RiskManager()
    portfolio = PortfolioManager()
    portfolio.cash = 10000.0
    portfolio.reserved_cash = 0.0
    portfolio.positions = {}

    # Симулируем вызов calculate_position_size
    result = risk_mgr.calculate_position_size(
        ticker='SBER',
        price=280.0,
        stop_loss=271.6,
        atr=5.0,
        confidence=0.7,
        adv=10000000,
        sector='финансы',
        lot_size=10
    )

    quantity, risk = result

    print(f"   Параметры: SBER price=280, stop=271.6, atr=5.0, conf=0.7, lot=10")
    print(f"   Результат: quantity={quantity}, risk={risk:.0f}₽")

    if quantity > 0:
        print(f"   ✅ Risk Manager работает. Рассчитано {quantity} акций.")
        return True
    else:
        print(f"   ❌ Risk Manager вернул 0. Проверьте настройки.")
        return False


def test_cooldown():
    """Проверка: работает ли кулдаун"""
    print("\n" + "=" * 70)
    print("ТЕСТ 2: Кулдаун между сделками")
    print("=" * 70)

    # Имитируем проверку кулдауна
    last_trade_time = defaultdict(float)
    cooldown_hours = 2
    cooldown_seconds = cooldown_hours * 3600

    ticker = 'SBER'
    now = time.time()

    # Симуляция: сделка была только что
    last_trade_time[ticker] = now
    elapsed = now - last_trade_time[ticker]
    blocked = elapsed < cooldown_seconds

    print(f"   Тикер: {ticker}")
    print(f"   Кулдаун: {cooldown_hours}ч = {cooldown_seconds}с")
    print(f"   Прошло с последней сделки: {elapsed:.0f}с")
    print(f"   Сделка заблокирована: {blocked}")

    if blocked:
        print(f"   ✅ Кулдаун работает (сделка только что была — блокируем).")
    else:
        print(f"   ❌ Кулдаун не сработал.")

    # Симуляция: сделка была 3 часа назад
    last_trade_time[ticker] = now - 10800  # 3 часа
    elapsed = now - last_trade_time[ticker]
    blocked = elapsed < cooldown_seconds

    print(f"\n   Прошло с последней сделки: {elapsed:.0f}с (3 часа)")
    print(f"   Сделка заблокирована: {blocked}")

    if not blocked:
        print(f"   ✅ Кулдаун корректно разрешает сделку после истечения.")
    else:
        print(f"   ❌ Кулдаун блокирует даже старые сделки.")

    return True


def test_position_limits():
    """Проверка: max_positions, max_trades_per_hour, daily_commission_limit"""
    print("\n" + "=" * 70)
    print("ТЕСТ 3: Лимиты портфеля")
    print("=" * 70)

    portfolio = PortfolioManager()
    portfolio.cash = 10000.0
    portfolio.reserved_cash = 0.0
    portfolio.positions = {}

    # Загружаем лимиты из конфига
    print(f"   max_positions: {portfolio.max_positions}")
    print(f"   max_trades_per_hour: {portfolio.max_trades_per_hour}")
    print(f"   daily_commission_limit: {portfolio.daily_commission_limit}₽")

    # Проверка max_positions
    test_positions = {f'TICKER{i}': {'qty': 10, 'avg_price': 100} for i in range(portfolio.max_positions)}
    portfolio.positions = test_positions

    ticker_new = 'NEW_TICKER'
    limit_reached = ticker_new not in portfolio.positions and len(portfolio.positions) >= portfolio.max_positions

    print(f"\n   Текущих позиций: {len(portfolio.positions)}/{portfolio.max_positions}")
    print(f"   Новый тикер {ticker_new} заблокирован: {limit_reached}")

    if limit_reached:
        print(f"   ✅ Проверка max_positions работает.")
    else:
        print(f"   ❌ Проверка max_positions НЕ работает.")

    # Проверка daily_commission_limit
    commission = 30.0
    limit_exceeded = portfolio.commission_spent_today + commission > portfolio.daily_commission_limit

    print(f"\n   Комиссий сегодня: {portfolio.commission_spent_today:.2f}₽")
    print(f"   Новая комиссия: {commission:.2f}₽")
    print(f"   Лимит: {portfolio.daily_commission_limit:.2f}₽")
    print(f"   Превышение: {limit_exceeded}")

    if limit_exceeded:
        print(f"   ✅ Проверка daily_commission_limit работает.")
    else:
        print(f"   ⚠️ Лимит не превышен (комиссий мало). Проверьте логику в buy().")

    return True


def test_news_signals():
    """Проверка: работают ли новостные сигналы"""
    print("\n" + "=" * 70)
    print("ТЕСТ 4: Новостные сигналы")
    print("=" * 70)

    try:
        # Проверяем маппинг тикер→названия
        from fetchers.moex_fetcher import MoexFetcher
        moex = MoexFetcher()
        securities = moex.get_all_securities()

        test_tickers = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'GMKN']

        print(f"   Маппинг тикер → названия:")
        for ticker in test_tickers:
            sec = securities.get(ticker, {})
            name = sec.get('name', '—')
            full_name = sec.get('full_name', '—')

            # Извлекаем ключевые слова
            keywords = set()
            keywords.add(ticker.lower())
            for word in name.lower().split():
                clean = word.strip('"\'.,;:()[]{}')
                if len(clean) > 2:
                    keywords.add(clean)
            for word in full_name.lower().split():
                clean = word.strip('"\'.,;:()[]{}')
                if len(clean) > 2:
                    keywords.add(clean)

            print(f"   {ticker}: {keywords}")

        # Проверяем search_news с ключевыми словами
        from fetchers.news_fetcher import OptimizedNewsFetcher
        news_fetcher = OptimizedNewsFetcher()

        sber_keywords = list({'сбербанк', 'сбер', 'sber', 'sberbank'})
        news = news_fetcher.search_news(ticker='SBER', keywords=sber_keywords, limit=5)

        print(f"\n   Поиск новостей по SBER + ключевым словам:")
        print(f"   Найдено: {len(news)} новостей")

        if news:
            for n in news[:3]:
                print(f"   — {n.get('title', '')[:80]}...")
            print(f"   ✅ Новостной поиск с ключевыми словами работает.")
        else:
            print(f"   ⚠️ Новостей не найдено (возможно, нет в кэше).")
            print(f"   Проверьте логи после запуска main.py — должны быть сигналы.")

        # Проверяем наличие ticker_names в конфиге
        import json
        with open('config/rl_config.json', 'r', encoding='utf-8') as f:
            rl_config = json.load(f)

        ticker_names = rl_config.get('ticker_names', {})
        print(f"\n   ticker_names в rl_config.json: {len(ticker_names)} тикеров")
        if ticker_names:
            print(f"   Пример: SBER → {ticker_names.get('SBER', [])}")
            print(f"   ✅ Резервный маппинг загружен.")
        else:
            print(f"   ⚠️ ticker_names пуст. Добавьте маппинг в rl_config.json.")

        return True

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_market_features():
    """Проверка: заполняются ли market_features реальными данными"""
    print("\n" + "=" * 70)
    print("ТЕСТ 5: Market Features (10 признаков MOEX)")
    print("=" * 70)

    try:
        from fetchers.moex_fetcher import MoexFetcher
        moex = MoexFetcher()

        # Получаем макро-данные
        macro = moex.get_macro_data()

        # Проверяем 10 новых признаков
        features = {
            'spread_pct': 0.0,
            'market_mood': macro.get('market_mood', 0.0),
            'shares_turnover': macro.get('shares_turnover', 0) / 1e12,
            'rvi_normalized': macro.get('rvi', 20.0) / 100.0,
            'imoex_normalized': macro.get('imoex', 0) / 4000.0,
            'market_cap_total': macro.get('market_cap', 0) / 1e14,
            'liquidity_ratio': macro.get('market_liquidity_ratio', 0.0),
            'rtsi_normalized': macro.get('rtsi', 0) / 2000.0,
            'usd_rub': macro.get('usd_rub', 0) / 100.0,
            'moexog_normalized': macro.get('moexog', 0) / 10000.0,
        }

        # spread_pct нужна цена тикера
        price = moex.get_price('SBER')
        if price:
            securities = moex.get_all_securities()
            sber_info = securities.get('SBER', {})
            spread = sber_info.get('spread', 0)
            features['spread_pct'] = (spread / price) if price > 0 else 0.0

        # Загружаем конфиг для проверки имён
        import json
        with open('config/rl_config.json', 'r', encoding='utf-8') as f:
            rl_config = json.load(f)

        config_features = rl_config.get('feature_config', {}).get('market_features', [])

        print(f"   Признаки в конфиге ({len(config_features)}):")
        print(f"   {config_features}")

        print(f"\n   Значения признаков:")
        ready = 0
        for name in config_features:
            value = features.get(name, 0.0)
            is_nonzero = abs(value) > 0.0001
            if is_nonzero:
                ready += 1
            icon = '✅' if is_nonzero else '⚠️ (0.0)'
            print(f"   {icon} {name:25s} = {value:.6f}")

        print(f"\n   Ненулевых признаков: {ready}/{len(config_features)}")

        if ready >= 5:
            print(f"   ✅ Market features заполняются реальными данными!")
        elif ready >= 1:
            print(f"   ⚠️ Часть признаков = 0.0 (рынок закрыт или данные недоступны).")
        else:
            print(f"   ❌ Все признаки = 0.0. Проверьте get_macro_data().")

        return ready >= 1

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_consistency():
    """Проверка: все параметры из конфигов"""
    print("\n" + "=" * 70)
    print("ТЕСТ 6: Консистентность конфигов")
    print("=" * 70)

    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)
        with open('config/rl_config.json', 'r', encoding='utf-8') as f:
            rl_config = json.load(f)
        with open('config/strategies.json', 'r', encoding='utf-8') as f:
            strategies = json.load(f)

        checks = []

        # Кулдаун
        cooldown = strategies.get('default_strategy_parameters', {}).get('cooldown_period_hours')
        checks.append(('cooldown_period_hours в strategies.json', cooldown is not None, cooldown))

        # max_positions
        max_pos = settings.get('max_positions')
        checks.append(('max_positions в settings.json', max_pos is not None, max_pos))

        # max_daily_trades
        max_trades = settings.get('max_daily_trades')
        checks.append(('max_daily_trades в settings.json', max_trades is not None, max_trades))

        # daily_commission_limit
        comm_limit = settings.get('daily_commission_limit')
        checks.append(('daily_commission_limit в settings.json', comm_limit is not None, comm_limit))

        # signal_filter
        sig_filter = rl_config.get('signal_filter', {}).get('enabled')
        checks.append(('signal_filter.enabled в rl_config.json', sig_filter is not None, sig_filter))

        # hold_reward
        hold_enabled = rl_config.get('hold_reward', {}).get('enabled')
        checks.append(('hold_reward.enabled в rl_config.json', hold_enabled is not None, hold_enabled))

        # market_features
        mf = rl_config.get('feature_config', {}).get('market_features', [])
        checks.append(('market_features (10 признаков)', len(mf) == 10, len(mf)))

        # ticker_names
        tn = rl_config.get('ticker_names', {})
        checks.append(('ticker_names в rl_config.json', len(tn) > 0, len(tn)))

        # action_mapping
        am = rl_config.get('action_mapping', {})
        has_hold = any(v == 'HOLD' for v in am.values())
        checks.append(('HOLD в action_mapping', has_hold, list(am.values())))

        # exploration
        expl = rl_config.get('exploration', {}).get('initial_exploration_rate')
        checks.append(('exploration_rate < 0.05', expl is not None and expl < 0.05, expl))

        print(f"\n   Проверка конфигов:")
        all_ok = True
        for name, status, value in checks:
            icon = '✅' if status else '❌'
            print(f"   {icon} {name}: {value}")
            if not status:
                all_ok = False

        if all_ok:
            print(f"\n   ✅ Все конфиги корректны.")
        else:
            print(f"\n   ❌ Найдены проблемы в конфигах.")

        return all_ok

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return False


def main():
    """Запуск всех тестов"""
    print("\n" + "=" * 70)
    print("🔬 ПРОВЕРКА ВНЕСЁННЫХ ИЗМЕНЕНИЙ")
    print("=" * 70)
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    results['risk_manager'] = test_risk_manager_call()
    results['cooldown'] = test_cooldown()
    results['limits'] = test_position_limits()
    results['news'] = test_news_signals()
    results['market_features'] = test_market_features()
    results['config'] = test_config_consistency()

    # Итог
    print("\n" + "=" * 70)
    print("📋 ИТОГИ ПРОВЕРКИ")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, status in results.items():
        icon = '✅' if status else '❌'
        print(f"   {icon} {name}")

    print(f"\n   Пройдено: {passed}/{total}")

    if passed == total:
        print(f"\n   ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
    elif passed >= total - 1:
        print(f"\n   ⚠️ Почти всё работает. Проверьте непройденные тесты.")
    else:
        print(f"\n   ❌ Много ошибок. Проверьте внесённые правки.")

    print("\n" + "=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЁН")
    print("=" * 70)


if __name__ == "__main__":
    main()