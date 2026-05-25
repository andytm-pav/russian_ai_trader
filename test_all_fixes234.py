#!/usr/bin/env python3
"""
ПРОВЕРКА ВСЕХ 8 ЭТАПОВ С ПРОВЕРКОЙ РАЗМЕРНОСТИ МОДЕЛИ
"""

import sys
import time
import json
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, '.')

from models.trader_model import trader_model_instance
from core.risk_manager import RiskManager
from utils.portfolio_manager import PortfolioManager
from utils.logger import get_logger

logger = get_logger("TEST_8_STAGES")


def test_stage_1_model_dimension():
    """Этап 1: Проверка размерности модели"""
    print("\n" + "=" * 70)
    print("ЭТАП 1: РАЗМЕРНОСТЬ МОДЕЛИ")
    print("=" * 70)

    model = trader_model_instance

    # Заявленные размерности
    state_vector_size = model.rl_config.get('state_parameters', {}).get('state_vector_size', 0)
    total_state_size = model.rl_config.get('state_parameters', {}).get('total_state_size', 0)
    strategy_params_size = model.rl_config.get('state_parameters', {}).get('strategy_params_size', 0)

    print(f"   state_vector_size (заявлено): {state_vector_size}")
    print(f"   total_state_size (заявлено): {total_state_size}")
    print(f"   strategy_params_size: {strategy_params_size}")

    # Фактическая размерность из кода
    state_dimensions = model.rl_config.get('state_dimensions', {})
    config_sum = sum(state_dimensions.values())
    print(f"   Сумма state_dimensions: {config_sum}")

    # Резервные слоты и market_features
    feature_config = model.rl_config.get('feature_config', {})
    reserved_slots = feature_config.get('reserved_slots', 0)
    market_features_count = len(feature_config.get('market_features', []))
    liquidity_features = 2

    expected_total = config_sum + reserved_slots + market_features_count + liquidity_features
    print(f"   reserved_slots: {reserved_slots}")
    print(f"   market_features: {market_features_count}")
    print(f"   liquidity: {liquidity_features}")
    print(f"   Ожидаемая размерность: {expected_total}")

    # Проверка bias последнего слоя (должен быть 7)
    bias = model.policy_net.action_net[2].bias.data.cpu().numpy()
    action_dim = len(bias)
    config_action_dim = model.rl_config.get('action_dim', 0)

    print(f"\n   action_dim (конфиг): {config_action_dim}")
    print(f"   action_dim (веса): {action_dim}")

    # Проверка гистерезиса и горизонта в reserved_slots
    if reserved_slots == 0 and state_vector_size == 210:
        print(f"\n   ✅ Гистерезис и горизонты встроены (reserved_slots=0, размерность=210)")
        hysteresis_ok = True
    elif reserved_slots == 6:
        print(f"\n   ⚠️ reserved_slots=6 — гистерезис и горизонты не встроены")
        hysteresis_ok = False
    else:
        print(f"\n   ⚠️ Неожиданная конфигурация reserved_slots={reserved_slots}")
        hysteresis_ok = False

    # Итог
    all_ok = (state_vector_size == expected_total and
              action_dim == config_action_dim and
              state_vector_size == 210)

    if all_ok:
        print(f"\n   ✅ РАЗМЕРНОСТЬ КОРРЕКТНА")
    else:
        print(f"\n   ❌ РАЗМЕРНОСТЬ НЕ СОВПАДАЕТ!")

    return {
        'state_vector_size': state_vector_size,
        'total_state_size': total_state_size,
        'action_dim': action_dim,
        'reserved_slots': reserved_slots,
        'hysteresis_ok': hysteresis_ok,
        'all_ok': all_ok
    }


def test_stage_2_risk_manager():
    """Этап 2: Risk Manager + кулдаун + лимиты"""
    print("\n" + "=" * 70)
    print("ЭТАП 2: RISK MANAGER + КУЛДАУН + ЛИМИТЫ")
    print("=" * 70)

    results = {}

    # 2.1 Risk Manager
    risk_mgr = RiskManager()
    quantity, risk = risk_mgr.calculate_position_size(
        ticker='SBER',
        price=280.0,
        stop_loss=271.6,
        atr=3.0,
        confidence=0.7,
        adv=50000000,
        sector='финансы',
        lot_size=10
    )
    results['risk_manager'] = quantity > 0
    print(f"   2.1 Risk Manager: quantity={quantity}, risk={risk:.0f}₽ — {'✅' if quantity > 0 else '❌'}")

    # 2.2 Кулдаун
    with open('config/settings.json', 'r') as f:
        settings = json.load(f)
    cooldown = settings.get('cooldown_seconds', 0)
    results['cooldown'] = cooldown > 0
    print(f"   2.2 Кулдаун: cooldown_seconds={cooldown} — {'✅' if cooldown > 0 else '❌'}")

    # 2.3 max_positions
    portfolio = PortfolioManager()
    results['max_positions'] = hasattr(portfolio, 'max_positions') and portfolio.max_positions > 0
    print(
        f"   2.3 max_positions: {getattr(portfolio, 'max_positions', 'НЕТ')} — {'✅' if results['max_positions'] else '❌'}")

    # 2.4 trades_lookback
    lookback = settings.get('trades_lookback_seconds', 0)
    results['trades_lookback'] = lookback > 0
    print(f"   2.4 trades_lookback_seconds: {lookback} — {'✅' if lookback > 0 else '❌'}")

    # 2.5 daily_commission_limit
    comm_limit = settings.get('daily_commission_limit', 0)
    results['commission_limit'] = comm_limit > 0
    print(f"   2.5 daily_commission_limit: {comm_limit} — {'✅' if comm_limit > 0 else '❌'}")

    # 2.6 max_positions_per_horizon
    max_per_horizon = settings.get('max_positions_per_horizon', {})
    results['horizon_limits'] = len(max_per_horizon) > 0
    print(f"   2.6 max_positions_per_horizon: {max_per_horizon} — {'✅' if len(max_per_horizon) > 0 else '❌'}")

    all_ok = all(results.values())
    print(f"\n   Итог: {'✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_ok else '❌ ЕСТЬ ПРОБЛЕМЫ'}")

    return results


def test_stage_3_news_signals():
    """Этап 3: Новостные сигналы"""
    print("\n" + "=" * 70)
    print("ЭТАП 3: НОВОСТНЫЕ СИГНАЛЫ")
    print("=" * 70)

    results = {}

    # 3.1 ticker_names в конфиге
    with open('config/rl_config.json', 'r') as f:
        rl_config = json.load(f)

    ticker_names = rl_config.get('ticker_names', {})
    results['ticker_names'] = len(ticker_names) > 0
    print(f"   3.1 ticker_names: {len(ticker_names)} тикеров — {'✅' if len(ticker_names) > 0 else '❌'}")

    # 3.2 sentiment_threshold
    sentiment_config = rl_config.get('sentiment_integration', {})
    threshold = sentiment_config.get('ticker_sentiment_weight', 0)
    results['sentiment_threshold'] = threshold > 0
    print(f"   3.2 sentiment_threshold: {threshold} — {'✅' if threshold > 0 else '❌'}")

    # 3.3 search_news с keywords
    from fetchers.news_fetcher import OptimizedNewsFetcher
    news_fetcher = OptimizedNewsFetcher()
    import inspect
    sig = inspect.signature(news_fetcher.search_news)
    has_keywords = 'keywords' in sig.parameters
    results['search_keywords'] = has_keywords
    print(f"   3.3 search_news(keywords=): {'✅' if has_keywords else '❌'}")

    # 3.4 Маппинг тикеров из MOEX
    from fetchers.moex_fetcher import MoexFetcher
    moex = MoexFetcher()
    securities = moex.get_all_securities()
    sber = securities.get('SBER', {})
    has_russian_name = bool(sber.get('name', '') or sber.get('full_name', ''))
    results['russian_names'] = has_russian_name
    print(f"   3.4 Русские названия в MOEX: {'✅' if has_russian_name else '❌'}")

    all_ok = all(results.values())
    print(f"\n   Итог: {'✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_ok else '❌ ЕСТЬ ПРОБЛЕМЫ'}")

    return results


def test_stage_4_market_features():
    """Этап 4: Market Features MOEX"""
    print("\n" + "=" * 70)
    print("ЭТАП 4: MARKET FEATURES MOEX")
    print("=" * 70)

    results = {}

    # 4.1 Список в конфиге
    with open('config/rl_config.json', 'r') as f:
        rl_config = json.load(f)

    mf = rl_config.get('feature_config', {}).get('market_features', [])
    results['config_list'] = len(mf) == 10
    print(f"   4.1 market_features: {len(mf)} признаков — {'✅' if len(mf) == 10 else '❌'}")
    print(f"       {mf}")

    # 4.2 Нормализация в конфиге
    norm = rl_config.get('normalization', {})
    divisors = ['shares_turnover_divisor', 'rvi_divisor', 'imoex_divisor',
                'market_cap_divisor_total', 'rtsi_divisor', 'usd_rub_divisor', 'moexog_divisor']
    norm_ok = all(d in norm for d in divisors)
    results['normalization'] = norm_ok
    print(f"   4.2 Делители нормализации: {'✅ все 7' if norm_ok else '❌ не хватает'}")

    # 4.3 Реальные данные
    from fetchers.moex_fetcher import MoexFetcher
    moex = MoexFetcher()
    macro = moex.get_macro_data()

    feature_values = {
        'spread_pct': 0.0,
        'market_mood': macro.get('market_mood', 0.0),
        'shares_turnover': macro.get('shares_turnover', 0),
        'rvi_normalized': macro.get('rvi', 20.0),
        'imoex_normalized': macro.get('imoex', 0),
        'market_cap_total': macro.get('market_cap', 0),
        'liquidity_ratio': macro.get('market_liquidity_ratio', 0.0),
        'rtsi_normalized': macro.get('rtsi', 0),
        'usd_rub': macro.get('usd_rub', 0),
        'moexog_normalized': macro.get('moexog', 0),
    }

    nonzero = sum(1 for v in feature_values.values() if abs(v) > 0.0001)
    results['data_available'] = nonzero >= 5
    print(f"   4.3 Данные доступны: {nonzero}/10 признаков ненулевые — {'✅' if nonzero >= 5 else '❌'}")

    all_ok = all(results.values())
    print(f"\n   Итог: {'✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_ok else '❌ ЕСТЬ ПРОБЛЕМЫ'}")

    return results


def test_stage_5_hysteresis_horizons():
    """Этап 5: Гистерезис и горизонты"""
    print("\n" + "=" * 70)
    print("ЭТАП 5: ГИСТЕРЕЗИС И ВРЕМЕННЫЕ ГОРИЗОНТЫ")
    print("=" * 70)

    results = {}

    # 5.1 reserved_slots = 0
    with open('config/rl_config.json', 'r') as f:
        rl_config = json.load(f)

    reserved = rl_config.get('feature_config', {}).get('reserved_slots', -1)
    results['reserved_zero'] = reserved == 0
    print(f"   5.1 reserved_slots: {reserved} — {'✅' if reserved == 0 else '❌ (ожидается 0)'}")

    # 5.2 Нормализация гистерезиса
    norm = rl_config.get('normalization', {})
    has_hold_norm = 'max_hold_time_hours_norm' in norm
    has_streak_norm = 'same_action_streak_max' in norm
    results['hysteresis_norm'] = has_hold_norm and has_streak_norm
    print(f"   5.2 max_hold_time_hours_norm: {'✅' if has_hold_norm else '❌'}")
    print(f"       same_action_streak_max: {'✅' if has_streak_norm else '❌'}")

    # 5.3 horizon в стратегиях
    with open('config/strategies.json', 'r') as f:
        strategies = json.load(f)

    strats = strategies.get('strategies', {})
    horizons_found = set()
    for name, params in strats.items():
        h = params.get('horizon', 'НЕТ')
        horizons_found.add(h)

    expected_horizons = {'day_session', 'three_days', 'week'}
    has_all_horizons = expected_horizons.issubset(horizons_found)
    results['strategy_horizons'] = has_all_horizons
    print(f"   5.3 Горизонты стратегий: {horizons_found} — {'✅' if has_all_horizons else '❌'}")

    # 5.4 max_positions_per_horizon
    with open('config/settings.json', 'r') as f:
        settings = json.load(f)

    max_per_horizon = settings.get('max_positions_per_horizon', {})
    has_horizon_limits = all(h in max_per_horizon for h in ['day_session', 'three_days', 'week'])
    results['horizon_limits'] = has_horizon_limits
    print(f"   5.4 max_positions_per_horizon: {max_per_horizon} — {'✅' if has_horizon_limits else '❌'}")

    # 5.5 Проверка build_state_vector — гистерезис и горизонт в признаках
    model = trader_model_instance
    try:
        test_state = model.build_state_vector(
            ticker='SBER',
            price=280.0,
            momentum=0.0,
            sentiment=0.0,
            news_features=model.encode_news(['тестовая новость']),
            market_data={
                'volume': 1000000, 'spread': 0.01, 'rsi': 50,
                'sma_10_ratio': 1.0, 'sma_20_ratio': 1.0, 'bb_position': 0.5,
                'atr': 3.0, 'volume_ratio': 1.0, 'market_cap': 1e12,
                'lot_size': 10, 'min_step': 0.01, 'sector': 'финансы',
                'momentum': 0.0, 'imoex': 2600, 'imoex_change': 0,
                'rtsi': 1150, 'rtsi_change': 0, 'rvi': 21, 'rvi_change': 0,
                'moexog': 6700, 'moexfn': 0, 'brent': 80, 'brent_change': 0,
                'market_liquidity_ratio': 0.5, 'market_activity_score': 0.5,
                'market_mood': 0.0, 'shares_turnover': 1e10,
                'rvi_normalized': 0.21, 'imoex_normalized': 0.65,
                'market_cap_total': 0.5, 'liquidity_ratio': 0.5,
                'rtsi_normalized': 0.57, 'usd_rub': 0.8, 'moexog_normalized': 0.67,
                'spread_pct': 0.0001,
            },
            market_sentiment=0.0,
            portfolio=None
        )

        state_len = len(test_state)
        results['state_builds'] = state_len == 210
        print(f"   5.5 build_state_vector: {state_len} признаков — {'✅' if state_len == 210 else '❌ (ожидается 210)'}")

        # Проверяем, что позиции 192-197 не нули (для тикера без позиции — нули допустимы)
        state_np = test_state.cpu().numpy()
        hysteresis_slots = state_np[192:195]
        horizon_slots = state_np[195:198]
        print(f"       Гистерезис (192-194): {hysteresis_slots}")
        print(f"       Горизонт (195-197): {horizon_slots}")

    except Exception as e:
        results['state_builds'] = False
        print(f"   5.5 build_state_vector: ❌ ОШИБКА: {e}")

    all_ok = all(results.values())
    print(f"\n   Итог: {'✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_ok else '❌ ЕСТЬ ПРОБЛЕМЫ'}")

    return results


def test_stage_6_api_moex():
    """Этап 6: API MOEX (Brent, USD/RUB)"""
    print("\n" + "=" * 70)
    print("ЭТАП 6: API MOEX (BRENT, USD/RUB)")
    print("=" * 70)

    results = {}

    from fetchers.moex_fetcher import MoexFetcher
    moex = MoexFetcher()

    # 6.1 Brent контракты
    try:
        url = f"{moex.base_url}/engines/futures/markets/forts/securities.json"
        params = {
            'iss.meta': 'off',
            'iss.only': 'securities',
            'securities.columns': 'SECID,LASTTRADEDATE',
            'limit': 500
        }
        response = moex.session.get(url, params=params, timeout=10)
        data = response.json()
        brent_found = 0
        for row in data.get('securities', {}).get('data', []):
            cols = data['securities']['columns']
            secid = row[cols.index('SECID')]
            if secid.upper().startswith('BR'):
                brent_found += 1
        results['brent_contracts'] = brent_found > 0
        print(f"   6.1 Brent контракты: найдено {brent_found} — {'✅' if brent_found > 0 else '❌'}")
    except Exception as e:
        results['brent_contracts'] = False
        print(f"   6.1 Brent контракты: ❌ {str(e)[:50]}")

    # 6.2 USD/RUB
    usd_ok = False
    for instr in ['USD000000TOD', 'USD000UTSTOM']:
        try:
            url = f"{moex.base_url}/engines/currency/markets/selt/securities/{instr}.json"
            params = {'iss.meta': 'off', 'iss.only': 'marketdata', 'marketdata.columns': 'SECID,LAST'}
            response = moex.session.get(url, params=params, timeout=10)
            data = response.json()
            if 'marketdata' in data and data['marketdata']['data']:
                last = data['marketdata']['data'][0][1]
                if last:
                    usd_ok = True
                    break
        except:
            pass
    results['usd_rub'] = usd_ok
    print(f"   6.2 USD/RUB: {'✅ доступен' if usd_ok else '⚠️ недоступен (вне сессии)'}")

    # 6.3 market_mood в macro_data
    macro = moex.get_macro_data()
    has_mood = 'market_mood' in macro
    results['market_mood'] = has_mood
    print(f"   6.3 market_mood в macro_data: {'✅' if has_mood else '❌'}")

    all_ok = all(results.values())
    print(f"\n   Итог: {'✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_ok else '⚠️ ЧАСТИЧНО (USD/RUB может быть недоступен)'}")

    return results


def test_stage_7_action_mapping():
    """Этап 7: Архитектурный баланс действий"""
    print("\n" + "=" * 70)
    print("ЭТАП 7: ACTION MAPPING")
    print("=" * 70)

    results = {}

    with open('config/rl_config.json', 'r') as f:
        rl_config = json.load(f)

    action_mapping = rl_config.get('action_mapping', {})

    # Считаем действия
    buy_actions = sum(1 for v in action_mapping.values() if v.startswith('BUY'))
    hold_actions = sum(1 for v in action_mapping.values() if v == 'HOLD')
    sell_actions = sum(1 for v in action_mapping.values() if v.startswith('SELL'))

    print(f"   BUY: {buy_actions}, HOLD: {hold_actions}, SELL: {sell_actions}")
    print(f"   action_mapping: {action_mapping}")

    # Проверка: HOLD должно быть не менее 2 из 7 (для баланса)
    balanced = hold_actions >= 2
    results['balanced'] = balanced
    print(
        f"   7.1 Баланс (HOLD >= 2): {'✅' if balanced else '⚠️ HOLD только 1 (рекомендуется 3 HOLD / 2 BUY / 2 SELL)'}")

    # Проверка: action_dim из конфига
    action_dim_config = rl_config.get('action_dim', 0)
    action_dim_actual = len(action_mapping)
    results['dim_consistent'] = action_dim_config == action_dim_actual
    print(
        f"   7.2 action_dim (конфиг={action_dim_config}, mapping={action_dim_actual}): {'✅' if action_dim_config == action_dim_actual else '❌'}")

    # Проверка: exploration
    exploration = rl_config.get('exploration', {}).get('initial_exploration_rate', 0)
    results['exploration_low'] = exploration <= 0.05
    print(f"   7.3 exploration_rate: {exploration} — {'✅' if exploration <= 0.05 else '⚠️ высокий'}")

    all_ok = all(results.values())
    print(f"\n   Итог: {'✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_ok else '⚠️ ЕСТЬ РЕКОМЕНДАЦИИ'}")

    return results


def test_stage_8_profit_strategy():
    """Этап 8: Стратегия выхода и фиксация прибыли"""
    print("\n" + "=" * 70)
    print("ЭТАП 8: СТРАТЕГИЯ ВЫХОДА И ПРИБЫЛЬ")
    print("=" * 70)

    results = {}

    # 8.1 stop_loss и take_profit в settings
    with open('config/settings.json', 'r') as f:
        settings = json.load(f)

    sl = settings.get('stop_loss_percent', 0)
    tp = settings.get('take_profit_percent', 0)
    results['sl_tp'] = sl >= 3 and tp >= 6
    print(f"   8.1 stop_loss={sl}%, take_profit={tp}% — {'✅' if sl >= 3 and tp >= 6 else '❌'}")

    # 8.2 hold_reward
    with open('config/rl_config.json', 'r') as f:
        rl_config = json.load(f)

    hold = rl_config.get('hold_reward', {})
    hold_enabled = hold.get('enabled', False)
    hold_bonus = hold.get('max_bonus', 0)
    results['hold_reward'] = hold_enabled and hold_bonus >= 0.3
    print(
        f"   8.2 hold_reward: enabled={hold_enabled}, max_bonus={hold_bonus} — {'✅' if hold_enabled and hold_bonus >= 0.3 else '❌'}")

    # 8.3 reward_clip
    clip_min = rl_config.get('reward_clip_min', 0)
    clip_max = rl_config.get('reward_clip_max', 0)
    results['reward_clip'] = clip_min <= -3 and clip_max >= 3
    print(f"   8.3 reward_clip: [{clip_min}, {clip_max}] — {'✅' if clip_min <= -3 and clip_max >= 3 else '❌'}")

    # 8.4 commission_penalty
    reward_config = rl_config.get('reward_config', {})
    comm_penalty = reward_config.get('commission_penalty_scale', 0)
    results['commission_penalty'] = comm_penalty >= 100
    print(f"   8.4 commission_penalty_scale: {comm_penalty} — {'✅' if comm_penalty >= 100 else '❌'}")

    # 8.5 signal_filter отключён
    sig_filter = rl_config.get('signal_filter', {}).get('enabled', True)
    results['signal_filter_off'] = not sig_filter
    print(f"   8.5 signal_filter.enabled: {sig_filter} — {'✅' if not sig_filter else '⚠️ включён'}")

    all_ok = all(results.values())
    print(f"\n   Итог: {'✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_ok else '⚠️ ЕСТЬ РЕКОМЕНДАЦИИ'}")

    return results


def main():
    """Запуск всех тестов"""
    print("\n" + "=" * 70)
    print("🔬 ПОЛНАЯ ПРОВЕРКА ВСЕХ 8 ЭТАПОВ")
    print("=" * 70)
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    all_results['stage1'] = test_stage_1_model_dimension()
    all_results['stage2'] = test_stage_2_risk_manager()
    all_results['stage3'] = test_stage_3_news_signals()
    all_results['stage4'] = test_stage_4_market_features()
    all_results['stage5'] = test_stage_5_hysteresis_horizons()
    all_results['stage6'] = test_stage_6_api_moex()
    all_results['stage7'] = test_stage_7_action_mapping()
    all_results['stage8'] = test_stage_8_profit_strategy()

    # ИТОГ
    print("\n" + "=" * 70)
    print("📋 ИТОГОВАЯ СВОДКА")
    print("=" * 70)

    stages_passed = 0
    stages_total = len(all_results)

    for stage_name, results in all_results.items():
        if isinstance(results, dict):
            if stage_name == 'stage1':
                ok = results.get('all_ok', False)
            else:
                ok = all(results.values())

            icon = '✅' if ok else '⚠️'
            print(f"   {icon} {stage_name}")
            if ok:
                stages_passed += 1

    print(f"\n   Этапов пройдено: {stages_passed}/{stages_total}")

    if stages_passed == stages_total:
        print(f"\n   ✅ ВСЕ ЭТАПЫ ПРОЙДЕНЫ! СИСТЕМА ГОТОВА.")
    elif stages_passed >= 6:
        print(f"\n   ⚠️ Большинство этапов пройдено. Проверьте непройденные.")
    else:
        print(f"\n   ❌ Много проблем. Проверьте правки.")

    # Детали размерности
    stage1 = all_results.get('stage1', {})
    if isinstance(stage1, dict):
        print(f"\n📐 РАЗМЕРНОСТЬ МОДЕЛИ:")
        print(f"   state_vector_size: {stage1.get('state_vector_size', '?')}")
        print(f"   total_state_size: {stage1.get('total_state_size', '?')}")
        print(f"   action_dim: {stage1.get('action_dim', '?')}")
        print(f"   reserved_slots: {stage1.get('reserved_slots', '?')}")
        print(f"   Гистерезис+горизонты: {'✅ Встроены' if stage1.get('hysteresis_ok') else '❌ Не встроены'}")

    print("\n" + "=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЁН")
    print("=" * 70)


if __name__ == "__main__":
    main()