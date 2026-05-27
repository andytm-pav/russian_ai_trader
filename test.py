#!/usr/bin/env python3
"""
ТЕСТ: проверка реальных значений turnover и market_cap
для подбора liquidity_scale_factor
"""

import sys
sys.path.insert(0, '.')

from fetchers.moex_fetcher import MoexFetcher

print("\n" + "=" * 60)
print("🔍 ТЕСТ: TURNOVER И MARKET CAP")
print("=" * 60)

moex = MoexFetcher()

# Получаем сырые данные
turnover = moex.get_shares_turnover()
market_cap = moex.get_market_capitalization()

print(f"\n📊 СЫРЫЕ ДАННЫЕ:")
print(f"   turnover (оборот): {turnover:,.2f}")
print(f"   market_cap (капитализация): {market_cap:,.2f}")

if market_cap > 0:
    raw_ratio = turnover / market_cap
    print(f"\n📐 РАСЧЁТ:")
    print(f"   turnover / market_cap = {raw_ratio:.10f}")

    # Подбираем scale_factor для разных целевых значений
    print(f"\n🎯 ПОДБОР liquidity_scale_factor:")
    for target in [0.1, 0.3, 0.5, 0.7, 0.9]:
        scale = target / raw_ratio
        print(f"   Для liquidity={target:.1f}: scale_factor = {scale:,.0f}")

    # Текущий расчёт
    current_scale = moex.settings.get('market_data', {}).get('liquidity_scale_factor', 1000.0)
    current_ratio = raw_ratio * current_scale
    print(f"\n📈 ТЕКУЩИЙ РЕЗУЛЬТАТ:")
    print(f"   scale_factor из конфига: {current_scale:,.1f}")
    print(f"   liquidity = {current_ratio:.6f}")

print("\n" + "=" * 60)
print("✅ ТЕСТ ЗАВЕРШЁН")
print("=" * 60)