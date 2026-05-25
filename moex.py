#!/usr/bin/env python3
"""
Проверка доступности данных MOEX для заполнения market_features
ИСПРАВЛЕННАЯ ВЕРСИЯ — учтены особенности API MOEX ISS
"""

import sys
import json
import time
from datetime import datetime

sys.path.insert(0, '.')

from fetchers.moex_fetcher import MoexFetcher
from utils.logger import get_logger

logger = get_logger("MOEX_TEST")


def test_brent_contracts(moex):
    """Ручная проверка всех фьючерсов на Brent"""
    print("\n📊 ТЕСТ 0: Поиск всех Brent-контрактов (BR-)")

    try:
        url = f"{moex.base_url}/engines/futures/markets/forts/securities.json"
        params = {
            'iss.meta': 'off',
            'iss.only': 'securities',
            'securities.columns': 'SECID,SHORTNAME,LASTTRADEDATE,LATNAME',
            'limit': 500
        }

        response = moex.session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if 'securities' not in data:
            print("   ❌ Нет данных securities")
            return []

        cols = data['securities']['columns']
        rows = data['securities']['data']

        brent_all = []
        for row in rows:
            secid = row[cols.index('SECID')] if 'SECID' in cols else ''
            shortname = row[cols.index('SHORTNAME')] if 'SHORTNAME' in cols else ''
            last_trade = row[cols.index('LASTTRADEDATE')] if 'LASTTRADEDATE' in cols else None
            latname = row[cols.index('LATNAME')] if 'LATNAME' in cols else ''

            # Расширенный фильтр: BR в любом регистре в начале тикера
            if secid.upper().startswith('BR') and last_trade:
                brent_all.append({
                    'secid': secid,
                    'shortname': shortname,
                    'last_trade': last_trade,
                    'latname': latname
                })

        if brent_all:
            print(f"   ✅ Найдено {len(brent_all)} контрактов BR-*:")
            for c in sorted(brent_all, key=lambda x: x['last_trade'])[:10]:
                print(f"     {c['secid']:15s} {c['shortname']:20s} last_trade={c['last_trade']} ({c['latname']})")
            return brent_all
        else:
            print(f"   ❌ Контракты BR-* не найдены. Всего инструментов: {len(rows)}")
            # Покажем примеры фьючерсов для диагностики
            print(f"   Примеры фьючерсов (первые 10):")
            for row in rows[:10]:
                secid = row[cols.index('SECID')] if 'SECID' in cols else '?'
                name = row[cols.index('SHORTNAME')] if 'SHORTNAME' in cols else '?'
                print(f"     {secid:15s} {name}")
            return []

    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return []


def test_usd_rub_all(moex):
    """Проверка всех доступных инструментов USD/RUB"""
    print("\n📊 ТЕСТ 0.2: Поиск всех инструментов USD/RUB")

    instruments = ['USD000000TOD', 'USD000UTSTOM', 'USD/RUB']
    results = {}

    for instr in instruments:
        try:
            # Пробуем валютный рынок
            url = f"{moex.base_url}/engines/currency/markets/selt/securities/{instr}.json"
            params = {
                'iss.meta': 'off',
                'iss.only': 'marketdata',
                'marketdata.columns': 'SECID,LAST,CHANGE,CHANGEPRC'
            }

            response = moex.session.get(url, params=params, timeout=10)
            if response.status_code != 200:
                results[instr] = f"HTTP {response.status_code}"
                continue

            data = response.json()
            if 'marketdata' in data and data['marketdata']['data']:
                row = data['marketdata']['data'][0]
                cols = data['marketdata']['columns']
                if 'LAST' in cols:
                    idx = cols.index('LAST')
                    if idx < len(row) and row[idx] is not None:
                        price = float(row[idx])
                        if price > 0:
                            results[instr] = f"✅ {price:.4f}"
                            continue
            results[instr] = "нет данных"
        except Exception as e:
            results[instr] = f"ошибка: {str(e)[:50]}"

    for instr, status in results.items():
        print(f"   {instr:20s}: {status}")

    return results


def test_orderbook_fix(moex, ticker='SBER'):
    """Проверка стакана с правильной обработкой"""
    print(f"\n📊 ТЕСТ 0.3: get_orderbook('{ticker}') — разные board")

    boards = ['TQBR', 'TQBR', 'SNDX']
    for board in boards:
        try:
            url = f"{moex.base_url}/engines/stock/markets/shares/boards/{board}/securities/{ticker}/orderbook.json"
            params = {'iss.meta': 'off'}

            response = moex.session.get(url, params=params, timeout=8)
            print(f"   Board {board}: HTTP {response.status_code}", end='')

            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'orderbook' in data and data['orderbook']['data']:
                        print(f" — ✅ данные получены ({len(data['orderbook']['data'])} строк)")
                    else:
                        print(f" — пустой ответ (требуется авторизация/подписка)")
                except:
                    print(f" — не JSON")
            else:
                print(f" — ошибка")
        except Exception as e:
            print(f"   Board {board}: ❌ {str(e)[:50]}")


def test_moex_data():
    """Тестирование всех источников данных MOEX"""

    print("\n" + "=" * 70)
    print("🔍 ТЕСТ ДАННЫХ MOEX ДЛЯ MARKET_FEATURES (v2)")
    print("=" * 70)
    print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    moex = MoexFetcher()
    results = {}

    # ===== ПРЕДВАРИТЕЛЬНЫЕ ТЕСТЫ =====
    brent_contracts = test_brent_contracts(moex)
    usd_results = test_usd_rub_all(moex)
    test_orderbook_fix(moex, 'SBER')

    # ===== ТЕСТ 1: get_all_securities =====
    print("\n📊 ТЕСТ 1: get_all_securities()")
    t1 = time.time()
    securities = moex.get_all_securities()
    t1 = time.time() - t1

    if securities:
        sample_ticker = list(securities.keys())[0] if securities else None
        sample_data = securities.get(sample_ticker, {}) if sample_ticker else {}

        print(f"   Статус: ✅ OK ({len(securities)} бумаг за {t1:.1f}с)")
        print(f"   Пример ({sample_ticker}):")
        for key in ['price', 'volume', 'spread', 'market_cap', 'momentum', 'liquidity']:
            print(f"     {key}: {sample_data.get(key, 'НЕТ')}")

        with_price = sum(1 for s in securities.values() if s.get('price', 0) > 0)
        with_volume = sum(1 for s in securities.values() if s.get('volume', 0) > 0)
        with_spread = sum(1 for s in securities.values() if s.get('spread', 0) > 0)
        with_market_cap = sum(1 for s in securities.values() if s.get('market_cap', 0) > 0)

        print(f"   Статистика:")
        print(f"     С ценой: {with_price}/{len(securities)}")
        print(f"     С объёмом: {with_volume}/{len(securities)}")
        print(f"     Со спредом: {with_spread}/{len(securities)}")
        print(f"     С капитализацией: {with_market_cap}/{len(securities)}")

        results['securities'] = len(securities)
        results['spread_available'] = with_spread > 0
    else:
        print(f"   Статус: ❌ FAIL")
        results['securities'] = 0

    # ===== ТЕСТ 2: get_price =====
    print("\n📊 ТЕСТ 2: get_price('SBER')")
    price = moex.get_price('SBER')
    if price and price > 0:
        print(f"   Статус: ✅ OK (SBER = {price:.2f}₽)")
        results['price'] = price
    else:
        print(f"   Статус: ⚠️ WARN")
        results['price'] = None

    # ===== ТЕСТ 3: get_market_indices =====
    print("\n📊 ТЕСТ 3: get_market_indices()")
    indices = moex.get_market_indices()

    if indices:
        imoex = indices.get('IMOEX', 0)
        rtsi = indices.get('RTSI', 0)
        rvi = indices.get('RVI', 0)
        market_mood = indices.get('market_mood', 0)
        moexog = indices.get('MOEXOG', 0)

        print(f"   Статус: ✅ OK")
        print(f"   IMOEX: {imoex:.2f} (→ imoex_norm: {imoex / 4000:.4f})")
        print(f"   RTSI: {rtsi:.2f} (→ rtsi_norm: {rtsi / 2000:.4f})")
        print(f"   RVI: {rvi:.2f} (→ rvi_norm: {rvi / 100:.4f})")
        print(f"   MOEXOG: {moexog:.2f} (→ moexog_norm: {moexog / 10000:.4f})")
        print(f"   Market Mood: {market_mood:.4f}")

        results['indices'] = True
        results['imoex'] = imoex
        results['rtsi'] = rtsi
        results['rvi'] = rvi
        results['moexog'] = moexog
        results['market_mood'] = market_mood
    else:
        print(f"   Статус: ❌ FAIL")
        results['indices'] = False

    # ===== ТЕСТ 4: get_macro_data =====
    print("\n📊 ТЕСТ 4: get_macro_data()")
    macro = moex.get_macro_data()

    if macro:
        print(f"   Статус: ✅ OK")
        for key in ['imoex', 'rtsi', 'rvi', 'brent', 'usd_rub',
                    'shares_turnover', 'market_cap',
                    'market_liquidity_ratio', 'market_activity_score']:
            print(f"   {key}: {macro.get(key, 'НЕТ')}")

        results['macro'] = True
        results['shares_turnover'] = macro.get('shares_turnover', 0)
        results['market_cap_total'] = macro.get('market_cap', 0)
        results['liquidity_ratio'] = macro.get('market_liquidity_ratio', 0)
        results['brent'] = macro.get('brent', 0)
        results['usd_rub'] = macro.get('usd_rub', 0)
    else:
        print(f"   Статус: ❌ FAIL")
        results['macro'] = False

    # ===== ТЕСТ 5: get_shares_turnover =====
    print("\n📊 ТЕСТ 5: get_shares_turnover()")
    turnover = moex.get_shares_turnover()
    if turnover > 0:
        print(f"   Статус: ✅ OK (Оборот = {turnover:,.0f}₽ → norm: {turnover / 1e12:.4f})")
        results['turnover'] = turnover
    else:
        print(f"   Статус: ⚠️ WARN")
        results['turnover'] = 0

    # ===== ТЕСТ 6: get_market_capitalization =====
    print("\n📊 ТЕСТ 6: get_market_capitalization()")
    market_cap = moex.get_market_capitalization()
    if market_cap > 0:
        print(f"   Статус: ✅ OK (Капитализация = {market_cap:,.0f}₽ → norm: {market_cap / 1e14:.4f})")
        results['market_cap'] = market_cap
    else:
        print(f"   Статус: ⚠️ WARN")
        results['market_cap'] = 0

    # ===== ТЕСТ 7: get_candles =====
    print("\n📊 ТЕСТ 7: get_candles('SBER', interval=60, count=5)")
    candles = moex.get_candles('SBER', interval=60, count=5)
    if candles is not None and not candles.empty:
        print(f"   Статус: ✅ OK ({len(candles)} свечей)")
        results['candles'] = len(candles)
    else:
        print(f"   Статус: ⚠️ WARN")
        results['candles'] = 0

    # ===== ИТОГОВАЯ СВОДКА =====
    print("\n" + "=" * 70)
    print("📋 ИТОГОВАЯ СВОДКА: MARKET_FEATURES (v2)")
    print("=" * 70)

    # Новый список признаков с заменами
    features_status = {
        '1. spread_pct': results.get('spread_available', False),
        '2. market_mood': results.get('indices', False),
        '3. shares_turnover': results.get('turnover', 0) > 0,
        '4. rvi_normalized': results.get('indices', False),
        '5. imoex_normalized': results.get('indices', False),
        '6. market_cap_total': results.get('market_cap', 0) > 0,
        '7. liquidity_ratio (замена order_flow)': results.get('liquidity_ratio', 0) > 0,
        '8. rtsi_normalized (замена futures_premium)': results.get('indices', False),
        '9. usd_rub': any('✅' in str(v) for v in usd_results.values()),
        '10. moexog_normalized (замена brent)': results.get('indices', False),
    }

    ready = 0
    for name, status in features_status.items():
        icon = '✅' if status else '❌'
        if status:
            ready += 1
        print(f"   {icon} {name}")

    print(f"\n   Готово признаков: {ready}/10")

    if ready >= 8:
        print(f"\n   ✅ ДАННЫХ ДОСТАТОЧНО для заполнения market_features!")
    elif ready >= 5:
        print(f"\n   ⚠️ ДАННЫХ ЧАСТИЧНО ДОСТАТОЧНО.")
    else:
        print(f"\n   ❌ ДАННЫХ НЕДОСТАТОЧНО. Запустите в торговую сессию.")

    # Дополнительная информация
    print(f"\n📝 ПРИМЕЧАНИЯ:")
    if not brent_contracts:
        print(f"   ⚠️ Brent-фьючерсы не найдены. Используем MOEXOG как замену.")
    if not any('✅' in str(v) for v in usd_results.values()):
        print(f"   ⚠️ USD/RUB не доступен. Используем данные из macro_data.")
    print(f"   ℹ️ order_flow_imbalance ТРЕБУЕТ ПЛАТНУЮ ПОДПИСКУ MOEX. Заменён на liquidity_ratio.")

    print("\n" + "=" * 70)
    print("✅ ТЕСТ ЗАВЕРШЁН")
    print("=" * 70)

    return results


if __name__ == "__main__":
    test_moex_data()