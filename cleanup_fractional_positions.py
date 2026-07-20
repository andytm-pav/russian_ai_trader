#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Очистка "мусорных" позиций (qty не кратно lot_size).

Логика:
  1. Загружает portfolio_state.json
  2. Для каждой позиции проверяет кратность лотности
  3. Если qty < lot_size → позиция "мусорная", нужно продать ВСЁ
     (система paper-trading позволяет продать некратное количество)
  4. Если qty > lot_size, но не кратно → 2 варианта:
     - ДОКУПИТЬ до ближайшего кратного (если кэш позволяет и P&L положительный)
     - ПРОДАТЬ всё (если P&L отрицательный или нет кэша)
  5. Создаёт backup перед изменениями
  6. Записывает очистленный portfolio_state.json

Запуск:
  python cleanup_fractional_positions.py          # сухой прогон
  python cleanup_fractional_positions.py --apply   # применить
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# КОНФИГ
# ============================================================
PROJECT_ROOT = Path('')
PORTFOLIO_FILE = PROJECT_ROOT / 'data' / 'portfolio_state.json'
TICKERS_FILE = PROJECT_ROOT / 'config' / 'tickers.json'
BACKUP_DIR = PROJECT_ROOT / 'data' / 'backups' / 'cleanup'

# Комиссия Т-Банка
COMMISSION_RATE = 0.003
MIN_COMMISSION = 0.01


def load_ticker_lots() -> dict:
    """Загружает справочник лотностей."""
    with open(TICKERS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lots = {}
    for item in data.get('watchlist', []):
        lots[item['ticker']] = item.get('lot_size', 1)
    return lots


def analyze_positions(portfolio: dict, ticker_lots: dict) -> list:
    """Анализирует все позиции на кратность лотам."""
    positions = portfolio.get('positions', {})
    results = []

    for ticker, pos in positions.items():
        qty = pos.get('qty', 0)
        lot = ticker_lots.get(ticker, 1)
        avg_price = pos.get('avg_price', 0)

        if lot <= 1:
            # Лот = 1 — любая qty корректна
            results.append({
                'ticker': ticker, 'qty': qty, 'lot': lot, 'avg_price': avg_price,
                'status': 'OK', 'action': 'none', 'reason': 'lot=1'
            })
            continue

        if qty % lot == 0:
            results.append({
                'ticker': ticker, 'qty': qty, 'lot': lot, 'avg_price': avg_price,
                'status': 'OK', 'action': 'none', 'reason': 'кратно лоту'
            })
            continue

        # Проблема: qty не кратно lot
        if qty < lot:
            # Меньше 1 лота — мусорная позиция
            results.append({
                'ticker': ticker, 'qty': qty, 'lot': lot, 'avg_price': avg_price,
                'status': 'GARBAGE', 'action': 'sell_all',
                'reason': f'qty {qty} < lot {lot} — мусорная позиция'
            })
        else:
            # Больше 1 лота, но не кратно — нужно выровнять
            remainder = qty % lot
            full_lots = qty // lot
            results.append({
                'ticker': ticker, 'qty': qty, 'lot': lot, 'avg_price': avg_price,
                'status': 'FRACTIONAL', 'action': 'sell_to_multiple',
                'sell_qty': qty - (full_lots * lot),  # продаём остаток
                'keep_qty': full_lots * lot,
                'reason': f'{full_lots}×lot + {remainder} — продать {remainder}'
            })

    return results


def apply_cleanup(portfolio: dict, analysis: list, dry_run: bool = True) -> dict:
    """Применяет очистку к портфелю."""
    positions = portfolio.get('positions', {})
    cash = portfolio.get('cash', 0)
    trade_history = portfolio.get('trade_history', [])

    changes = []
    total_pnl = 0
    total_commission = 0

    for item in analysis:
        if item['action'] == 'none':
            continue

        ticker = item['ticker']
        qty_to_sell = item.get('sell_qty', item['qty'])

        if ticker not in positions:
            continue

        pos = positions[ticker]
        avg_price = pos['avg_price']
        # Текущая цена — берём из последней сделки или используем avg
        # В реальности нужно получить из MOEX, но для cleanup используем avg + небольшой markup
        # (в paper-trading нет реальных цен, используем сохранённые)
        current_price = pos.get('last_price', avg_price)

        revenue = qty_to_sell * current_price
        commission = max(revenue * COMMISSION_RATE, MIN_COMMISSION)
        pnl = (current_price - avg_price) * qty_to_sell - commission

        changes.append({
            'ticker': ticker,
            'action': 'SELL',
            'qty_sold': qty_to_sell,
            'price': current_price,
            'revenue': revenue,
            'commission': commission,
            'pnl': pnl,
            'remaining_qty': pos['qty'] - qty_to_sell,
        })

        total_pnl += pnl
        total_commission += commission

        if not dry_run:
            # Обновляем позицию
            if item['action'] == 'sell_all' or pos['qty'] - qty_to_sell <= 0:
                # Полностью закрываем
                del positions[ticker]
            else:
                # Частичная продажа
                pos['qty'] -= qty_to_sell

            # Обновляем кэш
            cash += revenue - commission

            # Записываем в историю
            trade_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'SELL',
                'ticker': ticker,
                'quantity': qty_to_sell,
                'price': current_price,
                'revenue': revenue,
                'commission': commission,
                'pnl': pnl,
                'strategy': pos.get('strategy', 'cleanup'),
                'reason': f'cleanup_fractional lot={item["lot"]}'
            })

    portfolio['positions'] = positions
    portfolio['cash'] = cash
    portfolio['trade_history'] = trade_history

    return {
        'changes': changes,
        'total_pnl': total_pnl,
        'total_commission': total_commission,
        'portfolio': portfolio,
    }


def main():
    dry_run = '--apply' not in sys.argv

    print("=" * 70)
    print("ОЧИСТКА МУСОРНЫХ ПОЗИЦИЙ (дробные qty)")
    print("=" * 70)
    print(f"Режим: {'DRY RUN (проверка)' if dry_run else 'APPLY (применение)'}")
    print()

    # Загружаем данные
    with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)

    ticker_lots = load_ticker_lots()

    print(f"Текущий кэш: {portfolio.get('cash', 0):.2f}₽")
    print(f"Позиций в портфеле: {len(portfolio.get('positions', {}))}")
    print()

    # Анализ
    analysis = analyze_positions(portfolio, ticker_lots)

    print("АНАЛИЗ ПОЗИЦИЙ:")
    print(f"{'Тикер':7} | {'qty':>5} | {'lot':>4} | {'статус':10} | {'действие':30}")
    print("-" * 75)
    for item in analysis:
        action_desc = item['reason']
        if item['action'] == 'sell_to_multiple':
            action_desc = f"продать {item['sell_qty']} (останется {item['keep_qty']})"
        elif item['action'] == 'sell_all':
            action_desc = f"ПРОДАТЬ ВСЁ ({item['qty']} шт)"
        print(f"{item['ticker']:7} | {item['qty']:>5} | {item['lot']:>4} | "
              f"{item['status']:10} | {action_desc}")

    # Применяем
    result = apply_cleanup(portfolio, analysis, dry_run=dry_run)

    print()
    print("ИЗМЕНЕНИЯ:")
    if not result['changes']:
        print("  Нет позиций для очистки — все позиции корректны! ✅")
    else:
        for c in result['changes']:
            print(f"  {c['ticker']:7} SELL {c['qty_sold']:>5} @ {c['price']:.2f} "
                  f"= {c['revenue']:.0f}₽ (comm {c['commission']:.2f}₽, P&L {c['pnl']:+.2f}₽)")

        print()
        print(f"Итого P&L от очистки: {result['total_pnl']:+.2f}₽")
        print(f"Итого комиссий: {result['total_commission']:.2f}₽")
        print(f"Чистыми: {result['total_pnl']:+.2f}₽")

    if dry_run:
        print()
        print("=" * 70)
        print("Это был DRY RUN. Для применения запустите:")
        print("  python cleanup_fractional_positions.py --apply")
        print("=" * 70)
    else:
        # Backup
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_file = BACKUP_DIR / f"portfolio_before_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
            original = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(original)
        print(f"\n✅ Backup сохранён: {backup_file}")

        # Сохраняем очищенный портфель
        with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
            json.dump(result['portfolio'], f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Портфель обновлён: {PORTFOLIO_FILE}")
        print(f"   Кэш после очистки: {result['portfolio']['cash']:.2f}₽")
        print(f"   Позиций осталось: {len(result['portfolio']['positions'])}")


if __name__ == '__main__':
    main()
