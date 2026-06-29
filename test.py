#!/usr/bin/env python3
"""Тест: анализ реальных сделок — время удержания и комиссия"""
import json
from datetime import datetime

# Вставь свои сделки из лога
trades = [
    {"time": "2026-06-25 10:51", "ticker": "PIKK", "action": "BUY", "qty": 5, "price": 568.30},
    {"time": "2026-06-25 11:13", "ticker": "PIKK", "action": "SELL", "qty": 5, "price": 569.00},
    {"time": "2026-06-25 10:51", "ticker": "VKCO", "action": "BUY", "qty": 18, "price": 197.85},
    {"time": "2026-06-25 11:00", "ticker": "OZON", "action": "BUY", "qty": 1, "price": 3487.50},
    {"time": "2026-06-25 12:05", "ticker": "HEAD", "action": "BUY", "qty": 1, "price": 2525.00},
    # Добавь SELL-сделки, если были
]

# Анализ
positions = {}
hold_times = []
commission_losses = []

for t in trades:
    ticker = t['ticker']
    if t['action'] == 'BUY':
        positions[ticker] = t
    elif t['action'] == 'SELL' and ticker in positions:
        buy = positions[ticker]
        # Время удержания
        buy_time = datetime.strptime(buy['time'], "%Y-%m-%d %H:%M")
        sell_time = datetime.strptime(t['time'], "%Y-%m-%d %H:%M")
        hold_minutes = (sell_time - buy_time).total_seconds() / 60
        hold_times.append(hold_minutes)

        # Комиссия vs прибыль
        buy_cost = buy['qty'] * buy['price']
        sell_revenue = t['qty'] * t['price']
        commission = (buy_cost + sell_revenue) * 0.003
        pnl = (t['price'] - buy['price']) * t['qty']
        net = pnl - commission
        commission_losses.append({
            'ticker': ticker,
            'hold_min': hold_minutes,
            'pnl': pnl,
            'commission': commission,
            'net': net
        })
        del positions[ticker]

print("=" * 60)
print("АНАЛИЗ РЕАЛЬНЫХ СДЕЛОК")
print("=" * 60)

if hold_times:
    print(f"\nСреднее время удержания: {sum(hold_times)/len(hold_times):.1f} мин")
    print(f"Минимальное: {min(hold_times):.1f} мин")
    print(f"Максимальное: {max(hold_times):.1f} мин")

if commission_losses:
    print(f"\nСделки, где комиссия > прибыли:")
    for cl in commission_losses:
        if cl['commission'] > cl['pnl']:
            print(f"  {cl['ticker']}: PnL={cl['pnl']:+.2f}₽, комиссия={cl['commission']:.2f}₽, чистый={cl['net']:+.2f}₽, держали {cl['hold_min']:.0f} мин")

print(f"\nОткрытых позиций: {len(positions)}")
for tkr in positions:
    print(f"  {tkr}: куплен в {positions[tkr]['time']}, ещё не продан")