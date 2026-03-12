# quick_check_advanced.py
from models.trader_model import trader_model_instance
from collections import Counter
import numpy as np

m = trader_model_instance
memory_list = list(m.memory)

# Общая статистика
print("="*50)
print("📊 ОБЩАЯ СТАТИСТИКА")
print("="*50)
print(f"Память: {len(m.memory)} опытов")
print(f"Размерность: {m.policy_net.state_dim}")
print(f"Стратегий: {len(m.strategies)}")
print(f"Тикеров в статистике: {len(m.ticker_stats)}")

# Анализ reward по периодам
print("\n" + "="*50)
print("💰 АНАЛИЗ REWARD")
print("="*50)

periods = {
    "первые 100": memory_list[:100] if len(memory_list) >= 100 else memory_list,
    "последние 100": memory_list[-100:] if len(memory_list) >= 100 else memory_list,
    "все": memory_list
}

for period_name, period_data in periods.items():
    if period_data:
        rewards = [e['reward'] for e in period_data]
        print(f"\n{period_name.upper()}:")
        print(f"  средний: {sum(rewards)/len(rewards):.3f}")
        print(f"  медиана: {sorted(rewards)[len(rewards)//2]:.3f}")
        print(f"  мин/макс: {min(rewards):.3f}/{max(rewards):.3f}")

# Анализ действий
print("\n" + "="*50)
print("🔄 АНАЛИЗ ДЕЙСТВИЙ")
print("="*50)

actions = [e['action'] for e in memory_list]
action_counts = Counter(actions)
total = len(actions)

action_names = {0: "BUY", 1: "HOLD", 2: "SELL"}
for action_num, count in sorted(action_counts.items()):
    name = action_names.get(action_num, f"UNKNOWN({action_num})")
    print(f"{name}: {count} ({count/total*100:.1f}%)")

# Анализ по стратегиям
print("\n" + "="*50)
print("📈 ЭФФЕКТИВНОСТЬ СТРАТЕГИЙ")
print("="*50)

for strategy, perf in m.strategy_performance.items():
    if perf['total_trades'] > 0:
        print(f"\n{strategy}:")
        print(f"  сделок: {perf['total_trades']}")
        print(f"  win rate: {perf['win_rate']*100:.1f}%")
        print(f"  avg pnl: {perf['avg_pnl']:.3f}")

# Топ тикеров
print("\n" + "="*50)
print("🏆 ТОП-5 ТИКЕРОВ ПО PNL")
print("="*50)

tickers_with_pnl = []
for ticker, stats in m.ticker_stats.items():
    if stats['total_trades'] > 0:
        tickers_with_pnl.append((ticker, stats['total_pnl']))

for ticker, pnl in sorted(tickers_with_pnl, key=lambda x: x[1], reverse=True)[:5]:
    stats = m.ticker_stats[ticker]
    print(f"{ticker}: {pnl:.3f} (win rate: {stats['success_rate']*100:.1f}%, сделок: {stats['total_trades']})")

# Последние 10 опытов
print("\n" + "="*50)
print("🆕 ПОСЛЕДНИЕ 10 ОПЫТОВ")
print("="*50)

for i, exp in enumerate(memory_list[-10:]):
    action_name = action_names.get(exp['action'], f"UNK({exp['action']})")
    print(f"{i+1:2d}. {action_name:4} | reward: {exp['reward']:6.3f} | pnl_rub: {exp.get('pnl_rub', 0):6.2f}₽")

print("\n" + "="*50)
print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
print("="*50)