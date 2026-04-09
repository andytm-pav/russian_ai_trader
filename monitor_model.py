#!/usr/bin/env python3
"""
Визуализация обучения модели TraderModel
Показывает:
- Распределение стратегий и их эффективность
- Статистику по тикерам
- Распределение наград в памяти
- Графики обучения и сентимента
- Тепловые карты корреляций
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

# Попытка импорта библиотек визуализации
try:
    import matplotlib

    matplotlib.use('TkAgg')  # Для Windows
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib не установлен. Установите: pip install matplotlib")

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def print_separator(title=""):
    """Печать разделителя"""
    width = 80
    if title:
        padding = (width - len(title) - 2) // 2
        print("\n" + "=" * padding + f" {title} " + "=" * padding)
    else:
        print("\n" + "=" * width)


def analyze_model_knowledge():
    """Основная функция анализа"""
    print_separator("АНАЛИЗ ЗНАНИЙ МОДЕЛИ TRADERMODEL")
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Импорт модели
    try:
        from models.trader_model import trader_model_instance
        model = trader_model_instance
        print(f"✅ Модель загружена, device: {model.device}")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    # Сбор всей статистики
    knowledge = collect_model_knowledge(model)

    # Вывод в консоль
    print_knowledge_summary(knowledge)

    # Визуализация (если доступна)
    if MATPLOTLIB_AVAILABLE:
        visualize_knowledge(knowledge, model)
    else:
        print("\n⚠️ Визуализация недоступна (установите matplotlib)")

    # Сохранение отчета
    save_knowledge_report(knowledge)

    return knowledge


def collect_model_knowledge(model):
    """Сбор всех знаний модели"""
    knowledge = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {},
        'strategies': {},
        'tickers': {},
        'error_memory': {},
        'memory_analysis': {},
        'sentiment': {},
        'weights_analysis': {}
    }

    # ---------- Информация о модели ----------
    knowledge['model_info'] = {
        'device': str(model.device),
        'state_dim': model.total_state_dim,
        'base_dim': model.base_state_dim,
        'strategy_dim': model.strategy_params_dim,
        'news_dim': model.news_encoded_dim,
        'gamma': model.gamma,
        'exploration_rate': model.exploration_rate,
        'market_sentiment': model.market_sentiment,
        'volatility_index': model.volatility_index,
        'memory_size': len(model.memory),
        'prioritized_size': model.prioritized_buffer.size if hasattr(model, 'prioritized_buffer') else 0
    }

    # ---------- Стратегии ----------
    for strategy_name, perf in model.strategy_performance.items():
        params = model.strategies.get(strategy_name, {})
        knowledge['strategies'][strategy_name] = {
            'total_trades': perf['total_trades'],
            'profitable_trades': perf['profitable_trades'],
            'total_pnl': perf['total_pnl'],
            'avg_pnl': perf['avg_pnl'],
            'win_rate': perf['win_rate'],
            'params': {
                'risk_multiplier': params.get('risk_multiplier', 1.0),
                'target_hold_hours': params.get('target_hold_time_hours', 6),
                'stop_loss_percent': params.get('stop_loss_percent', 2.5),
                'take_profit_percent': params.get('take_profit_percent', 5.0),
                'news_weight': params.get('news_weight', 0.5),
                'tech_weight': params.get('tech_weight', 0.5)
            }
        }

    # ---------- Тикеры ----------
    for ticker, stats in model.ticker_stats.items():
        if stats['total_trades'] > 0:
            error_data = model.error_memory.get(ticker, {})
            knowledge['tickers'][ticker] = {
                'total_trades': stats['total_trades'],
                'profitable_trades': stats['profitable_trades'],
                'total_pnl': stats['total_pnl'],
                'avg_hold_time': stats['avg_hold_time'],
                'success_rate': stats['success_rate'],
                'failure_count': error_data.get('failure_count', 0),
                'avg_loss': error_data.get('avg_loss', 0.0),
                'last_trade': stats.get('last_trade')
            }

    # ---------- Анализ памяти ----------
    if len(model.memory) > 0:
        rewards = []
        actions = []
        pnls = []

        for exp in model.memory:
            if isinstance(exp, dict):
                rewards.append(exp.get('reward', 0))
                actions.append(exp.get('action', 1))
                pnls.append(exp.get('pnl_rub', 0))

        if rewards:
            knowledge['memory_analysis'] = {
                'total_experiences': len(model.memory),
                'rewards': {
                    'mean': float(np.mean(rewards)),
                    'std': float(np.std(rewards)),
                    'min': float(np.min(rewards)),
                    'max': float(np.max(rewards)),
                    'positive_ratio': float(np.mean(np.array(rewards) > 0))
                },
                'actions_distribution': {
                    'BUY': actions.count(0),
                    'HOLD': actions.count(1),
                    'SELL': actions.count(2)
                },
                'pnl': {
                    'mean': float(np.mean(pnls)),
                    'std': float(np.std(pnls)),
                    'total': float(np.sum(pnls))
                }
            }

    # ---------- Приоритетный буфер ----------
    if hasattr(model, 'prioritized_buffer') and model.prioritized_buffer.size > 0:
        priorities = model.prioritized_buffer.priorities[:model.prioritized_buffer.size]
        knowledge['memory_analysis']['prioritized'] = {
            'size': model.prioritized_buffer.size,
            'priorities_mean': float(np.mean(priorities)),
            'priorities_std': float(np.std(priorities)),
            'priorities_max': float(np.max(priorities)),
            'alpha': model.prioritized_buffer.alpha,
            'beta': model.prioritized_buffer.beta
        }

    # ---------- Сентимент ----------
    if hasattr(model, 'sentiment_history') and len(model.sentiment_history) > 0:
        sent_history = list(model.sentiment_history)
        knowledge['sentiment'] = {
            'history_length': len(sent_history),
            'current': model.market_sentiment,
            'mean': float(np.mean(sent_history)),
            'std': float(np.std(sent_history)),
            'min': float(np.min(sent_history)),
            'max': float(np.max(sent_history)),
            'recent_10': sent_history[-10:] if len(sent_history) >= 10 else sent_history
        }

    # ---------- Анализ весов ----------
    try:
        # Анализ первого слоя PolicyNet
        first_weight = None
        for name, param in model.policy_net.named_parameters():
            if 'weight' in name and param.dim() == 2:
                first_weight = param.detach().cpu().numpy()
                break

        if first_weight is not None:
            knowledge['weights_analysis'] = {
                'mean': float(np.mean(first_weight)),
                'std': float(np.std(first_weight)),
                'min': float(np.min(first_weight)),
                'max': float(np.max(first_weight)),
                'sparsity': float(np.mean(np.abs(first_weight) < 0.01)),
                'gradient_norm': None
            }

            # Градиенты (если есть)
            for name, param in model.policy_net.named_parameters():
                if param.grad is not None and 'weight' in name:
                    grad_norm = torch.norm(param.grad).item()
                    knowledge['weights_analysis']['gradient_norm'] = grad_norm
                    break
    except Exception as e:
        knowledge['weights_analysis']['error'] = str(e)

    return knowledge


def print_knowledge_summary(knowledge):
    """Вывод сводки в консоль"""
    print_separator("ИНФОРМАЦИЯ О МОДЕЛИ")
    info = knowledge['model_info']
    print(f"  Device: {info['device']}")
    print(f"  Размерность: {info['base_dim']} + {info['strategy_dim']} = {info['state_dim']}")
    print(f"  Память: {info['memory_size']} опытов (приоритетных: {info['prioritized_size']})")
    print(f"  Market sentiment: {info['market_sentiment']:.4f}")
    print(f"  Volatility index: {info['volatility_index']:.4f}")
    print(f"  Exploration rate: {info['exploration_rate']:.4f}")

    print_separator("СТРАТЕГИИ")
    if knowledge['strategies']:
        print(f"  {'Стратегия':<18} {'Сделок':<8} {'Win Rate':<10} {'Avg PnL':<10} {'Total PnL':<12} {'Risk':<6}")
        print("  " + "-" * 70)
        for name, data in sorted(knowledge['strategies'].items(), key=lambda x: x[1]['total_trades'], reverse=True):
            win_rate = data['win_rate']
            win_str = f"{win_rate * 100:.1f}%"
            avg_pnl = data['avg_pnl']
            avg_str = f"{avg_pnl * 100:.2f}%" if avg_pnl else "0.00%"
            total_pnl = data['total_pnl']
            total_str = f"{total_pnl * 100:.2f}%" if total_pnl else "0.00%"
            risk = data['params']['risk_multiplier']
            print(f"  {name:<18} {data['total_trades']:<8} {win_str:<10} {avg_str:<10} {total_str:<12} {risk:.2f}x")
    else:
        print("  Нет данных по стратегиям")

    print_separator("ТОП-10 ТИКЕРОВ ПО КОЛИЧЕСТВУ СДЕЛОК")
    if knowledge['tickers']:
        sorted_tickers = sorted(knowledge['tickers'].items(), key=lambda x: x[1]['total_trades'], reverse=True)[:10]
        print(f"  {'Тикер':<8} {'Сделок':<7} {'Успех':<8} {'PnL':<10} {'Удерж.(ч)':<10} {'Ошибок':<7}")
        print("  " + "-" * 60)
        for ticker, data in sorted_tickers:
            success = f"{data['success_rate'] * 100:.1f}%"
            pnl = f"{data['total_pnl'] * 100:.2f}%" if data['total_pnl'] else "0.00%"
            hold = f"{data['avg_hold_time']:.1f}"
            fails = data['failure_count']
            print(f"  {ticker:<8} {data['total_trades']:<7} {success:<8} {pnl:<10} {hold:<10} {fails:<7}")
    else:
        print("  Нет данных по тикерам")

    print_separator("ТОП-5 ЛУЧШИХ И ХУДШИХ ТИКЕРОВ")
    if knowledge['tickers']:
        # Лучшие по PnL
        profitable = [(t, d) for t, d in knowledge['tickers'].items() if d['total_trades'] >= 3]
        profitable.sort(key=lambda x: x[1]['total_pnl'], reverse=True)

        print("\n  ЛУЧШИЕ (по PnL):")
        for ticker, data in profitable[:5]:
            print(
                f"    {ticker}: {data['total_pnl'] * 100:.2f}% ({data['total_trades']} сделок, win rate: {data['success_rate'] * 100:.1f}%)")

        # Худшие по PnL
        profitable.sort(key=lambda x: x[1]['total_pnl'])
        print("\n  ХУДШИЕ (по PnL):")
        for ticker, data in profitable[:5]:
            print(
                f"    {ticker}: {data['total_pnl'] * 100:.2f}% ({data['total_trades']} сделок, win rate: {data['success_rate'] * 100:.1f}%)")

    print_separator("АНАЛИЗ ПАМЯТИ")
    mem = knowledge.get('memory_analysis', {})
    if mem:
        rewards = mem.get('rewards', {})
        print(f"  Всего опытов: {mem.get('total_experiences', 0)}")
        print(f"  Награды: mean={rewards.get('mean', 0):.4f}, std={rewards.get('std', 0):.4f}")
        print(f"           min={rewards.get('min', 0):.4f}, max={rewards.get('max', 0):.4f}")
        print(f"  Доля положительных: {rewards.get('positive_ratio', 0) * 100:.1f}%")

        actions = mem.get('actions_distribution', {})
        total_actions = sum(actions.values()) or 1
        print(f"  Распределение действий:")
        print(f"    BUY:  {actions.get('BUY', 0)} ({actions.get('BUY', 0) / total_actions * 100:.1f}%)")
        print(f"    HOLD: {actions.get('HOLD', 0)} ({actions.get('HOLD', 0) / total_actions * 100:.1f}%)")
        print(f"    SELL: {actions.get('SELL', 0)} ({actions.get('SELL', 0) / total_actions * 100:.1f}%)")

        if 'prioritized' in mem:
            p = mem['prioritized']
            print(f"  Приоритетный буфер: {p['size']} опытов")
            print(f"    Приоритеты: mean={p['priorities_mean']:.4f}, max={p['priorities_max']:.4f}")

    print_separator("СЕНТИМЕНТ")
    sent = knowledge.get('sentiment', {})
    if sent:
        print(f"  История: {sent.get('history_length', 0)} точек")
        print(f"  Текущий: {sent.get('current', 0):.4f}")
        print(f"  Средний: {sent.get('mean', 0):.4f} ± {sent.get('std', 0):.4f}")
        print(f"  Диапазон: [{sent.get('min', 0):.4f}, {sent.get('max', 0):.4f}]")


def visualize_knowledge(knowledge, model):
    """Визуализация знаний"""
    print_separator("ВИЗУАЛИЗАЦИЯ")

    # Настройка стиля
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")
        sns.set_palette("husl")

    # Создаем фигуру с подграфиками
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'Анализ знаний модели TraderModel\n{datetime.now().strftime("%Y-%m-%d %H:%M")}',
                 fontsize=16, fontweight='bold')

    plot_idx = 1

    # 1. Эффективность стратегий (bar chart)
    ax1 = fig.add_subplot(3, 4, plot_idx)
    if knowledge['strategies']:
        strategies = list(knowledge['strategies'].keys())
        win_rates = [knowledge['strategies'][s]['win_rate'] * 100 for s in strategies]
        total_pnls = [knowledge['strategies'][s]['total_pnl'] * 100 for s in strategies]

        x = np.arange(len(strategies))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, win_rates, width, label='Win Rate %', color='green', alpha=0.7)
        ax1.set_ylabel('Win Rate (%)', color='green')
        ax1.tick_params(axis='y', labelcolor='green')

        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x + width / 2, total_pnls, width, label='Total PnL %', color='blue', alpha=0.7)
        ax1_twin.set_ylabel('Total PnL (%)', color='blue')
        ax1_twin.tick_params(axis='y', labelcolor='blue')

        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.set_title('Эффективность стратегий')
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    else:
        ax1.text(0.5, 0.5, 'Нет данных по стратегиям', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Эффективность стратегий')
    plot_idx += 1

    # 2. Распределение Win Rate по тикерам
    ax2 = fig.add_subplot(3, 4, plot_idx)
    if knowledge['tickers']:
        win_rates = [d['success_rate'] * 100 for d in knowledge['tickers'].values() if d['total_trades'] >= 3]
        if win_rates:
            ax2.hist(win_rates, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50%')
            ax2.axvline(x=np.mean(win_rates), color='green', linestyle='--', alpha=0.7,
                        label=f'Mean: {np.mean(win_rates):.1f}%')
            ax2.set_xlabel('Win Rate (%)')
            ax2.set_ylabel('Количество тикеров')
            ax2.set_title(f'Распределение Win Rate (n={len(win_rates)})')
            ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Распределение Win Rate')
    plot_idx += 1

    # 3. Распределение наград в памяти
    ax3 = fig.add_subplot(3, 4, plot_idx)
    if 'memory_analysis' in knowledge and 'rewards' in knowledge['memory_analysis']:
        # Собираем награды из памяти
        rewards = []
        for exp in model.memory:
            if isinstance(exp, dict):
                rewards.append(exp.get('reward', 0))

        if rewards:
            ax3.hist(rewards, bins=50, color='coral', edgecolor='black', alpha=0.7)
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax3.axvline(x=np.mean(rewards), color='green', linestyle='--', alpha=0.7,
                        label=f'Mean: {np.mean(rewards):.3f}')
            ax3.set_xlabel('Reward')
            ax3.set_ylabel('Частота')
            ax3.set_title(f'Распределение наград (n={len(rewards)})')
            ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Распределение наград')
    plot_idx += 1

    # 4. История сентимента
    ax4 = fig.add_subplot(3, 4, plot_idx)
    if hasattr(model, 'sentiment_history') and len(model.sentiment_history) > 0:
        sent_history = list(model.sentiment_history)
        ax4.plot(sent_history, color='purple', alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=np.mean(sent_history), color='green', linestyle='--', alpha=0.5,
                    label=f'Mean: {np.mean(sent_history):.3f}')
        ax4.fill_between(range(len(sent_history)), 0, sent_history, alpha=0.3)
        ax4.set_xlabel('Шаг')
        ax4.set_ylabel('Сентимент')
        ax4.set_title(f'История рыночного сентимента (n={len(sent_history)})')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('История сентимента')
    plot_idx += 1

    # 5. Пирог действий
    ax5 = fig.add_subplot(3, 4, plot_idx)
    mem = knowledge.get('memory_analysis', {})
    actions = mem.get('actions_distribution', {})
    if actions and sum(actions.values()) > 0:
        labels = ['BUY', 'HOLD', 'SELL']
        sizes = [actions.get('BUY', 0), actions.get('HOLD', 0), actions.get('SELL', 0)]
        colors = ['green', 'gray', 'red']
        explode = (0.05, 0, 0.05)
        ax5.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax5.set_title('Распределение действий в памяти')
    else:
        ax5.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Распределение действий')
    plot_idx += 1

    # 6. Топ-10 тикеров по количеству сделок
    ax6 = fig.add_subplot(3, 4, plot_idx)
    if knowledge['tickers']:
        sorted_tickers = sorted(knowledge['tickers'].items(), key=lambda x: x[1]['total_trades'], reverse=True)[:10]
        tickers = [t[0] for t in sorted_tickers]
        trades = [t[1]['total_trades'] for t in sorted_tickers]
        success_rates = [t[1]['success_rate'] * 100 for t in sorted_tickers]

        x = np.arange(len(tickers))
        bars = ax6.bar(x, trades, color=['green' if s >= 50 else 'red' for s in success_rates], alpha=0.7)
        ax6.set_xticks(x)
        ax6.set_xticklabels(tickers, rotation=45, ha='right')
        ax6.set_ylabel('Количество сделок')
        ax6.set_title('Топ-10 тикеров по сделкам\n(цвет: зеленый - win rate ≥50%)')
    else:
        ax6.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Топ тикеров')
    plot_idx += 1

    # 7. Сравнение параметров стратегий
    ax7 = fig.add_subplot(3, 4, plot_idx)
    if knowledge['strategies']:
        strategies = list(knowledge['strategies'].keys())
        risk_mult = [knowledge['strategies'][s]['params']['risk_multiplier'] for s in strategies]
        stop_loss = [knowledge['strategies'][s]['params']['stop_loss_percent'] for s in strategies]
        take_profit = [knowledge['strategies'][s]['params']['take_profit_percent'] for s in strategies]

        x = np.arange(len(strategies))
        width = 0.25

        ax7.bar(x - width, risk_mult, width, label='Risk Multiplier', alpha=0.7)
        ax7.bar(x, stop_loss, width, label='Stop Loss %', alpha=0.7)
        ax7.bar(x + width, take_profit, width, label='Take Profit %', alpha=0.7)

        ax7.set_xticks(x)
        ax7.set_xticklabels(strategies, rotation=45, ha='right')
        ax7.set_ylabel('Значение')
        ax7.set_title('Параметры стратегий')
        ax7.legend(loc='upper right', fontsize='small')
    else:
        ax7.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Параметры стратегий')
    plot_idx += 1

    # 8. PnL по тикерам (scatter)
    ax8 = fig.add_subplot(3, 4, plot_idx)
    if knowledge['tickers']:
        tickers_data = [(t, d) for t, d in knowledge['tickers'].items() if d['total_trades'] >= 2]
        if tickers_data:
            trades = [d['total_trades'] for _, d in tickers_data]
            pnl = [d['total_pnl'] * 100 for _, d in tickers_data]
            colors = ['green' if p > 0 else 'red' for p in pnl]

            scatter = ax8.scatter(trades, pnl, c=colors, alpha=0.6, s=50)
            ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax8.set_xlabel('Количество сделок')
            ax8.set_ylabel('Total PnL (%)')
            ax8.set_title('PnL vs Количество сделок')

            # Добавляем подписи для лучших/худших
            sorted_by_pnl = sorted(tickers_data, key=lambda x: x[1]['total_pnl'])
            for t, d in sorted_by_pnl[:3] + sorted_by_pnl[-3:]:
                ax8.annotate(t, (d['total_trades'], d['total_pnl'] * 100), fontsize=8, alpha=0.8)
    else:
        ax8.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('PnL vs Сделки')
    plot_idx += 1

    # 9. Распределение приоритетов
    ax9 = fig.add_subplot(3, 4, plot_idx)
    if hasattr(model, 'prioritized_buffer') and model.prioritized_buffer.size > 0:
        priorities = model.prioritized_buffer.priorities[:model.prioritized_buffer.size]
        ax9.hist(priorities, bins=50, color='orange', edgecolor='black', alpha=0.7)
        ax9.axvline(x=np.mean(priorities), color='red', linestyle='--', alpha=0.7,
                    label=f'Mean: {np.mean(priorities):.4f}')
        ax9.set_xlabel('Priority')
        ax9.set_ylabel('Частота')
        ax9.set_title(f'Приоритеты в буфере (n={len(priorities)})')
        ax9.legend()
    else:
        ax9.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Приоритеты')
    plot_idx += 1

    # 10. Тепловая карта корреляции ошибок
    ax10 = fig.add_subplot(3, 4, plot_idx)
    if knowledge['tickers'] and len(knowledge['tickers']) >= 5:
        # Собираем данные для корреляции
        tickers_list = list(knowledge['tickers'].keys())[:15]
        data = []
        for t in tickers_list:
            d = knowledge['tickers'][t]
            data.append([
                d['total_trades'],
                d['success_rate'],
                d['total_pnl'],
                d['failure_count'],
                d['avg_hold_time']
            ])
        data = np.array(data)

        # Вычисляем корреляцию
        corr = np.corrcoef(data.T)

        im = ax10.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax10.set_xticks(range(5))
        ax10.set_yticks(range(5))
        ax10.set_xticklabels(['Trades', 'Success', 'PnL', 'Failures', 'Hold'], rotation=45)
        ax10.set_yticklabels(['Trades', 'Success', 'PnL', 'Failures', 'Hold'])
        ax10.set_title('Корреляция метрик')

        # Добавляем значения
        for i in range(5):
            for j in range(5):
                ax10.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=8)

        plt.colorbar(im, ax=ax10)
    else:
        ax10.text(0.5, 0.5, 'Недостаточно данных', ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Корреляция')
    plot_idx += 1

    # 11. Статистика по дням недели (если есть данные)
    ax11 = fig.add_subplot(3, 4, plot_idx)
    # Здесь можно добавить анализ по времени, если есть timestamp в опытах
    ax11.text(0.5, 0.5, 'Анализ по времени\n(требуются временные метки)',
              ha='center', va='center', transform=ax11.transAxes, fontsize=10)
    ax11.set_title('Временной анализ')
    plot_idx += 1

    # 12. Сводная информация
    ax12 = fig.add_subplot(3, 4, plot_idx)
    ax12.axis('off')

    info_text = f"""
    СВОДКА МОДЕЛИ
    ─────────────────────────────
    Устройство: {knowledge['model_info']['device']}
    Размерность: {knowledge['model_info']['base_dim']} + {knowledge['model_info']['strategy_dim']}
    Память: {knowledge['model_info']['memory_size']} опытов

    Стратегий: {len(knowledge['strategies'])}
    Тикеров: {len(knowledge['tickers'])}

    Market Sentiment: {knowledge['model_info']['market_sentiment']:.4f}
    Volatility: {knowledge['model_info']['volatility_index']:.4f}
    Exploration: {knowledge['model_info']['exploration_rate']:.4f}

    Средний Win Rate: {np.mean([s['win_rate'] for s in knowledge['strategies'].values()]) * 100:.1f}%
    Всего сделок: {sum(s['total_trades'] for s in knowledge['strategies'].values())}
    """

    if 'memory_analysis' in knowledge:
        mem = knowledge['memory_analysis']
        if 'pnl' in mem:
            info_text += f"\n    Total PnL (руб): {mem['pnl']['total']:.2f}"

    ax12.text(0.1, 0.9, info_text, transform=ax12.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax12.set_title('Сводка', fontweight='bold')

    # Настройка layout
    plt.tight_layout()

    # Сохранение
    output_path = Path('data/model_knowledge_visualization.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Визуализация сохранена: {output_path}")

    # Показ
    plt.show()


def save_knowledge_report(knowledge):
    """Сохранение отчета в JSON"""
    output_path = Path('data/model_knowledge_report.json')
    output_path.parent.mkdir(exist_ok=True)

    # Конвертируем numpy типы
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    knowledge_clean = convert_numpy(knowledge)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_clean, f, indent=2, ensure_ascii=False)

    print(f"✅ Отчет сохранен: {output_path}")


if __name__ == "__main__":
    analyze_model_knowledge()