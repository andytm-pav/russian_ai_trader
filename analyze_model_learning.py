#!/usr/bin/env python3
"""
Скрипт анализа обучения модели - показывает, чему научилась модель
Запуск: python analyze_model_learning.py
"""

import json
import numpy as np
import torch
from collections import Counter, defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Импортируем модель
from models.trader_model import trader_model_instance


class ModelLearningAnalyzer:
    def __init__(self, model):
        self.model = model
        self.stats = {}

    def analyze_memory(self):
        """Анализ содержимого памяти (700 опытов)"""
        memory = list(self.model.memory)

        if not memory:
            print("❌ Память пуста")
            return

        print("\n" + "=" * 60)
        print("📊 АНАЛИЗ ПАМЯТИ МОДЕЛИ")
        print("=" * 60)
        print(f"Всего опытов: {len(memory)}")

        # Распределение действий
        actions = [exp['action'] for exp in memory]
        action_counts = Counter(actions)
        action_names = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}

        print("\n🎯 Распределение действий:")
        for action, count in sorted(action_counts.items()):
            name = action_names.get(action, f'UNKNOWN({action})')
            print(f"  {name}: {count} ({count / len(memory) * 100:.1f}%)")

        # Анализ наград
        rewards = [exp['reward'] for exp in memory]
        print(f"\n💰 Анализ наград:")
        print(f"  Средняя награда: {np.mean(rewards):.4f}")
        print(f"  Медианная: {np.median(rewards):.4f}")
        print(f"  Стандартное отклонение: {np.std(rewards):.4f}")
        print(f"  Min: {min(rewards):.4f}")
        print(f"  Max: {max(rewards):.4f}")

        # Процент положительных/отрицательных наград
        pos_rewards = sum(1 for r in rewards if r > 0)
        neg_rewards = sum(1 for r in rewards if r < 0)
        print(f"  Положительных: {pos_rewards} ({pos_rewards / len(rewards) * 100:.1f}%)")
        print(f"  Отрицательных: {neg_rewards} ({neg_rewards / len(rewards) * 100:.1f}%)")

        # Динамика наград (последние 100 vs первые 100)
        if len(rewards) >= 200:
            first_100 = np.mean(rewards[:100])
            last_100 = np.mean(rewards[-100:])
            print(f"\n📈 Динамика обучения:")
            print(f"  Первые 100 опытов (среднее): {first_100:.4f}")
            print(f"  Последние 100 опытов (среднее): {last_100:.4f}")
            print(f"  Изменение: {last_100 - first_100:+.4f} ({((last_100 / first_100) - 1) * 100:+.1f}%)")

        return {
            'total_experiences': len(memory),
            'action_distribution': action_counts,
            'reward_stats': {
                'mean': np.mean(rewards),
                'median': np.median(rewards),
                'std': np.std(rewards),
                'min': min(rewards),
                'max': max(rewards)
            }
        }

    def analyze_strategies(self):
        """Анализ эффективности стратегий"""
        print("\n" + "=" * 60)
        print("🎯 АНАЛИЗ ЭФФЕКТИВНОСТИ СТРАТЕГИЙ")
        print("=" * 60)

        strategies = self.model.strategy_performance

        if not strategies:
            print("❌ Нет данных по стратегиям")
            return

        # Сортируем по win_rate
        sorted_strategies = sorted(
            strategies.items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )

        print("\n📊 Рейтинг стратегий:")
        for i, (name, perf) in enumerate(sorted_strategies, 1):
            print(f"\n  {i}. {name.upper()}")
            print(f"     Сделок: {perf['total_trades']}")
            print(f"     Win Rate: {perf['win_rate'] * 100:.1f}%")
            print(f"     Средний PnL: {perf['avg_pnl']:.4f}")
            print(f"     Общий PnL: {perf['total_pnl']:.4f}")

            # Текущий risk_multiplier
            current_mult = self.model.strategies[name].get('risk_multiplier', 1.0)
            print(f"     Risk multiplier: {current_mult:.2f}")

        # Лучшая и худшая стратегия
        best = sorted_strategies[0]
        worst = sorted_strategies[-1]

        print(f"\n🏆 Лучшая стратегия: {best[0]} (WR: {best[1]['win_rate'] * 100:.1f}%)")
        print(f"💩 Худшая стратегия: {worst[0]} (WR: {worst[1]['win_rate'] * 100:.1f}%)")

        return dict(sorted_strategies)

    def analyze_tickers(self):
        """Анализ статистики по тикерам"""
        print("\n" + "=" * 60)
        print("📈 АНАЛИЗ ПО ТИКЕРАМ")
        print("=" * 60)

        tickers = self.model.ticker_stats

        if not tickers:
            print("❌ Нет данных по тикерам")
            return

        # Топ-10 тикеров по количеству сделок
        top_by_trades = sorted(
            tickers.items(),
            key=lambda x: x[1]['total_trades'],
            reverse=True
        )[:10]

        print("\n🔝 Топ-10 тикеров по количеству сделок:")
        for ticker, stats in top_by_trades:
            print(f"  {ticker}: {stats['total_trades']} сделок, "
                  f"WR: {stats['success_rate'] * 100:.1f}%, "
                  f"среднее время: {stats['avg_hold_time']:.1f}ч")

        # Топ по успешности (мин 5 сделок)
        profitable = [
            (t, s) for t, s in tickers.items()
            if s['total_trades'] >= 5
        ]
        top_by_winrate = sorted(
            profitable,
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )[:5]

        print("\n💰 Топ-5 самых прибыльных тикеров (≥5 сделок):")
        for ticker, stats in top_by_winrate:
            print(f"  {ticker}: WR {stats['success_rate'] * 100:.1f}%, "
                  f"{stats['total_trades']} сделок")

        # Худшие тикеры
        worst_by_winrate = sorted(
            profitable,
            key=lambda x: x[1]['success_rate']
        )[:5]

        print("\n💩 Топ-5 самых убыточных тикеров (≥5 сделок):")
        for ticker, stats in worst_by_winrate:
            print(f"  {ticker}: WR {stats['success_rate'] * 100:.1f}%, "
                  f"{stats['total_trades']} сделок")

    def analyze_errors(self):
        """Анализ ошибок (убыточных сделок)"""
        print("\n" + "=" * 60)
        print("⚠️ АНАЛИЗ ОШИБОК")
        print("=" * 60)

        errors = self.model.error_memory

        if not errors:
            print("✅ Нет записанных ошибок!")
            return

        total_errors = sum(e['failure_count'] for e in errors.values())
        print(f"Всего ошибок: {total_errors}")
        print(f"Тикеров с ошибками: {len(errors)}")

        # Топ проблемных тикеров
        problematic = sorted(
            errors.items(),
            key=lambda x: x[1]['failure_count'],
            reverse=True
        )[:5]

        print("\n🔥 Самые проблемные тикеры:")
        for ticker, data in problematic:
            print(f"  {ticker}: {data['failure_count']} ошибок, "
                  f"средний убыток {data['avg_loss'] * 100:.2f}%")

        # Анализ последних ошибок
        print("\n🕒 Последние ошибки:")
        recent_errors = []
        for ticker, data in errors.items():
            if data.get('last_failure'):
                recent_errors.append((ticker, data['last_failure']))

        recent_errors.sort(key=lambda x: x[1], reverse=True)
        for ticker, time_str in recent_errors[:5]:
            print(f"  {ticker}: {time_str}")

    def analyze_policy_network(self):
        """Анализ политики нейросети"""
        print("\n" + "=" * 60)
        print("🧠 АНАЛИЗ ПОЛИТИКИ НЕЙРОСЕТИ")
        print("=" * 60)

        # Создаем тестовые состояния для разных сценариев
        device = self.model.device

        # Тестовые сценарии
        scenarios = {
            "Позитивный сценарий": {
                'price': 100.0,
                'momentum': 0.05,
                'sentiment': 0.7,
                'market_sentiment': 0.5
            },
            "Негативный сценарий": {
                'price': 100.0,
                'momentum': -0.05,
                'sentiment': -0.7,
                'market_sentiment': -0.5
            },
            "Нейтральный сценарий": {
                'price': 100.0,
                'momentum': 0.0,
                'sentiment': 0.0,
                'market_sentiment': 0.0
            }
        }

        self.model.policy_net.eval()

        print("\n📊 Предсказания модели для разных сценариев:")

        for name, params in scenarios.items():
            # Создаем тестовое состояние
            test_state = torch.zeros(1, self.model.policy_net.state_dim).to(device)

            with torch.no_grad():
                # ✅ ИСПРАВЛЕНИЕ: policy_net возвращает 3 значения
                action_probs, state_value, _ = self.model.policy_net(test_state)
                probs = action_probs.cpu().numpy()[0]

            print(f"\n  {name}:")
            print(f"    BUY:  {probs[0] * 100:.1f}%")
            print(f"    HOLD: {probs[1] * 100:.1f}%")
            print(f"    SELL: {probs[2] * 100:.1f}%")
            print(f"    Value: {state_value.item():.4f}")

    def generate_report(self):
        """Генерация полного отчета"""
        print("\n" + "🚀" * 30)
        print("📋 ПОЛНЫЙ ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛИ")
        print("🚀" * 30)

        # Общая статистика
        print(f"\n📊 Общая статистика:")
        print(f"  Устройство: {self.model.device}")
        print(f"  Размерность состояния: {self.model.policy_net.state_dim}")
        print(f"  Рыночный сентимент: {self.model.market_sentiment:.3f}")
        print(f"  Индекс волатильности: {self.model.volatility_index:.3f}")

        # Запускаем все анализы
        memory_stats = self.analyze_memory()
        self.analyze_strategies()
        self.analyze_tickers()
        self.analyze_errors()
        self.analyze_policy_network()

        # Выводы
        print("\n" + "=" * 60)
        print("💡 ВЫВОДЫ")
        print("=" * 60)

        if memory_stats:
            pos_pct = memory_stats['reward_stats']['mean'] > 0
            if pos_pct:
                print("✅ Модель в среднем получает ПОЛОЖИТЕЛЬНЫЕ награды")
            else:
                print("⚠️ Модель в среднем получает ОТРИЦАТЕЛЬНЫЕ награды")

        # Сохраняем отчет в файл
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_stats': self.model.get_model_stats(),
            'memory_analysis': memory_stats,
            'strategies': dict(self.model.strategy_performance)
        }

        with open('data/model_learning_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        print("\n📁 Отчет сохранен в data/model_learning_report.json")

        # Визуализация (если есть matplotlib)
        try:
            self.plot_learning_curve()
        except:
            print("⚠️ Не удалось создать график (matplotlib не настроен)")

    def plot_learning_curve(self):
        """Построение графика обучения"""
        rewards = [exp['reward'] for exp in list(self.model.memory)]

        if len(rewards) < 10:
            return

        # Скользящее среднее
        window = min(50, len(rewards) // 10)
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(rewards, alpha=0.3, label='Raw rewards')
        plt.plot(range(window - 1, len(rewards)), moving_avg, 'r-', linewidth=2,
                 label=f'{window}-episode moving avg')
        plt.xlabel('Experience #')
        plt.ylabel('Reward')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Распределение наград
        plt.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('data/learning_curve.png', dpi=100)
        print("📊 График обучения сохранен в data/learning_curve.png")


def main():
    print("🔍 Анализ обучения модели...")
    print(f"📊 Опыта в модели: {len(trader_model_instance.memory)}")

    analyzer = ModelLearningAnalyzer(trader_model_instance)
    analyzer.generate_report()


if __name__ == "__main__":
    main()