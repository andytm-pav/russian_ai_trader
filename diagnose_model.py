#!/usr/bin/env python3
# diagnose_model_fixed.py
"""
Диагностика обученной модели трейдера в реальном времени
Запуск: python diagnose_model_fixed.py
"""

import sys
import os
import json
import pickle
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
import torch
from tabulate import tabulate

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ✅ ПРАВИЛЬНЫЕ ИМПОРТЫ согласно структуре проекта
from models.trader_model import (
    AdvancedTraderModel,
    trader_model_instance,
    BASE_STATE_DIM,
    STRATEGY_PARAMS_DIM,
    TOTAL_STATE_DIM,
    NEWS_ENCODED_DIM,
    NEWS_EMBEDDING_DIM
)

from models.smart_broker import SmartPortfolioBroker  # ← Вот правильный путь!


class ModelDiagnostic:
    """Диагностика модели и её памяти"""

    def __init__(self):
        self.model = None
        self.broker = None
        self.memory_analysis = {}
        self.strategy_stats = {}
        self.error_patterns = defaultdict(list)
        self.ticker_analysis = {}

    def load_model(self, use_instance=True):
        """Загрузка модели"""
        if use_instance and trader_model_instance is not None:
            try:
                self.model = trader_model_instance
                print("✅ Загружен экземпляр trader_model_instance")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки instance: {e}")
                self.model = AdvancedTraderModel()
                print("✅ Создан новый экземпляр модели")
        else:
            self.model = AdvancedTraderModel()

        if self.model:
            print(f"   Устройство: {self.model.device}")
            print(f"   Память: {len(self.model.memory)} опытов")
            if hasattr(self.model, 'prioritized_buffer'):
                print(f"   Приоритетный буфер: {self.model.prioritized_buffer.size}")

    def load_broker(self, settings_path="config/settings.json"):
        """Загрузка брокера для доступа к портфелю"""
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                self.broker = SmartPortfolioBroker(settings)

                # Проверяем наличие портфеля
                if hasattr(self.broker, 'portfolio'):
                    print(f"✅ Загружен брокер: {self.broker.portfolio.cash:.2f}₽ кэша, "
                          f"{len(self.broker.portfolio.positions)} позиций")
                else:
                    print("✅ Загружен брокер (без портфеля)")
            else:
                print(f"⚠️ Файл настроек не найден: {settings_path}")
                # Пробуем создать с пустыми настройками
                self.broker = SmartPortfolioBroker({})
                print("✅ Брокер создан с пустыми настройками")

        except Exception as e:
            print(f"⚠️ Не удалось загрузить брокера: {e}")
            import traceback
            traceback.print_exc()

    def analyze_memory(self):
        """🔍 Анализ буфера опыта"""
        if not self.model or not self.model.memory:
            print("❌ Память пуста")
            return

        memory = list(self.model.memory)

        # Базовые метрики
        self.memory_analysis['total'] = len(memory)

        # Анализ действий
        actions = [exp['action'] for exp in memory if 'action' in exp]
        if actions:
            action_counts = Counter(actions)
            self.memory_analysis['actions'] = {
                'BUY': action_counts.get(0, 0),
                'HOLD': action_counts.get(1, 0),
                'SELL': action_counts.get(2, 0)
            }

        # Анализ reward'ов
        rewards = [exp['reward'] for exp in memory if 'reward' in exp]
        if rewards:
            self.memory_analysis['rewards'] = {
                'mean': float(np.mean(rewards)),
                'median': float(np.median(rewards)),
                'std': float(np.std(rewards)),
                'min': float(min(rewards)),
                'max': float(max(rewards)),
                'percentiles': {
                    '1%': float(np.percentile(rewards, 1)),
                    '5%': float(np.percentile(rewards, 5)),
                    '95%': float(np.percentile(rewards, 95)),
                    '99%': float(np.percentile(rewards, 99))
                }
            }

        # Анализ PnL если есть
        pnl_values = [exp.get('pnl_rub', 0) for exp in memory if 'pnl_rub' in exp]
        if pnl_values and any(pnl_values):
            self.memory_analysis['pnl'] = {
                'mean': float(np.mean(pnl_values)),
                'median': float(np.median(pnl_values)),
                'total': float(sum(pnl_values)),
                'profitable_trades': int(sum(1 for p in pnl_values if p > 0)),
                'losing_trades': int(sum(1 for p in pnl_values if p < 0))
            }

        # Анализ сентимента
        sentiments = []
        for exp in memory:
            if exp.get('sentiment_data'):
                sent = exp['sentiment_data'].get('sentiment', 0)
                if sent != 0:
                    sentiments.append(sent)
        if sentiments:
            self.memory_analysis['sentiment'] = {
                'mean': float(np.mean(sentiments)),
                'std': float(np.std(sentiments)),
                'positive': int(sum(1 for s in sentiments if s > 0.1)),
                'negative': int(sum(1 for s in sentiments if s < -0.1))
            }

        # Размерности состояний
        state_dims = []
        for exp in memory[:100]:  # Проверяем первые 100
            if 'state' in exp and hasattr(exp['state'], 'shape'):
                try:
                    state_dims.append(exp['state'].shape[0])
                except:
                    pass
        if state_dims:
            dim_counts = Counter(state_dims)
            self.memory_analysis['state_dim'] = {
                'unique': list(set(state_dims)),
                'most_common': dim_counts.most_common(1)[0]
            }

        return self.memory_analysis

    def analyze_strategies(self):
        """📊 Анализ эффективности стратегий"""
        if not self.model or not hasattr(self.model, 'strategy_performance'):
            print("❌ Нет данных о стратегиях")
            return

        for strategy, perf in self.model.strategy_performance.items():
            if perf['total_trades'] > 0:
                self.strategy_stats[strategy] = {
                    'trades': perf['total_trades'],
                    'win_rate': perf['win_rate'],
                    'avg_pnl': perf['avg_pnl'],
                    'total_pnl': perf['total_pnl'],
                    'profitable': perf['profitable_trades']
                }

        return self.strategy_stats

    def analyze_tickers(self):
        """📈 Анализ статистики по тикерам"""
        if not self.model or not hasattr(self.model, 'ticker_stats'):
            print("❌ Нет статистики по тикерам")
            return

        for ticker, stats in self.model.ticker_stats.items():
            if stats['total_trades'] > 0:
                self.ticker_analysis[ticker] = {
                    'trades': stats['total_trades'],
                    'success_rate': stats['success_rate'],
                    'profitable': stats['profitable_trades'],
                    'total_pnl': stats['total_pnl'],
                    'avg_hold_time': stats['avg_hold_time']
                }

        return self.ticker_analysis

    def analyze_error_memory(self):
        """⚠️ Анализ памяти ошибок"""
        if not self.model or not hasattr(self.model, 'error_memory'):
            return {}

        error_summary = {}
        for ticker, data in self.model.error_memory.items():
            if data['failure_count'] > 0:
                error_summary[ticker] = {
                    'failures': data['failure_count'],
                    'avg_loss': data['avg_loss'],
                    'success_rate': data['success_rate'],
                    'total_trades': data['total_trades']
                }

        return error_summary

    def check_dimension_consistency(self):
        """📏 Проверка согласованности размерностей"""
        issues = []

        # Проверка констант
        if BASE_STATE_DIM + STRATEGY_PARAMS_DIM != TOTAL_STATE_DIM:
            issues.append(f"❌ Несоответствие констант: {BASE_STATE_DIM}+{STRATEGY_PARAMS_DIM}≠{TOTAL_STATE_DIM}")

        # Проверка сети
        if self.model and hasattr(self.model, 'policy_net'):
            expected = self.model.policy_net.state_dim
            if expected != TOTAL_STATE_DIM:
                issues.append(f"❌ Сеть ожидает {expected}, константа {TOTAL_STATE_DIM}")

        # Проверка памяти
        if self.memory_analysis.get('state_dim'):
            dims = self.memory_analysis['state_dim']['unique']
            if len(dims) > 1:
                issues.append(f"⚠️ Разные размерности в памяти: {dims}")
            elif dims and dims[0] != TOTAL_STATE_DIM:
                issues.append(f"⚠️ Память хранит {dims[0]}, сеть ожидает {TOTAL_STATE_DIM}")

        return issues

    def check_reward_scale(self):
        """💰 Проверка масштаба наград"""
        if not self.memory_analysis.get('rewards'):
            return "Нет данных"

        mean_reward = self.memory_analysis['rewards']['mean']
        std_reward = self.memory_analysis['rewards']['std']

        if abs(mean_reward) > 1.0:
            return f"⚠️ КРИТИЧЕСКИ: средний reward={mean_reward:.2f} (ожидается -0.5..0.5)"
        elif abs(mean_reward) > 0.5:
            return f"⚠️ Высокий reward: {mean_reward:.2f}"
        else:
            return f"✅ Нормальный: {mean_reward:.2f} ± {std_reward:.2f}"

    def find_best_worst_tickers(self, n=5):
        """🏆 Лучшие и худшие тикеры"""
        if not self.ticker_analysis:
            return {}, {}

        sorted_by_pnl = sorted(
            self.ticker_analysis.items(),
            key=lambda x: x[1]['total_pnl'],
            reverse=True
        )

        best = dict(sorted_by_pnl[:n])
        worst = dict(sorted_by_pnl[-n:]) if len(sorted_by_pnl) >= n else dict(sorted_by_pnl)

        return best, worst

    def generate_report(self):
        """📋 Генерация полного отчета"""

        print("\n" + "=" * 60)
        print("🧠 ДИАГНОСТИКА МОДЕЛИ ТРЕЙДЕРА")
        print("=" * 60)

        if not self.model:
            print("\n❌ Модель не загружена!")
            return

        # 1. Общая информация
        print(f"\n📊 ОБЩАЯ ИНФОРМАЦИЯ:")
        print(f"  • Устройство: {self.model.device}")
        print(f"  • Размерность состояния (константа): {TOTAL_STATE_DIM}")
        print(f"  • Базовое состояние: {BASE_STATE_DIM}")
        print(f"  • Параметры стратегии: {STRATEGY_PARAMS_DIM}")

        # 2. Проверка согласованности
        print(f"\n🔍 ПРОВЕРКА СОГЛАСОВАННОСТИ:")
        issues = self.check_dimension_consistency()
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  ✅ Все размерности согласованы")

        # 3. Анализ памяти
        print(f"\n📊 БУФЕР ОПЫТА:")
        if self.memory_analysis:
            mem = self.memory_analysis
            print(f"  • Всего записей: {mem.get('total', 0)}")
            if 'actions' in mem:
                acts = mem['actions']
                total = mem['total']
                print(f"  • Действия:")
                print(f"    - BUY:  {acts['BUY']} ({acts['BUY'] / total * 100:.1f}%)")
                print(f"    - HOLD: {acts['HOLD']} ({acts['HOLD'] / total * 100:.1f}%)")
                print(f"    - SELL: {acts['SELL']} ({acts['SELL'] / total * 100:.1f}%)")

            if 'rewards' in mem:
                r = mem['rewards']
                print(f"  • Награды:")
                print(f"    - Средняя: {r['mean']:.4f}")
                print(f"    - Медиана: {r['median']:.4f}")
                print(f"    - Std:     {r['std']:.4f}")
                print(f"    - Мин/Макс: {r['min']:.2f} / {r['max']:.2f}")
                print(f"    - Процентили: 1%={r['percentiles']['1%']:.2f}, 5%={r['percentiles']['5%']:.2f}, "
                      f"95%={r['percentiles']['95%']:.2f}, 99%={r['percentiles']['99%']:.2f}")

            reward_status = self.check_reward_scale()
            print(f"  • {reward_status}")

            if 'pnl' in mem:
                p = mem['pnl']
                print(f"  • PnL:")
                print(f"    - Всего: {p['total']:.2f}₽")
                print(f"    - Средний: {p['mean']:.2f}₽")
                print(f"    - Медианный: {p['median']:.2f}₽")
                profitable = p.get('profitable_trades', 0)
                losing = p.get('losing_trades', 0)
                print(f"    - Прибыльных/Убыточных: {profitable}/{losing}")

            if 'sentiment' in mem:
                s = mem['sentiment']
                print(f"  • Сентимент:")
                print(f"    - Средний: {s['mean']:.3f}")
                print(f"    - Положительных: {s['positive']}")
                print(f"    - Отрицательных: {s['negative']}")

            if 'state_dim' in mem:
                print(f"  • Размерности состояний: {mem['state_dim']['unique']}")
        else:
            print("  ❌ Нет данных о памяти")

        # 4. Эффективность стратегий
        print(f"\n📈 ЭФФЕКТИВНОСТЬ СТРАТЕГИЙ:")
        if self.strategy_stats:
            table = []
            for s, data in self.strategy_stats.items():
                table.append([
                    s,
                    data['trades'],
                    f"{data['win_rate'] * 100:.1f}%",
                    f"{data['avg_pnl']:.2f}",
                    f"{data['total_pnl']:.2f}",
                    data['profitable']
                ])
            print(tabulate(
                table,
                headers=['Стратегия', 'Сделок', 'Win Rate', 'Avg PnL', 'Total PnL', 'Прибыльных'],
                tablefmt='grid'
            ))
        else:
            print("  ❌ Нет данных о стратегиях")

        # 5. Топ тикеров
        print(f"\n🔝 ТОП-5 ЛУЧШИХ ТИКЕРОВ (по PnL):")
        best, worst = self.find_best_worst_tickers(5)
        if best:
            for ticker, data in best.items():
                print(f"  • {ticker}: {data['trades']} сделок, "
                      f"PnL={data['total_pnl']:.2f}₽, WR={data['success_rate'] * 100:.1f}%")
        else:
            print("  ❌ Нет данных")

        print(f"\n🔻 ТОП-5 ХУДШИХ ТИКЕРОВ (по PnL):")
        if worst:
            for ticker, data in worst.items():
                print(f"  • {ticker}: {data['trades']} сделок, "
                      f"PnL={data['total_pnl']:.2f}₽, WR={data['success_rate'] * 100:.1f}%")
        else:
            print("  ❌ Нет данных")

        # 6. Анализ ошибок
        print(f"\n⚠️ АНАЛИЗ ОШИБОК:")
        error_data = self.analyze_error_memory()
        if error_data:
            sorted_errors = sorted(
                error_data.items(),
                key=lambda x: x[1]['failures'],
                reverse=True
            )[:5]
            for ticker, data in sorted_errors:
                print(f"  • {ticker}: {data['failures']} ошибок, "
                      f"ср.убыток={data['avg_loss']:.2f}%, WR={data['success_rate'] * 100:.1f}%")
        else:
            print("  ✅ Ошибок не зафиксировано")

        # 7. Состояние портфеля (если есть брокер)
        if self.broker and hasattr(self.broker, 'portfolio'):
            print(f"\n💼 СОСТОЯНИЕ ПОРТФЕЛЯ:")
            print(f"  • Кэш: {self.broker.portfolio.cash:.2f}₽")
            if hasattr(self.broker.portfolio, 'reserved_cash'):
                print(f"  • Резерв: {self.broker.portfolio.reserved_cash:.2f}₽")
            print(f"  • Позиций: {len(self.broker.portfolio.positions)}")

            # Текущие позиции
            if self.broker.portfolio.positions:
                print(f"\n  ТЕКУЩИЕ ПОЗИЦИИ:")
                pos_table = []
                for ticker, pos in self.broker.portfolio.positions.items():
                    pos_table.append([
                        ticker,
                        pos['qty'],
                        f"{pos['avg_price']:.2f}",
                        pos.get('strategy', 'unknown'),
                        pos.get('lot_size', 1)
                    ])
                print(tabulate(
                    pos_table,
                    headers=['Тикер', 'Кол-во', 'Цена', 'Стратегия', 'Лот'],
                    tablefmt='simple'
                ))

            # Проверка pending комиссий
            if hasattr(self.broker.portfolio, 'pending_commissions'):
                pending = self.broker.portfolio.pending_commissions
                if pending:
                    total_pending = sum(c['amount'] for c in pending if not c.get('processed', False))
                    print(f"\n  💰 Ожидают списания комиссий: {total_pending:.2f}₽")

        # 8. Рекомендации
        print(f"\n✅ РЕКОМЕНДАЦИИ:")

        if self.memory_analysis.get('rewards'):
            mean_r = self.memory_analysis['rewards']['mean']
            if abs(mean_r) > 1.0:
                print(f"  • 🔴 КРИТИЧЕСКИ: Слишком большие reward'ы! Нужно уменьшить reward_scaling")
                if hasattr(self.model, 'reward_scaling'):
                    print(f"    Текущий scaling: {self.model.reward_scaling}")

        if self.memory_analysis.get('state_dim'):
            dims = self.memory_analysis['state_dim']['unique']
            if 150 in dims and 156 in dims:
                print("  • 🟡 Проблема: смесь состояний 150 и 156. Нужно унифицировать!")

        # Анализ стратегий
        for strategy, data in self.strategy_stats.items():
            if data['win_rate'] < 0.4 and data['trades'] > 10:
                print(f"  • 🟠 Стратегия '{strategy}' показывает низкую эффективность ({data['win_rate'] * 100:.1f}%)")

        print("\n" + "=" * 60)
        print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
        print("=" * 60)

    def save_report(self, filename="model_diagnostic_report.json"):
        """Сохранение отчета в JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'memory_analysis': self.memory_analysis,
            'strategy_stats': self.strategy_stats,
            'ticker_analysis': self.ticker_analysis,
            'error_analysis': self.analyze_error_memory(),
            'issues': self.check_dimension_consistency(),
            'reward_scale': self.check_reward_scale()
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Отчет сохранен в {filename}")
        except Exception as e:
            print(f"\n⚠️ Не удалось сохранить отчет: {e}")


def main():
    """Основная функция"""

    print("=" * 60)
    print("🔍 ЗАПУСК ДИАГНОСТИКИ МОДЕЛИ")
    print("=" * 60)

    # Создаем диагностику
    diag = ModelDiagnostic()

    # Загружаем модель
    diag.load_model(use_instance=True)

    # Загружаем брокера (опционально)
    diag.load_broker()

    # Анализируем
    diag.analyze_memory()
    diag.analyze_strategies()
    diag.analyze_tickers()

    # Генерируем отчет
    diag.generate_report()

    # Сохраняем
    diag.save_report()

    # Дополнительная информация о приоритетном буфере
    if diag.model and hasattr(diag.model, 'prioritized_buffer'):
        print(f"\n📊 ПРИОРИТЕТНЫЙ БУФЕР:")
        print(f"  • Размер: {diag.model.prioritized_buffer.size}")
        print(f"  • Alpha: {diag.model.prioritized_buffer.alpha}")
        print(f"  • Beta: {diag.model.prioritized_buffer.beta:.3f}")


if __name__ == "__main__":
    main()