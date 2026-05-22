#!/usr/bin/env python3
"""
ГЛУБОКАЯ ДИАГНОСТИКА ТОРГОВОЙ СИСТЕМЫ
Проверяет:
1. Реальные сделки через portfolio.buy/sell с проверкой кэша и лимитов
2. Параметры модели: веса, градиенты, распределение действий
3. Параметры обучения: exploration, decay, reward
4. Статистику стратегий и тикеров
5. Память модели: какие действия преобладают
"""

import sys
import time
import json
import os
import random
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

sys.path.insert(0, '.')

from models.trader_model import trader_model_instance
from core.risk_manager import RiskManager
from core.core_technical_trader import TechnicalTraderCore
from utils.portfolio_manager import PortfolioManager
from utils.logger import get_logger

logger = get_logger("DEEP_DIAG")


class DeepDiagnostics:
    """Глубокая диагностика всех аспектов системы"""

    def __init__(self):
        self.model = trader_model_instance
        self.risk_manager = RiskManager()
        self.technical_core = TechnicalTraderCore()

        # Создаём СВЕЖИЙ портфель для чистоты эксперимента
        self.portfolio = PortfolioManager()
        self.portfolio.cash = 10000.0
        self.portfolio.reserved_cash = 0.0
        self.portfolio.positions = {}
        self.portfolio.initial_capital = 10000.0
        self.portfolio.max_positions = self.risk_manager.config.get('max_positions', 10)

        # Тестовые тикеры с реальными параметрами
        self.test_tickers = {
            'SBER': {'lot': 10, 'step': 0.01, 'sector': 'финансы'},
            'GAZP': {'lot': 10, 'step': 0.01, 'sector': 'нефтегаз'},
            'LKOH': {'lot': 1, 'step': 0.5, 'sector': 'нефтегаз'},
            'ROSN': {'lot': 10, 'step': 0.01, 'sector': 'нефтегаз'},
            'VTBR': {'lot': 10000, 'step': 0.0001, 'sector': 'финансы'},
            'GMKN': {'lot': 1, 'step': 2.0, 'sector': 'металлы'},
            'NVTK': {'lot': 10, 'step': 0.5, 'sector': 'нефтегаз'},
            'YNDX': {'lot': 1, 'step': 2.0, 'sector': 'телеком'},
            'TATN': {'lot': 10, 'step': 0.5, 'sector': 'нефтегаз'},
            'PLZL': {'lot': 1, 'step': 10.0, 'sector': 'металлы'},
        }

        self.base_prices = {
            'SBER': 280.0, 'GAZP': 165.0, 'LKOH': 7200.0,
            'ROSN': 580.0, 'VTBR': 0.02, 'GMKN': 14500.0,
            'NVTK': 1100.0, 'YNDX': 4200.0, 'TATN': 680.0,
            'PLZL': 15000.0
        }

        # Метрики
        self.cycles = 0
        self.real_trades = 0
        self.rejected_trades = 0
        self.rejection_reasons = Counter()
        self.actions_distribution = Counter()
        self.strategies_distribution = Counter()
        self.commission_total = 0.0
        self.portfolio_values = []

        # Статистика модели
        self.policy_weights_stats = {}
        self.gradient_stats = {}

        print("\n" + "=" * 80)
        print("🔬 ГЛУБОКАЯ ДИАГНОСТИКА ТОРГОВОЙ СИСТЕМЫ")
        print("=" * 80)
        self._print_system_info()

    def _print_system_info(self):
        """Вывод информации о системе"""
        print(f"\n📋 СИСТЕМНАЯ ИНФОРМАЦИЯ:")
        print(f"   Память модели: {len(self.model.memory)} опытов")
        print(f"   Exploration rate: {self.model.exploration_rate:.4f}")
        print(f"   Action dim: {self.model.action_dim}")
        print(f"   Действий BUY: 3, HOLD: 1, SELL: 3")
        print(f"   Стратегий: {len(self.model.strategies)}")

        # Статистика по стратегиям
        print(f"\n   Стратегии:")
        for name, params in self.model.strategies.items():
            perf = self.model.strategy_performance.get(name, {})
            print(f"     {name:20s} risk={params['risk_multiplier']:.1f} "
                  f"hold={params['target_hold_time_hours']:.0f}ч "
                  f"win_rate={perf.get('win_rate', 0):.1%} "
                  f"trades={perf.get('total_trades', 0)}")

        print(f"\n💰 ПОРТФЕЛЬ:")
        print(f"   Кэш: {self.portfolio.cash:,.0f}₽")
        print(f"   min_cash_per_trade: {self.risk_manager.config.get('min_cash_per_trade', 1000)}₽")
        print(f"   max_daily_trades: {self.risk_manager.config.get('max_daily_trades', 50)}")
        print(f"   max_positions: {self.risk_manager.config.get('max_positions', 10)}")
        print(f"   commission_rate: {self.portfolio.commission_rate * 100}%")

        print(f"\n🎲 EXPLORATION:")
        expl = self.model.rl_config.get('exploration', {})
        print(f"   initial: {expl.get('initial_exploration_rate', 0.1)}")
        print(f"   final: {expl.get('final_exploration_rate', 0.03)}")
        print(f"   action_exploration: {expl.get('action_exploration_rate', 0.05)}")

        print(f"\n🎯 REWARD:")
        rew = self.model.rl_config.get('reward_config', {})
        print(f"   pnl_scale: {rew.get('pnl_scale_factor', 100)}")
        print(f"   commission_penalty_scale: {rew.get('commission_penalty_scale', 50)}")
        print(f"   concentration_penalty: {rew.get('concentration_penalty_per_position', 0.1)}")
        print(f"   clip: [{rew.get('reward_clip_min', -10)}, {rew.get('reward_clip_max', 20)}]")

    def _create_state(self, ticker: str, price: float) -> torch.Tensor:
        """Создание состояния как в реальной системе"""
        security_info = {
            'lot_size': self.test_tickers[ticker]['lot'],
            'min_step': self.test_tickers[ticker]['step'],
            'sector': self.test_tickers[ticker]['sector'],
            'momentum': random.uniform(-0.02, 0.02),
            'volume': random.randint(100000, 10000000),
            'spread': 0.001,
            'market_cap': 1e11
        }

        indicators = self.technical_core.calculate_indicators(ticker)

        # News features
        news_features = torch.zeros(1, self.model.news_encoded_dim).to(self.model.device)

        market_data = {
            'volume': security_info['volume'],
            'spread': security_info['spread'],
            'rsi': indicators.get('rsi', 50),
            'atr': indicators.get('atr', price * 0.02),
            'sma_10_ratio': indicators.get('sma_10', price) / price if price > 0 else 1.0,
            'sma_20_ratio': indicators.get('sma_20', price) / price if price > 0 else 1.0,
            'bb_position': indicators.get('bb_position', 0.5),
            'volume_ratio': indicators.get('volume_ratio', 1.0),
            'market_cap': security_info['market_cap'],
            'lot_size': security_info['lot_size'],
            'min_step': security_info['min_step'],
            'sector': security_info['sector'],
            'momentum': security_info['momentum'],
            'imoex': 3000.0,
            'imoex_change': 0.0,
            'rtsi': 1000.0,
            'rtsi_change': 0.0,
            'rvi': 20.0,
            'rvi_change': 0.0,
            'moexog': 0,
            'moexfn': 0,
            'brent': 80.0,
            'brent_change': 0.0,
            'market_liquidity_ratio': 0.5,
            'market_activity_score': 0.5,
        }

        state = self.model.build_state_vector(
            ticker=ticker,
            price=price,
            momentum=security_info['momentum'],
            sentiment=0.0,
            news_features=news_features,
            market_data=market_data,
            market_sentiment=self.model.market_sentiment,
            portfolio=self.portfolio
        )

        return state

    def _try_real_trade(self, ticker: str, price: float, action_str: str,
                        strategy: str, confidence: float) -> Dict:
        """Попытка реальной сделки через portfolio.buy/sell"""
        result = {
            'executed': False,
            'rejected_reason': None,
            'commission': 0.0,
            'quantity': 0,
            'cost': 0.0
        }

        lot = self.test_tickers[ticker]['lot']
        step = self.test_tickers[ticker]['step']

        # Корректируем цену под шаг
        if step > 0:
            price = round(price / step) * step

        if action_str.startswith('BUY'):
            target_ratio = self.model.rl_config.get('position_sizes', {}).get(action_str, 0.05)
            portfolio_value = self.portfolio.get_total_value({ticker: price})

            if portfolio_value <= 0:
                result['rejected_reason'] = 'zero_portfolio'
                return result

            target_value = portfolio_value * target_ratio
            quantity = int(target_value / price)

            # Лотность
            if lot > 1:
                quantity = (quantity // lot) * lot

            if quantity < lot:
                quantity = lot  # Минимум 1 лот

            cost = quantity * price
            commission = self.portfolio._calculate_commission(cost)
            total_required = cost + commission

            available = self.portfolio.cash - self.portfolio.reserved_cash

            # Проверка 1: min_cash_per_trade
            min_cash = self.risk_manager.config.get('min_cash_per_trade', 1000)
            if cost < min_cash:
                result['rejected_reason'] = f'below_min_cash ({cost:.0f} < {min_cash})'
                return result

            # Проверка 2: доступный кэш
            if total_required > available:
                result['rejected_reason'] = f'insufficient_cash (need {total_required:.0f}, have {available:.0f})'
                return result

            # Проверка 3: max_positions (через portfolio)
            if ticker not in self.portfolio.positions:
                if len(self.portfolio.positions) >= self.portfolio.max_positions:
                    result[
                        'rejected_reason'] = f'max_positions ({len(self.portfolio.positions)}/{self.portfolio.max_positions})'
                    return result

            # Проверка 4: дневной лимит сделок
            if self.risk_manager.daily_trades >= self.risk_manager.config.get('max_daily_trades', 50):
                result['rejected_reason'] = 'daily_limit_reached'
                return result

            # Проверка 5: Risk Manager
            if not self.risk_manager.check_daily_limits():
                result['rejected_reason'] = 'risk_manager_blocked'
                return result

            # ВЫПОЛНЕНИЕ СДЕЛКИ
            success = self.portfolio.buy(
                ticker, quantity, price, strategy,
                lot_size=lot, min_step=step
            )

            if success:
                self.risk_manager.daily_trades += 1
                result['executed'] = True
                result['commission'] = commission
                result['quantity'] = quantity
                result['cost'] = cost
            else:
                result['rejected_reason'] = 'portfolio.buy_failed'

        elif action_str.startswith('SELL'):
            if ticker not in self.portfolio.positions:
                result['rejected_reason'] = 'no_position'
                return result

            pos = self.portfolio.positions[ticker]
            sell_ratio = self.model.rl_config.get('position_sizes', {}).get(action_str, 0.5)
            quantity = max(lot, int(pos['qty'] * sell_ratio))

            if lot > 1:
                quantity = (quantity // lot) * lot

            if quantity > pos['qty']:
                quantity = pos['qty']

            if quantity < lot:
                result['rejected_reason'] = 'below_lot_size'
                return result

            revenue = quantity * price
            commission = self.portfolio._calculate_commission(revenue)

            # ВЫПОЛНЕНИЕ СДЕЛКИ
            success, pnl = self.portfolio.sell(ticker, quantity, price)

            if success:
                self.risk_manager.daily_trades += 1
                result['executed'] = True
                result['commission'] = commission
                result['quantity'] = quantity
                result['cost'] = revenue
                result['pnl'] = pnl
            else:
                result['rejected_reason'] = 'portfolio.sell_failed'

        return result

    def analyze_model_weights(self):
        """Анализ весов модели"""
        print(f"\n🧠 АНАЛИЗ ВЕСОВ МОДЕЛИ:")

        # Веса action_net (последний слой — вероятности действий)
        for name, param in self.model.policy_net.named_parameters():
            if 'action_net' in name and param.requires_grad:
                weights = param.data.cpu().numpy()
                print(f"   {name}:")
                print(f"      shape={weights.shape}, mean={weights.mean():.4f}, "
                      f"std={weights.std():.4f}, "
                      f"min={weights.min():.4f}, max={weights.max():.4f}")

                # Для последнего слоя — анализ bias (склонность к действиям)
                if 'bias' in name and weights.shape[0] == self.model.action_dim:
                    print(f"      BIAS (склонность к каждому действию):")
                    action_names = ['BUY_MIN', 'BUY_SMALL', 'BUY_NORMAL',
                                    'HOLD', 'SELL_SMALL', 'SELL_NORMAL', 'SELL_ALL']
                    for i, (bias_val, action_name) in enumerate(zip(weights, action_names)):
                        prob_bias = np.exp(bias_val) / np.sum(np.exp(weights))  # softmax одного bias
                        print(f"        {action_name:15s}: bias={bias_val:+.4f} (~prob={prob_bias:.2%})")

        # Статистика градиентов (если есть)
        total_norm = 0
        for p in self.model.policy_net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        print(f"\n   Gradient norm: {total_norm:.6f}")

    def analyze_memory(self):
        """Анализ памяти модели"""
        print(f"\n📦 АНАЛИЗ ПАМЯТИ МОДЕЛИ:")
        print(f"   Всего опытов: {len(self.model.memory)}")

        if len(self.model.memory) == 0:
            return

        # Распределение действий в памяти
        actions_in_memory = Counter()
        rewards_in_memory = []
        dones_in_memory = []

        for exp in self.model.memory:
            actions_in_memory[exp.get('action', -1)] += 1
            rewards_in_memory.append(float(exp.get('reward', 0)))
            dones_in_memory.append(exp.get('done', False))

        print(f"   Распределение действий в памяти:")
        action_names = ['BUY_MIN', 'BUY_SMALL', 'BUY_NORMAL',
                        'HOLD', 'SELL_SMALL', 'SELL_NORMAL', 'SELL_ALL']
        for action_idx in range(self.model.action_dim):
            count = actions_in_memory.get(action_idx, 0)
            pct = count / len(self.model.memory) * 100
            print(f"     {action_names[action_idx]:15s}: {count:5d} ({pct:5.1f}%)")

        print(f"\n   Rewards в памяти:")
        print(f"     mean={np.mean(rewards_in_memory):.4f}, "
              f"median={np.median(rewards_in_memory):.4f}, "
              f"std={np.std(rewards_in_memory):.4f}")
        print(f"     min={np.min(rewards_in_memory):.4f}, max={np.max(rewards_in_memory):.4f}")
        print(f"     positive: {sum(1 for r in rewards_in_memory if r > 0)}/{len(rewards_in_memory)}")
        print(f"     negative: {sum(1 for r in rewards_in_memory if r < 0)}/{len(rewards_in_memory)}")

        print(f"\n   Done ratio: {sum(dones_in_memory) / len(dones_in_memory):.1%}")

    def analyze_action_choice(self, num_samples: int = 100):
        """Анализ выбора действий моделью"""
        print(f"\n🎯 АНАЛИЗ ВЫБОРА ДЕЙСТВИЙ ({num_samples} сэмплов):")

        action_names = ['BUY_MIN', 'BUY_SMALL', 'BUY_NORMAL',
                        'HOLD', 'SELL_SMALL', 'SELL_NORMAL', 'SELL_ALL']

        all_actions = []
        all_strategies = []
        all_confidences = []
        all_values = []

        for i in range(num_samples):
            ticker = random.choice(list(self.test_tickers.keys()))
            price = self.base_prices[ticker] * random.uniform(0.98, 1.02)

            state = self._create_state(ticker, price)

            market_context = {
                'market_sentiment': self.model.market_sentiment,
                'volatility': self.model.volatility_index,
                'confidence': 0.7,
                'time_of_day': datetime.now().hour / 24.0,
                'ticker_sentiment': 0.0,
                'assigned_horizon': 'week'
            }

            try:
                action, strategy, confidence = self.model.choose_action_with_strategy(
                    state=state, ticker=ticker, price=price,
                    market_context=market_context
                )

                all_actions.append(action)
                all_strategies.append(strategy)
                all_confidences.append(confidence)

                # Получаем value для состояния
                strategy_params = self.model.strategies.get(strategy, self.model.strategies['balanced'])
                full_state = self.model._create_strategy_state(state, strategy_params)
                value = self.model.get_state_value(full_state)
                all_values.append(value)

            except Exception as e:
                print(f"   ⚠ Ошибка на сэмпле {i}: {e}")

        print(f"\n   Распределение действий:")
        action_counts = Counter(all_actions)
        for action_idx in range(self.model.action_dim):
            count = action_counts.get(action_idx, 0)
            pct = count / len(all_actions) * 100
            bar = '█' * int(pct / 2)
            print(f"     {action_names[action_idx]:15s}: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\n   Распределение стратегий:")
        strategy_counts = Counter(all_strategies)
        for strategy_name in sorted(self.model.strategies.keys()):
            count = strategy_counts.get(strategy_name, 0)
            pct = count / len(all_actions) * 100
            print(f"     {strategy_name:20s}: {count:4d} ({pct:5.1f}%)")

        print(f"\n   Confidence:")
        print(f"     mean={np.mean(all_confidences):.3f}, "
              f"median={np.median(all_confidences):.3f}, "
              f"min={np.min(all_confidences):.3f}")

        print(f"\n   State Values:")
        print(f"     mean={np.mean(all_values):.4f}, "
              f"min={np.min(all_values):.4f}, max={np.max(all_values):.4f}")

        # Анализ: предпочитает ли модель HOLD при низкой уверенности?
        hold_actions = [a for a, c in zip(all_actions, all_confidences) if a == 3]
        non_hold_actions = [a for a, c in zip(all_actions, all_confidences) if a != 3]
        if hold_actions:
            print(
                f"\n   HOLD: {len(hold_actions)} раз, средняя confidence={np.mean([c for a, c in zip(all_actions, all_confidences) if a == 3]):.3f}")
        if non_hold_actions:
            non_hold_confs = [c for a, c in zip(all_actions, all_confidences) if a != 3]
            print(f"   NON-HOLD: {len(non_hold_actions)} раз, средняя confidence={np.mean(non_hold_confs):.3f}")

    def run_trading_simulation(self, cycles: int = 30):
        """Симуляция торговли с реальными проверками"""
        print(f"\n💹 СИМУЛЯЦИЯ ТОРГОВЛИ ({cycles} циклов):")
        print("-" * 80)

        for cycle in range(cycles):
            # Обновляем цены (±0.5%)
            for ticker in self.test_tickers:
                old_price = self.base_prices[ticker]
                new_price = old_price * (1 + random.uniform(-0.005, 0.005))
                self.base_prices[ticker] = new_price
                self.technical_core.update_price_data(ticker, new_price)

            # Для каждого тикера — выбор действия
            for ticker in list(self.test_tickers.keys())[:5]:  # первые 5
                price = self.base_prices[ticker]

                try:
                    state = self._create_state(ticker, price)
                except Exception as e:
                    continue

                market_context = {
                    'market_sentiment': self.model.market_sentiment,
                    'volatility': self.model.volatility_index,
                    'confidence': 0.7,
                    'time_of_day': datetime.now().hour / 24.0,
                    'ticker_sentiment': 0.0,
                    'assigned_horizon': 'week'
                }

                try:
                    action, strategy, confidence = self.model.choose_action_with_strategy(
                        state=state, ticker=ticker, price=price,
                        market_context=market_context
                    )
                except Exception as e:
                    continue

                action_str = self.model.rl_config.get('action_mapping', {}).get(str(action), 'HOLD')
                self.actions_distribution[action_str] += 1
                self.strategies_distribution[strategy] += 1

                if action_str == 'HOLD':
                    continue

                # Пробуем реальную сделку
                result = self._try_real_trade(ticker, price, action_str, strategy, confidence)

                if result['executed']:
                    self.real_trades += 1
                    self.commission_total += result['commission']
                else:
                    self.rejected_trades += 1
                    self.rejection_reasons[result['rejected_reason']] += 1

            self.cycles += 1
            portfolio_value = self.portfolio.get_total_value(self.base_prices)
            self.portfolio_values.append(portfolio_value)

            # Вывод каждые 10 циклов
            if (cycle + 1) % 10 == 0:
                print(f"\n   Цикл {cycle + 1}/{cycles}:")
                print(f"     Портфель: {portfolio_value:,.0f}₽ (кэш: {self.portfolio.cash:,.0f}₽)")
                print(f"     Позиций: {len(self.portfolio.positions)}")
                print(f"     Реальных сделок: {self.real_trades}")
                print(f"     Отклонено: {self.rejected_trades}")
                print(f"     Комиссий: {self.commission_total:.2f}₽")

    def print_final_report(self):
        """Итоговый отчёт"""
        print("\n" + "=" * 80)
        print("📋 ИТОГОВЫЙ ДИАГНОСТИЧЕСКИЙ ОТЧЁТ")
        print("=" * 80)

        print(f"\n💹 ТОРГОВЛЯ:")
        print(f"   Циклов: {self.cycles}")
        print(
            f"   Сигналов (не HOLD): {sum(self.actions_distribution.values()) - self.actions_distribution.get('HOLD', 0)}")
        print(f"   HOLD сигналов: {self.actions_distribution.get('HOLD', 0)}")
        print(f"   Реальных сделок: {self.real_trades}")
        print(f"   Отклонено сделок: {self.rejected_trades}")
        print(f"   Процент исполнения: {self.real_trades / (self.real_trades + self.rejected_trades) * 100:.1f}%"
              if (self.real_trades + self.rejected_trades) > 0 else "   Процент исполнения: N/A")

        if self.rejection_reasons:
            print(f"\n   Причины отклонения:")
            for reason, count in self.rejection_reasons.most_common():
                print(f"     {reason}: {count}")

        print(f"\n   Комиссий всего: {self.commission_total:.2f}₽")
        if self.cycles > 0:
            print(f"   Комиссий в час (прогноз): {self.commission_total / self.cycles * 360:.2f}₽")

        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ДЕЙСТВИЙ:")
        action_names = ['BUY_MIN', 'BUY_SMALL', 'BUY_NORMAL',
                        'HOLD', 'SELL_SMALL', 'SELL_NORMAL', 'SELL_ALL']
        total_actions = sum(self.actions_distribution.values())
        for action_name in action_names:
            count = self.actions_distribution.get(action_name, 0)
            pct = count / total_actions * 100 if total_actions > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"   {action_name:15s}: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\n📊 РАСПРЕДЕЛЕНИЕ СТРАТЕГИЙ:")
        total_strategies = sum(self.strategies_distribution.values())
        for name in sorted(self.strategies_distribution.keys()):
            count = self.strategies_distribution[name]
            pct = count / total_strategies * 100 if total_strategies > 0 else 0
            print(f"   {name:20s}: {count:4d} ({pct:5.1f}%)")

        print(f"\n📈 ДИНАМИКА ПОРТФЕЛЯ:")
        if self.portfolio_values:
            print(f"   Начало: {self.portfolio_values[0]:,.0f}₽")
            print(f"   Конец: {self.portfolio_values[-1]:,.0f}₽")
            change = self.portfolio_values[-1] - self.portfolio_values[0]
            print(f"   Изменение: {change:+,.0f}₽ ({change / self.portfolio_values[0] * 100:+.2f}%)")

        print(f"\n⚠ КЛЮЧЕВЫЕ ПРОБЛЕМЫ (по результатам диагностики):")
        print(
            f"   1. Модель выбирает HOLD в {self.actions_distribution.get('HOLD', 0) / total_actions * 100:.1f}% случаев")
        print(
            f"   2. BUY-действия: {sum(self.actions_distribution.get(a, 0) for a in ['BUY_MIN', 'BUY_SMALL', 'BUY_NORMAL'])} "
            f"({sum(self.actions_distribution.get(a, 0) for a in ['BUY_MIN', 'BUY_SMALL', 'BUY_NORMAL']) / total_actions * 100:.1f}%)")
        print(f"   3. Реальных сделок: {self.real_trades} (остальные отбиты проверками)")
        print(f"   4. Risk Manager вызывается при каждой сделке: {'ДА' if self.real_trades > 0 else 'Н/Д'}")
        print(
            f"   5. Основная причина отказов: {self.rejection_reasons.most_common(1)[0][0] if self.rejection_reasons else 'нет отказов'}")

        if self.actions_distribution.get('HOLD', 0) / total_actions < 0.3:
            print(f"\n   🔴 КРИТИЧНО: HOLD < 30% — модель архитектурно смещена к торговле!")
            print(f"      (6 из 7 действий — торговые, модель просто не может выбрать HOLD чаще)")

    def run_full_diagnostics(self, trading_cycles: int = 30, action_samples: int = 100):
        """Полная диагностика"""
        print("\n" + "=" * 80)
        print("🔬 ЗАПУСК ПОЛНОЙ ДИАГНОСТИКИ")
        print("=" * 80)

        # 1. Анализ весов модели
        self.analyze_model_weights()

        # 2. Анализ памяти
        self.analyze_memory()

        # 3. Анализ выбора действий
        self.analyze_action_choice(action_samples)

        # 4. Симуляция торговли
        self.run_trading_simulation(trading_cycles)

        # 5. Итоговый отчёт
        self.print_final_report()

        print("\n" + "=" * 80)
        print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
        print("=" * 80)


def main():
    diag = DeepDiagnostics()
    diag.run_full_diagnostics(trading_cycles=30, action_samples=100)


if __name__ == "__main__":
    main()