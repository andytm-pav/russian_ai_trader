#!/usr/bin/env python3
"""
ГЛУБОКАЯ ДИАГНОСТИКА ТОРГОВОЙ СИСТЕМЫ (v3)
Проверяет:
1. Реальные сделки через portfolio.buy/sell с проверкой кэша и лимитов
2. Параметры модели: веса, градиенты, распределение действий
3. Параметры обучения: exploration, decay, reward
4. Статистику стратегий и тикеров
5. Память модели: какие действия преобладают
ИСПОЛЬЗУЕТ ИЗОЛИРОВАННЫЙ ТЕСТОВЫЙ ПОРТФЕЛЬ (test_portfolio.json)
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

        # Читаем action_mapping из конфига
        self.action_mapping = self.model.rl_config.get('action_mapping', {})

        # Определяем категории действий
        self.buy_actions = set()
        self.hold_actions = set()
        self.sell_actions = set()

        for idx, name in self.action_mapping.items():
            if name.startswith('BUY'):
                self.buy_actions.add(int(idx))
            elif name.startswith('HOLD'):
                self.hold_actions.add(int(idx))
            elif name.startswith('SELL'):
                self.sell_actions.add(int(idx))

        # ИЗОЛИРОВАННЫЙ портфель для тестов (не затрагивает боевой portfolio_state.json)
        self.portfolio = PortfolioManager(portfolio_file="data/test_portfolio.json")
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

        print("\n" + "=" * 80)
        print("🔬 ГЛУБОКАЯ ДИАГНОСТИКА ТОРГОВОЙ СИСТЕМЫ (v3)")
        print("=" * 80)
        self._print_system_info()

    def _print_system_info(self):
        """Вывод информации о системе"""
        print(f"\n📋 СИСТЕМНАЯ ИНФОРМАЦИЯ:")
        print(f"   Память модели: {len(self.model.memory)} опытов")
        print(f"   Exploration rate: {self.model.exploration_rate:.4f}")
        print(f"   Action dim: {self.model.action_dim}")
        print(
            f"   Действий BUY: {len(self.buy_actions)}, HOLD: {len(self.hold_actions)}, SELL: {len(self.sell_actions)}")

        # Детализация action_mapping
        print(f"   action_mapping:")
        for idx in sorted(self.action_mapping.keys(), key=int):
            name = self.action_mapping[idx]
            print(f"     {idx}: {name}")

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
        """Создание состояния как в реальной системе — с накоплением истории"""
        security_info = {
            'lot_size': self.test_tickers[ticker]['lot'],
            'min_step': self.test_tickers[ticker]['step'],
            'sector': self.test_tickers[ticker]['sector'],
            'momentum': random.uniform(-0.02, 0.02),
            'volume': random.randint(100000, 10000000),
            'spread': 0.001,
            'market_cap': 1e11
        }

        # НАКАПЛИВАЕМ ИСТОРИЮ ЦЕН (как в реальной системе)
        self.technical_core.update_price_data(ticker, price)
        indicators = self.technical_core.calculate_indicators(ticker)

        # Если индикаторы пусты — используем дефолты
        if not indicators:
            indicators = {
                'rsi': 50, 'atr': price * 0.02, 'sma_10': price, 'sma_20': price,
                'bb_position': 0.5, 'volume_ratio': 1.0
            }

        # Кодируем новости с защитой от None
        news_features = self.model.encode_news(['тестовая новость'])
        if news_features is None or (hasattr(news_features, 'numel') and news_features.numel() == 0):
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
            'market_mood': 0.0,
            'shares_turnover': 0,
            'rvi_normalized': 0.2,
            'imoex_normalized': 0.75,
            'market_cap_total': 0.5,
            'liquidity_ratio': 0.5,
            'cbr_rate_normalized': 0.5,
            'usd_rub': 0.8,
            'moexog_normalized': 0.0,
            'spread_pct': 0.0001,
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

            if lot > 1:
                quantity = (quantity // lot) * lot

            if quantity < lot:
                quantity = lot

            cost = quantity * price
            commission = self.portfolio._calculate_commission(cost)
            total_required = cost + commission

            available = self.portfolio.cash - self.portfolio.reserved_cash

            min_cash = self.risk_manager.config.get('min_cash_per_trade', 1000)
            if cost < min_cash:
                result['rejected_reason'] = f'below_min_cash ({cost:.0f} < {min_cash})'
                return result

            if total_required > available:
                result['rejected_reason'] = f'insufficient_cash (need {total_required:.0f}, have {available:.0f})'
                return result

            if ticker not in self.portfolio.positions:
                if len(self.portfolio.positions) >= self.portfolio.max_positions:
                    result[
                        'rejected_reason'] = f'max_positions ({len(self.portfolio.positions)}/{self.portfolio.max_positions})'
                    return result

            if self.risk_manager.daily_trades >= self.risk_manager.config.get('max_daily_trades', 50):
                result['rejected_reason'] = 'daily_limit_reached'
                return result

            if not self.risk_manager.check_daily_limits():
                result['rejected_reason'] = 'risk_manager_blocked'
                return result

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

    def _classify_action(self, action_idx: int) -> str:
        """Классификация действия по action_mapping из конфига"""
        action_str = self.action_mapping.get(str(action_idx), f'UNKNOWN_{action_idx}')

        if action_str.startswith('BUY'):
            return 'BUY'
        elif action_str.startswith('HOLD'):
            return 'HOLD'
        elif action_str.startswith('SELL'):
            return 'SELL'
        return 'UNKNOWN'

    def analyze_model_weights(self):
        """Анализ весов модели"""
        print(f"\n🧠 АНАЛИЗ ВЕСОВ МОДЕЛИ:")

        for name, param in self.model.policy_net.named_parameters():
            if 'action_net' in name and param.requires_grad:
                weights = param.data.cpu().numpy()
                print(f"   {name}:")
                print(f"      shape={weights.shape}, mean={weights.mean():.4f}, "
                      f"std={weights.std():.4f}, "
                      f"min={weights.min():.4f}, max={weights.max():.4f}")

                if 'bias' in name and weights.shape[0] == self.model.action_dim:
                    print(f"      BIAS (склонность к каждому действию):")
                    for i, bias_val in enumerate(weights):
                        action_name = self.action_mapping.get(str(i), f'UNKNOWN_{i}')
                        prob_bias = np.exp(bias_val) / np.sum(np.exp(weights))
                        category = self._classify_action(i)
                        print(
                            f"        [{category:6s}] {action_name:15s}: bias={bias_val:+.4f} (~prob={prob_bias:.2%})")

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

        # Распределение действий в памяти с учётом категорий
        actions_in_memory = Counter()
        buy_in_memory = 0
        hold_in_memory = 0
        sell_in_memory = 0
        rewards_in_memory = []
        dones_in_memory = []

        for exp in self.model.memory:
            action = exp.get('action', -1)
            actions_in_memory[action] += 1

            category = self._classify_action(action)
            if category == 'BUY':
                buy_in_memory += 1
            elif category == 'HOLD':
                hold_in_memory += 1
            elif category == 'SELL':
                sell_in_memory += 1

            rewards_in_memory.append(float(exp.get('reward', 0)))
            dones_in_memory.append(exp.get('done', False))

        total = len(self.model.memory)
        print(f"   Распределение по категориям:")
        print(f"     BUY:  {buy_in_memory:5d} ({buy_in_memory / total * 100:5.1f}%)")
        print(f"     HOLD: {hold_in_memory:5d} ({hold_in_memory / total * 100:5.1f}%)")
        print(f"     SELL: {sell_in_memory:5d} ({sell_in_memory / total * 100:5.1f}%)")

        print(f"\n   Распределение действий в памяти:")
        for action_idx in sorted(actions_in_memory.keys()):
            count = actions_in_memory[action_idx]
            pct = count / total * 100
            action_name = self.action_mapping.get(str(action_idx), f'UNKNOWN_{action_idx}')
            category = self._classify_action(action_idx)
            print(f"     [{category:6s}] {action_name:15s}: {count:5d} ({pct:5.1f}%)")

        print(f"\n   Rewards в памяти:")
        print(f"     mean={np.mean(rewards_in_memory):.4f}, "
              f"median={np.median(rewards_in_memory):.4f}, "
              f"std={np.std(rewards_in_memory):.4f}")
        print(f"     min={np.min(rewards_in_memory):.4f}, max={np.max(rewards_in_memory):.4f}")
        print(f"     positive: {sum(1 for r in rewards_in_memory if r > 0)}/{total}")
        print(f"     negative: {sum(1 for r in rewards_in_memory if r < 0)}/{total}")

        print(f"\n   Done ratio: {sum(dones_in_memory) / total:.1%}")

    def analyze_action_choice(self, num_samples: int = 100):
        """Анализ выбора действий моделью"""
        print(f"\n🎯 АНАЛИЗ ВЫБОРА ДЕЙСТВИЙ ({num_samples} сэмплов):")

        all_actions = []
        all_strategies = []
        all_confidences = []
        all_values = []
        categories = Counter()
        errors_count = 0

        for i in range(num_samples):
            ticker = random.choice(list(self.test_tickers.keys()))
            price = self.base_prices[ticker] * random.uniform(0.98, 1.02)

            try:
                state = self._create_state(ticker, price)
            except Exception as e:
                errors_count += 1
                if errors_count <= 3:
                    print(f"   ⚠ Ошибка создания состояния {ticker}: {e}")
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

                all_actions.append(action)
                all_strategies.append(strategy)
                all_confidences.append(confidence)
                categories[self._classify_action(action)] += 1

                strategy_params = self.model.strategies.get(strategy, self.model.strategies['balanced'])
                full_state = self.model._create_strategy_state(state, strategy_params)
                value = self.model.get_state_value(full_state)
                all_values.append(value)

            except Exception as e:
                errors_count += 1
                if errors_count <= 3:
                    print(f"   ⚠ Ошибка на сэмпле {i}: {e}")

        total = len(all_actions)

        if total == 0:
            print(f"\n   ❌ Все {num_samples} сэмплов завершились с ошибками.")
            print(f"   Проверьте загрузку BERT модели и encode_news().")
            return

        if errors_count > 0:
            print(f"\n   ⚠ Успешных сэмплов: {total}/{num_samples} (ошибок: {errors_count})")

        print(f"\n   Распределение по категориям:")
        for cat in ['BUY', 'HOLD', 'SELL']:
            count = categories.get(cat, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"     {cat:6s}: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\n   Распределение действий:")
        action_counts = Counter(all_actions)
        for action_idx in sorted(action_counts.keys()):
            count = action_counts[action_idx]
            pct = count / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 2)
            action_name = self.action_mapping.get(str(action_idx), f'UNKNOWN_{action_idx}')
            category = self._classify_action(action_idx)
            print(f"     [{category:6s}] {action_name:15s}: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\n   Распределение стратегий:")
        strategy_counts = Counter(all_strategies)
        for strategy_name in sorted(self.model.strategies.keys()):
            count = strategy_counts.get(strategy_name, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"     {strategy_name:20s}: {count:4d} ({pct:5.1f}%)")

        if all_confidences:
            print(f"\n   Confidence:")
            print(f"     mean={np.mean(all_confidences):.3f}, "
                  f"median={np.median(all_confidences):.3f}, "
                  f"min={np.min(all_confidences):.3f}")

        if all_values:
            print(f"\n   State Values:")
            print(f"     mean={np.mean(all_values):.4f}, "
                  f"min={np.min(all_values):.4f}, max={np.max(all_values):.4f}")

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
            for ticker in list(self.test_tickers.keys())[:5]:
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

                action_str = self.action_mapping.get(str(action), 'HOLD')
                self.actions_distribution[action_str] += 1
                self.strategies_distribution[strategy] += 1

                if action_str.startswith('HOLD'):
                    continue

                # Пробуем реальную сделку
                result = self._try_real_trade(ticker, price, action_str, strategy, confidence)

                if result['executed']:
                    self.real_trades += 1
                    self.commission_total += result['commission']
                else:
                    self.rejected_trades += 1
                    reason = result.get('rejected_reason', 'Unknown')
                    if reason is None:
                        reason = 'None'
                    self.rejection_reasons[reason] += 1

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

        buy_total = sum(1 for a in self.actions_distribution if a.startswith('BUY'))
        hold_total = sum(1 for a in self.actions_distribution if a.startswith('HOLD'))
        sell_total = sum(1 for a in self.actions_distribution if a.startswith('SELL'))
        total_signals = sum(self.actions_distribution.values())

        print(f"\n💹 ТОРГОВЛЯ:")
        print(f"   Циклов: {self.cycles}")
        print(f"   Сигналов всего: {total_signals}")
        if total_signals > 0:
            print(f"   BUY сигналов: {buy_total} ({buy_total / total_signals * 100:.1f}%)")
            print(f"   HOLD сигналов: {hold_total} ({hold_total / total_signals * 100:.1f}%)")
            print(f"   SELL сигналов: {sell_total} ({sell_total / total_signals * 100:.1f}%)")
        print(f"   Реальных сделок: {self.real_trades}")
        print(f"   Отклонено сделок: {self.rejected_trades}")
        if self.real_trades + self.rejected_trades > 0:
            print(f"   Процент исполнения: {self.real_trades / (self.real_trades + self.rejected_trades) * 100:.1f}%")

        if self.rejection_reasons:
            print(f"\n   Причины отклонения:")
            for reason, count in self.rejection_reasons.most_common(10):
                print(f"     {reason}: {count}")

        print(f"\n   Комиссий всего: {self.commission_total:.2f}₽")
        if self.cycles > 0:
            print(f"   Комиссий в час (прогноз): {self.commission_total / self.cycles * 360:.2f}₽")

        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ДЕЙСТВИЙ:")
        for action_name in sorted(self.actions_distribution.keys()):
            count = self.actions_distribution[action_name]
            pct = count / total_signals * 100 if total_signals > 0 else 0
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

        print(f"\n⚠ КЛЮЧЕВЫЕ МЕТРИКИ:")
        if total_signals > 0:
            hold_pct = hold_total / total_signals * 100
            print(f"   1. Модель выбирает HOLD в {hold_pct:.1f}% случаев")
            if hold_pct < 20:
                print(f"      🔴 HOLD < 20% — модель всё ещё смещена к торговле")
            elif hold_pct < 40:
                print(f"      🟡 HOLD 20–40% — модель учится держать")
            else:
                print(f"      ✅ HOLD > 40% — хороший баланс")

        print(f"   2. Реальных сделок: {self.real_trades}")
        if self.rejected_trades > 0:
            top_reason = self.rejection_reasons.most_common(1)[0]
            print(f"   3. Основная причина отказов: {top_reason[0]} ({top_reason[1]} раз)")

    def run_full_diagnostics(self, trading_cycles: int = 30, action_samples: int = 100):
        """Полная диагностика"""
        print("\n" + "=" * 80)
        print("🔬 ЗАПУСК ПОЛНОЙ ДИАГНОСТИКИ")
        print("=" * 80)

        self.analyze_model_weights()
        self.analyze_memory()
        self.analyze_action_choice(action_samples)
        self.run_trading_simulation(trading_cycles)
        self.print_final_report()

        # Удаляем тестовый портфель, чтобы не загрязнять историю
        test_file = "data/test_portfolio.json"
        if os.path.exists(test_file):
            os.remove(test_file)

        print("\n" + "=" * 80)
        print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
        print("=" * 80)


def main():
    diag = DeepDiagnostics()
    diag.run_full_diagnostics(trading_cycles=30, action_samples=100)


if __name__ == "__main__":
    main()