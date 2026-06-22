#!/usr/bin/env python3
"""
ГЛУБОКАЯ ДИАГНОСТИКА ТОРГОВОЙ СИСТЕМЫ (v4)
Проверяет через РЕАЛЬНЫЕ вызовы Risk Manager и Portfolio Manager.
Использует изолированный тестовый портфель (test_portfolio.json).
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
from typing import Dict

sys.path.insert(0, '.')

from models.trader_model import trader_model_instance
from core.risk_manager import RiskManager
from core.core_technical_trader import TechnicalTraderCore
from utils.portfolio_manager import PortfolioManager
from utils.logger import get_logger

logger = get_logger("DEEP_DIAG")


class DeepDiagnostics:
    """Глубокая диагностика через реальные вызовы системы"""

    def __init__(self):
        self.model = trader_model_instance
        self.risk_manager = RiskManager()
        self.technical_core = TechnicalTraderCore()

        # Загружаем training_wheels
        self.tw = self._load_training_wheels()

        self.action_mapping = self.model.rl_config.get('action_mapping', {})

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

        # Изолированный портфель
        self.portfolio = PortfolioManager(portfolio_file="data/test_portfolio.json")
        self.portfolio.cash = 10000.0
        self.portfolio.reserved_cash = 0.0
        self.portfolio.positions = {}
        self.portfolio.initial_capital = 10000.0
        self.portfolio.max_positions = self.risk_manager.config.get('max_positions', 10)

        # Тестовые тикеры
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

        self.cycles = 0
        self.real_trades = 0
        self.rejected_trades = 0
        self.rejection_reasons = Counter()
        self.actions_distribution = Counter()
        self.strategies_distribution = Counter()
        self.strategy_pnl = defaultdict(float)
        self.strategy_trades = defaultdict(int)
        self.commission_total = 0.0
        self.portfolio_values = []

        print("\n" + "=" * 80)
        print("🔬 ГЛУБОКАЯ ДИАГНОСТИКА ТОРГОВОЙ СИСТЕМЫ (v4)")
        print("=" * 80)
        self._print_system_info()

    def _load_training_wheels(self) -> Dict:
        try:
            with open("config/training_wheels.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def _get_commission(self, amount: float) -> float:
        """Расчёт комиссии через публичный интерфейс"""
        rate = getattr(self.portfolio, 'commission_rate', 0.003)
        return amount * rate

    def _print_system_info(self):
        print(f"\n📋 СИСТЕМНАЯ ИНФОРМАЦИЯ:")
        print(f"   Память модели: {len(self.model.memory)} опытов")
        print(f"   Exploration rate: {self.model.exploration_rate:.4f}")
        print(f"   Action dim: {self.model.action_dim}")
        print(f"   Действий BUY: {len(self.buy_actions)}, HOLD: {len(self.hold_actions)}, SELL: {len(self.sell_actions)}")

        print(f"\n   action_mapping:")
        for idx in sorted(self.action_mapping.keys(), key=int):
            print(f"     {idx}: {self.action_mapping[idx]}")

        print(f"\n   Стратегии:")
        for name, params in self.model.strategies.items():
            perf = self.model.strategy_performance.get(name, {})
            print(f"     {name:20s} risk={params['risk_multiplier']:.1f} "
                  f"hold={params['target_hold_time_hours']:.0f}ч "
                  f"win_rate={perf.get('win_rate', 0):.1%} "
                  f"trades={perf.get('total_trades', 0)}")

        print(f"\n💰 ПОРТФЕЛЬ:")
        print(f"   Кэш: {self.portfolio.cash:,.0f}₽")
        tw_limits = self.tw.get('trade_limits', {})
        tw_risk = self.tw.get('risk_params', {})
        print(f"   min_cash_per_trade: {tw_limits.get('min_cash_per_trade', 1000)}₽ (training_wheels)")
        print(f"   max_daily_trades: {tw_limits.get('max_daily_trades', 50)} (training_wheels)")
        print(f"   risk_per_trade: {tw_risk.get('risk_per_trade_percent', 3.0)}% (training_wheels)")
        print(f"   max_position_weight: {tw_risk.get('max_position_weight_percent', 40)}% (training_wheels)")
        print(f"   commission_rate: {getattr(self.portfolio, 'commission_rate', 0.003) * 100}%")

        print(f"\n🎲 EXPLORATION:")
        expl = self.model.rl_config.get('exploration', {})
        init_rate = expl.get('initial_exploration_rate', 0.1)
        action_rate = expl.get('action_exploration_rate', 0.01)
        print(f"   strategy_exploration: {init_rate}")
        print(f"   action_exploration: {action_rate}")
        effective = init_rate * action_rate
        print(f"   ЭФФЕКТИВНЫЙ exploration (strategy × action): {effective:.4f} = {effective*100:.2f}%")
        if effective < 0.01:
            print(f"   🔴 КРИТИЧНО: эффективный exploration < 1% — модель застрянет")
        elif effective < 0.05:
            print(f"   🟡 НИЗКИЙ: эффективный exploration < 5% — медленное исследование")
        else:
            print(f"   ✅ НОРМА: эффективный exploration достаточен")

        print(f"\n🎯 REWARD:")
        rew = self.model.rl_config.get('reward_config', {})
        print(f"   pnl_scale: {rew.get('pnl_scale_factor', 100)}")
        print(f"   clip: [{rew.get('reward_clip_min', -5)}, {rew.get('reward_clip_max', 5)}]")
        print(f"   concentration_penalty: {rew.get('concentration_penalty_per_position', 0.5)} за позицию "
              f"(> {rew.get('max_positions_before_penalty', 3)})")

    def _create_state(self, ticker: str, price: float) -> torch.Tensor:
        security_info = {
            'lot_size': self.test_tickers[ticker]['lot'],
            'min_step': self.test_tickers[ticker]['step'],
            'sector': self.test_tickers[ticker]['sector'],
            'momentum': random.uniform(-0.02, 0.02),
            'volume': random.randint(100000, 10000000),
            'spread': 0.001,
            'market_cap': 1e11
        }

        self.technical_core.update_price_data(ticker, price)
        indicators = self.technical_core.calculate_indicators(ticker)
        if not indicators:
            indicators = {
                'rsi': 50, 'atr': price * 0.02, 'sma_10': price, 'sma_20': price,
                'bb_position': 0.5, 'volume_ratio': 1.0
            }

        news_features = self.model.encode_news(['тестовая новость'])
        if news_features is None or news_features.numel() == 0:
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
            'imoex': 3000.0, 'imoex_change': 0.0, 'rtsi': 1000.0, 'rtsi_change': 0.0,
            'rvi': 20.0, 'rvi_change': 0.0, 'moexog': 0, 'moexfn': 0,
            'brent': 80.0, 'brent_change': 0.0,
            'market_liquidity_ratio': 0.5, 'market_activity_score': 0.5,
            'market_mood': 0.0, 'shares_turnover': 0,
            'rvi_normalized': 0.2, 'imoex_normalized': 0.75, 'market_cap_total': 0.5,
            'liquidity_ratio': 0.5, 'cbr_rate_normalized': 0.5, 'usd_rub': 0.8,
            'moexog_normalized': 0.0, 'spread_pct': 0.0001,
            'market_regime': 0,
        }

        return self.model.build_state_vector(
            ticker=ticker, price=price,
            momentum=security_info['momentum'], sentiment=0.0,
            news_features=news_features, market_data=market_data,
            market_sentiment=self.model.market_sentiment,
            portfolio=self.portfolio
        )

    def _try_real_trade(self, ticker: str, price: float, action_str: str,
                        strategy: str, confidence: float) -> Dict:
        """Попытка сделки через РЕАЛЬНЫЙ Risk Manager"""
        result = {'executed': False, 'rejected_reason': None, 'quantity': 0, 'cost': 0.0}

        lot = self.test_tickers[ticker]['lot']
        step = self.test_tickers[ticker]['step']
        if step > 0:
            price = round(price / step) * step

        # Обновляем состояние портфеля в risk_manager
        self.risk_manager.update_portfolio_state({
            'total_value': self.portfolio.get_total_value({ticker: price}),
            'cash': self.portfolio.cash,
            'positions': self.portfolio.positions
        })

        if not self.risk_manager.check_daily_limits():
            result['rejected_reason'] = 'daily_limits_exceeded'
            return result

        if action_str.startswith('BUY'):
            quantity, actual_risk = self.risk_manager.calculate_position_size(
                ticker=ticker, price=price, stop_loss=None,
                atr=None, confidence=confidence,
                adv=10000000, sector=self.test_tickers[ticker]['sector'],
                lot_size=lot
            )

            if quantity <= 0:
                result['rejected_reason'] = 'risk_manager_rejected'
                return result

            cost = quantity * price
            commission = self._get_commission(cost)
            total_required = cost + commission
            available = self.portfolio.cash - self.portfolio.reserved_cash

            if total_required > available:
                result['rejected_reason'] = f'insufficient_cash (need {total_required:.0f}, have {available:.0f})'
                return result

            success = self.portfolio.buy(ticker, quantity, price, strategy,
                                         lot_size=lot, min_step=step)
            if success:
                self.risk_manager.update_trade_result(ticker, 'BUY', quantity, price, 0)
                result['executed'] = True
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
            commission = self._get_commission(revenue)
            success, pnl = self.portfolio.sell(ticker, quantity, price)

            if success:
                self.risk_manager.update_trade_result(ticker, 'SELL', quantity, price, pnl)
                result['executed'] = True
                result['quantity'] = quantity
                result['cost'] = revenue
                result['pnl'] = pnl
            else:
                result['rejected_reason'] = 'portfolio.sell_failed'

        return result

    def _classify_action(self, action_idx: int) -> str:
        action_str = self.action_mapping.get(str(action_idx), f'UNKNOWN_{action_idx}')
        if action_str.startswith('BUY'):
            return 'BUY'
        elif action_str.startswith('HOLD'):
            return 'HOLD'
        elif action_str.startswith('SELL'):
            return 'SELL'
        return 'UNKNOWN'

    def analyze_model_weights(self):
        print(f"\n🧠 АНАЛИЗ ВЕСОВ МОДЕЛИ:")
        for name, param in self.model.policy_net.named_parameters():
            if 'action_net' in name and param.requires_grad:
                weights = param.data.cpu().numpy()
                print(f"   {name}: shape={weights.shape}, mean={weights.mean():.4f}, "
                      f"std={weights.std():.4f}, min={weights.min():.4f}, max={weights.max():.4f}")
                if 'bias' in name and weights.shape[0] == self.model.action_dim:
                    print(f"      BIAS (склонность к каждому действию):")
                    for i, bias_val in enumerate(weights):
                        action_name = self.action_mapping.get(str(i), f'UNKNOWN_{i}')
                        category = self._classify_action(i)
                        print(f"        [{category:6s}] {action_name:15s}: bias={bias_val:+.4f}")

        total_norm = 0
        for p in self.model.policy_net.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        print(f"\n   Gradient norm: {total_norm ** 0.5:.6f}")

    def analyze_memory(self):
        print(f"\n📦 АНАЛИЗ ПАМЯТИ МОДЕЛИ:")
        print(f"   Всего опытов: {len(self.model.memory)}")
        if len(self.model.memory) == 0:
            return

        actions_in_memory = Counter()
        categories_in_memory = Counter()
        rewards_in_memory = []
        dones_in_memory = []

        for exp in self.model.memory:
            action = exp.get('action', -1)
            actions_in_memory[action] += 1
            categories_in_memory[self._classify_action(action)] += 1
            rewards_in_memory.append(float(exp.get('reward', 0)))
            dones_in_memory.append(exp.get('done', False))

        total = len(self.model.memory)
        print(f"   По категориям:")
        for cat in ['BUY', 'HOLD', 'SELL']:
            count = categories_in_memory.get(cat, 0)
            print(f"     {cat:6s}: {count:5d} ({count/total*100:5.1f}%)")

        print(f"\n   По действиям:")
        for action_idx in sorted(actions_in_memory.keys()):
            count = actions_in_memory[action_idx]
            action_name = self.action_mapping.get(str(action_idx), f'UNKNOWN_{action_idx}')
            print(f"     [{self._classify_action(action_idx):6s}] {action_name:15s}: {count:5d} ({count/total*100:5.1f}%)")

        print(f"\n   Rewards: mean={np.mean(rewards_in_memory):.4f}, "
              f"median={np.median(rewards_in_memory):.4f}, "
              f"min={np.min(rewards_in_memory):.4f}, max={np.max(rewards_in_memory):.4f}")
        print(f"   Positive: {sum(1 for r in rewards_in_memory if r > 0)}/{total}")
        print(f"   Done ratio: {sum(dones_in_memory)/total:.1%}")

    def analyze_action_choice(self, num_samples: int = 100):
        print(f"\n🎯 АНАЛИЗ ВЫБОРА ДЕЙСТВИЙ ({num_samples} сэмплов):")

        all_actions = []
        all_strategies = []
        all_values = []
        categories = Counter()
        errors = 0

        for i in range(num_samples):
            ticker = random.choice(list(self.test_tickers.keys()))
            price = self.base_prices[ticker] * random.uniform(0.98, 1.02)
            try:
                state = self._create_state(ticker, price)
            except Exception as e:
                errors += 1
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
                categories[self._classify_action(action)] += 1

                strategy_params = self.model.strategies.get(strategy, list(self.model.strategies.values())[0])
                full_state = self.model._create_strategy_state(state, strategy_params)
                all_values.append(self.model.get_state_value(full_state))
            except Exception as e:
                errors += 1

        total = len(all_actions)
        if total == 0:
            print(f"\n   ❌ Все сэмплы с ошибками ({errors}). Проверьте BERT.")
            return

        if errors:
            print(f"\n   ⚠ Успешно: {total}/{num_samples}, ошибок: {errors}")

        print(f"\n   По категориям:")
        for cat in ['BUY', 'HOLD', 'SELL']:
            count = categories.get(cat, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"     {cat:6s}: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\n   По действиям:")
        action_counts = Counter(all_actions)
        for action_idx in sorted(action_counts.keys()):
            count = action_counts[action_idx]
            pct = count / total * 100 if total > 0 else 0
            action_name = self.action_mapping.get(str(action_idx), f'UNKNOWN_{action_idx}')
            print(f"     [{self._classify_action(action_idx):6s}] {action_name:15s}: {count:4d} ({pct:5.1f}%)")

        print(f"\n   Стратегии:")
        strategy_counts = Counter(all_strategies)
        for name in sorted(self.model.strategies.keys()):
            count = strategy_counts.get(name, 0)
            pct = count / total * 100 if total > 0 else 0
            print(f"     {name:20s}: {count:4d} ({pct:5.1f}%)")

        if all_values:
            print(f"\n   State Values: mean={np.mean(all_values):.4f}, "
                  f"min={np.min(all_values):.4f}, max={np.max(all_values):.4f}")

    def run_trading_simulation(self, cycles: int = 30):
        print(f"\n💹 СИМУЛЯЦИЯ ТОРГОВЛИ ({cycles} циклов, через Risk Manager):")
        print("-" * 80)

        for cycle in range(cycles):
            for ticker in self.test_tickers:
                old = self.base_prices[ticker]
                self.base_prices[ticker] = old * (1 + random.uniform(-0.005, 0.005))
                self.technical_core.update_price_data(ticker, self.base_prices[ticker])

            for ticker in list(self.test_tickers.keys())[:5]:
                price = self.base_prices[ticker]
                try:
                    state = self._create_state(ticker, price)
                except:
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
                        state=state, ticker=ticker, price=price, market_context=market_context
                    )
                except:
                    continue

                action_str = self.action_mapping.get(str(action), 'HOLD')
                self.actions_distribution[action_str] += 1
                self.strategies_distribution[strategy] += 1

                if action_str.startswith('HOLD'):
                    continue

                result = self._try_real_trade(ticker, price, action_str, strategy, confidence)

                if result['executed']:
                    self.real_trades += 1
                    self.commission_total += self._get_commission(result['cost'])
                    if 'pnl' in result:
                        self.strategy_pnl[strategy] += result['pnl']
                        self.strategy_trades[strategy] += 1
                else:
                    self.rejected_trades += 1
                    self.rejection_reasons[result.get('rejected_reason', 'unknown')] += 1

            self.cycles += 1
            self.portfolio_values.append(self.portfolio.get_total_value(self.base_prices))

            if (cycle + 1) % 10 == 0:
                print(f"\n   Цикл {cycle+1}/{cycles}: портфель={self.portfolio_values[-1]:,.0f}₽, "
                      f"сделок={self.real_trades}, отклонено={self.rejected_trades}")

    def print_final_report(self):
        print("\n" + "=" * 80)
        print("📋 ИТОГОВЫЙ ДИАГНОСТИЧЕСКИЙ ОТЧЁТ")
        print("=" * 80)

        total_signals = sum(self.actions_distribution.values())
        buy_total = sum(1 for a in self.actions_distribution if a.startswith('BUY'))
        hold_total = sum(1 for a in self.actions_distribution if a.startswith('HOLD'))
        sell_total = sum(1 for a in self.actions_distribution if a.startswith('SELL'))

        print(f"\n💹 ТОРГОВЛЯ:")
        print(f"   Циклов: {self.cycles}, сигналов: {total_signals}")
        if total_signals > 0:
            print(f"   BUY: {buy_total} ({buy_total/total_signals*100:.1f}%)")
            print(f"   HOLD: {hold_total} ({hold_total/total_signals*100:.1f}%)")
            print(f"   SELL: {sell_total} ({sell_total/total_signals*100:.1f}%)")
        print(f"   Исполнено: {self.real_trades}, отклонено: {self.rejected_trades}")
        if self.real_trades + self.rejected_trades > 0:
            print(f"   Процент исполнения: {self.real_trades/(self.real_trades+self.rejected_trades)*100:.1f}%")

        if self.rejection_reasons:
            print(f"\n   Причины отклонений:")
            for reason, count in self.rejection_reasons.most_common(10):
                print(f"     {reason}: {count}")

        print(f"\n   Комиссий: {self.commission_total:.2f}₽")

        if self.strategy_pnl:
            print(f"\n   PnL по стратегиям:")
            for name in sorted(self.strategy_pnl.keys()):
                pnl = self.strategy_pnl[name]
                trades = self.strategy_trades[name]
                avg = pnl / trades if trades > 0 else 0
                print(f"     {name:20s}: {pnl:+.2f}₽ ({trades} сделок, сред: {avg:+.2f}₽)")

        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ДЕЙСТВИЙ:")
        for name in sorted(self.actions_distribution.keys()):
            count = self.actions_distribution[name]
            pct = count / total_signals * 100 if total_signals > 0 else 0
            print(f"   {name:15s}: {count:4d} ({pct:5.1f}%) {'█' * int(pct/2)}")

        if self.portfolio_values:
            start, end = self.portfolio_values[0], self.portfolio_values[-1]
            change = end - start
            print(f"\n📈 ПОРТФЕЛЬ: {start:,.0f}₽ → {end:,.0f}₽ ({change:+,.0f}₽ / {change/start*100:+.2f}%)")

        print(f"\n⚠ КЛЮЧЕВЫЕ МЕТРИКИ:")
        if total_signals > 0:
            hold_pct = hold_total / total_signals * 100
            print(f"   1. HOLD: {hold_pct:.1f}%")
            if hold_pct > 80:
                print(f"      🔴 Модель слишком консервативна — проверьте exploration")
            elif hold_pct < 20:
                print(f"      🔴 Модель слишком агрессивна — проверьте reward")
            else:
                print(f"      ✅ Баланс в норме")

        print(f"   2. Исполнение: {self.real_trades} сделок")
        if self.real_trades == 0 and self.rejected_trades > 0:
            top = self.rejection_reasons.most_common(1)[0]
            print(f"   3. 🔴 НЕТ СДЕЛОК. Причина: {top[0]} ({top[1]} раз)")
        elif self.real_trades == 0:
            print(f"   3. 🔴 НЕТ СДЕЛОК. Модель выбирает только HOLD.")

    def run_full_diagnostics(self, trading_cycles: int = 30, action_samples: int = 100):
        print("\n" + "=" * 80)
        print("🔬 ЗАПУСК ПОЛНОЙ ДИАГНОСТИКИ")
        print("=" * 80)

        self.analyze_model_weights()
        self.analyze_memory()
        self.analyze_action_choice(action_samples)
        self.run_trading_simulation(trading_cycles)
        self.print_final_report()

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