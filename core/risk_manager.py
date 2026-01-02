"""
Централизованный риск-менеджер системы
"""

import json
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

from utils.logger import setup_logger

logger = setup_logger("RISK_MANAGER")


class RiskManager:
    """Централизованное управление рисками"""

    def __init__(self, config_path: str = "config/settings.json"):
        self.config = self._load_config(config_path)
        self.portfolio_state = self._load_portfolio_state()
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_loss = 0.0
        self.trade_history = []

        logger.info(f"Инициализирован Risk Manager (риск на сделку: {self.config['risk_per_trade_percent']}%)")

    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return {
                'risk_per_trade_percent': 1.5,
                'daily_loss_limit_percent': 5,
                'max_positions': 5,
                'initial_capital_rub': 10000
            }

    def _load_portfolio_state(self) -> Dict:
        """Загрузка состояния портфеля"""
        try:
            with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {
                'total_value': self.config.get('initial_capital_rub', 10000),
                'cash': self.config.get('initial_capital_rub', 10000),
                'positions': {},
                'last_update': datetime.now().isoformat()
            }

    def calculate_position_size(self,
                                ticker: str,
                                price: float,
                                stop_loss: float,
                                confidence: float = 0.5) -> Tuple[int, float]:
        """
        Расчет размера позиции на основе риска

        Возвращает: (количество, риск в рублях)
        """
        try:
            # Текущая стоимость портфеля
            portfolio_value = self.portfolio_state.get('total_value',
                                                       self.config.get('initial_capital_rub', 10000))

            # Максимальный риск на сделку (в рублях)
            max_risk_rub = portfolio_value * (self.config['risk_per_trade_percent'] / 100)

            # Корректировка риска на основе уверенности
            confidence_adjustment = min(confidence * 2, 1.0)  # 0.5 уверенность = 1.0, 1.0 уверенность = 2.0
            adjusted_risk_rub = max_risk_rub * confidence_adjustment

            # Риск на одну акцию (в рублях)
            risk_per_share = abs(price - stop_loss)

            if risk_per_share <= 0:
                logger.warning(f"Нулевой риск на акцию для {ticker}")
                return 0, 0

            # Расчет количества
            quantity = int(adjusted_risk_rub / risk_per_share)

            # Минимальная проверка
            if quantity < 1:
                return 0, 0

            # Проверка на минимальную сделку
            min_trade_value = self.config.get('min_cash_per_trade', 1000)
            trade_value = quantity * price

            if trade_value < min_trade_value:
                # Пытаемся увеличить до минимального
                quantity = math.ceil(min_trade_value / price)
                trade_value = quantity * price

                # Пересчитываем риск
                actual_risk_rub = quantity * risk_per_share

                if actual_risk_rub > max_risk_rub * 1.5:  # Не более 150% от макс риска
                    logger.warning(f"Сделка {ticker} превышает лимит риска")
                    return 0, 0

            # Финальная проверка диверсификации
            if not self.check_diversification(ticker, trade_value):
                logger.warning(f"Сделка {ticker} нарушает диверсификацию")
                return 0, 0

            actual_risk = quantity * risk_per_share
            logger.info(f"Расчет позиции {ticker}: {quantity} акций, риск {actual_risk:.0f}₽ "
                        f"({actual_risk / portfolio_value * 100:.1f}% от портфеля)")

            return quantity, actual_risk

        except Exception as e:
            logger.error(f"Ошибка расчета размера позиции для {ticker}: {e}")
            return 0, 0

    def check_diversification(self, ticker: str, new_position_value: float) -> bool:
        """Проверка диверсификации портфеля"""
        try:
            max_positions = self.config.get('max_positions', 5)
            current_positions = len(self.portfolio_state.get('positions', {}))

            # Проверка максимального количества позиций
            if ticker not in self.portfolio_state.get('positions', {}) and current_positions >= max_positions:
                logger.warning(f"Достигнут лимит позиций: {current_positions}/{max_positions}")
                return False

            # Проверка максимального веса позиции
            portfolio_value = self.portfolio_state.get('total_value',
                                                       self.config.get('initial_capital_rub', 10000))
            max_weight = self.config.get('max_position_weight_percent', 20) / 100

            if ticker in self.portfolio_state.get('positions', {}):
                # Уже есть позиция, проверяем общий вес
                current_position = self.portfolio_state['positions'][ticker]
                current_value = current_position.get('current_value', 0)
                new_total_value = current_value + new_position_value

                if new_total_value / portfolio_value > max_weight:
                    logger.warning(f"Превышен максимальный вес позиции {ticker}: "
                                   f"{new_total_value / portfolio_value * 100:.1f}% > {max_weight * 100}%")
                    return False
            else:
                # Новая позиция
                if new_position_value / portfolio_value > max_weight:
                    logger.warning(f"Новая позиция {ticker} превышает лимит веса: "
                                   f"{new_position_value / portfolio_value * 100:.1f}% > {max_weight * 100}%")
                    return False

            return True

        except Exception as e:
            logger.error(f"Ошибка проверки диверсификации: {e}")
            return False

    def check_daily_limits(self) -> bool:
        """Проверка дневных лимитов"""
        try:
            portfolio_value = self.portfolio_state.get('total_value',
                                                       self.config.get('initial_capital_rub', 10000))
            daily_limit = portfolio_value * (self.config.get('daily_loss_limit_percent', 5) / 100)

            if self.daily_pnl <= -daily_limit:
                logger.critical(f"Достигнут дневной лимит убытков: {self.daily_pnl:.0f}₽ "
                                f"(лимит: {-daily_limit:.0f}₽)")
                return False

            # Проверка максимального количества сделок в день (опционально)
            max_daily_trades = self.config.get('max_daily_trades', 20)
            if max_daily_trades and self.daily_trades >= max_daily_trades:
                logger.warning(f"Достигнут лимит сделок за день: {self.daily_trades}/{max_daily_trades}")
                return False

            return True

        except Exception as e:
            logger.error(f"Ошибка проверки дневных лимитов: {e}")
            return True

    def update_trade_result(self,
                            ticker: str,
                            action: str,
                            quantity: int,
                            price: float,
                            pnl: float):
        """Обновление результатов сделки"""
        try:
            # Обновляем дневной PnL
            self.daily_pnl += pnl
            self.daily_trades += 1

            # Записываем в историю
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'action': action,
                'quantity': quantity,
                'price': price,
                'pnl': pnl,
                'daily_pnl': self.daily_pnl
            }

            self.trade_history.append(trade_record)

            # Ограничиваем размер истории
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]

            logger.info(f"Trade update: {ticker} {action} {quantity} @ {price:.2f}, "
                        f"PnL: {pnl:+.0f}₽, Daily: {self.daily_pnl:+.0f}₽")

            # Проверяем, не установлен ли новый максимальный убыток
            if pnl < self.max_daily_loss:
                self.max_daily_loss = pnl

        except Exception as e:
            logger.error(f"Ошибка обновления результатов сделки: {e}")

    def get_risk_metrics(self) -> Dict:
        """Получение метрик риска"""
        portfolio_value = self.portfolio_state.get('total_value',
                                                   self.config.get('initial_capital_rub', 10000))

        metrics = {
            'portfolio_value': portfolio_value,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'max_daily_loss': self.config.get('daily_loss_limit_percent', 5),
            'current_daily_loss_pct': (self.daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0,
            'risk_per_trade': self.config.get('risk_per_trade_percent', 1.5),
            'positions_count': len(self.portfolio_state.get('positions', {})),
            'max_positions': self.config.get('max_positions', 5),
            'can_trade': self.check_daily_limits()
        }

        # Рассчитываем концентрацию портфеля
        positions = self.portfolio_state.get('positions', {})
        if positions:
            position_values = [p.get('current_value', 0) for p in positions.values()]
            total_positions_value = sum(position_values)

            if total_positions_value > 0:
                # Индекс Херфиндаля-Хиршмана (HHI) для концентрации
                hhi = sum((v / total_positions_value * 100) ** 2 for v in position_values)
                metrics['concentration_hhi'] = hhi

                # Максимальный вес позиции
                max_weight = max(position_values) / portfolio_value * 100 if portfolio_value > 0 else 0
                metrics['max_position_weight'] = max_weight

        return metrics

    def reset_daily_metrics(self):
        """Сброс дневных метрик (вызывается в начале дня)"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_loss = 0.0
        logger.info("Дневные метрики риска сброшены")

    def update_portfolio_state(self, new_state: Dict):
        """Обновление состояния портфеля"""
        self.portfolio_state = new_state

        # Сохраняем состояние
        try:
            with open('data/portfolio_state.json', 'w', encoding='utf-8') as f:
                json.dump(new_state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния портфеля: {e}")