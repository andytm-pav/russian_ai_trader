"""
Централизованный риск-менеджер системы - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""

import json
import math
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque

from utils.logger import get_logger

logger = get_logger('RISK_MANAGER')


class RiskManager:
    """Централизованное управление рисками"""

    def __init__(self, config_path: str = "config/settings.json",
                 training_wheels_path: str = "config/training_wheels.json"):
        self.config = self._load_config(config_path)
        self.tw = self._load_training_wheels(training_wheels_path)
        self.portfolio_state = self._load_portfolio_state()

        # Потокобезопасность - критическое исправление
        self.lock = threading.RLock()

        # Метрики риска - ИСПРАВЛЕНО: отслеживаем просадку, а не только PnL
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_drawdown = 0.0  # Положительное число: максимальная просадка от максимума
        self.daily_high_watermark = self.portfolio_state.get('total_value', 0)
        self.trade_history = deque(maxlen=1000)  # Используем deque для фиксированного размера

        risk_per_trade = self.tw.get('risk_params', {}).get('risk_per_trade_percent', 3.0)
        logger.info(f"Инициализирован Risk Manager (риск на сделку: {risk_per_trade}%)")

    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                logger.debug(f"Загружен конфиг settings: {list(loaded_config.keys())}")
                return loaded_config

        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации settings: {e}")
            return {}

    def _load_training_wheels(self, training_wheels_path: str) -> Dict:
        """Загрузка учебных костылей"""
        try:
            with open(training_wheels_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки training_wheels: {e}")
            return {}

    def _load_portfolio_state(self) -> Dict:
        """Загрузка состояния портфеля"""
        try:
            with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)

                # Добавляем поля для расширенной функциональности
                if 'sector_allocation' not in state:
                    state['sector_allocation'] = {}
                if 'correlation_matrix' not in state:
                    state['correlation_matrix'] = {}

                return state

        except Exception as e:
            logger.error(f"Ошибка загрузки состояния портфеля: {e}")
            return {
                'total_value': self.config.get('initial_capital_rub', 10000),
                'cash': self.config.get('initial_capital_rub', 10000),
                'positions': {},
                'sector_allocation': {},
                'correlation_matrix': {},
                'last_update': datetime.now().isoformat()
            }

    def calculate_position_size(self,
                                ticker: str,
                                price: float,
                                stop_loss: float = None,
                                atr: float = None,
                                confidence: float = 0.5,
                                adv: float = None,
                                sector: str = None,
                                lot_size: int = 1) -> Tuple[int, float]:
        """Расчет размера позиции на основе риска"""
        with self.lock:
            try:

                # Проверка готовности технических индикаторов
                min_history_points = self.config.get('min_history_points_for_trade', 0)
                ticker_history = self.portfolio_state.get('price_history', {}).get(ticker, [])
                if len(ticker_history) < min_history_points:
                    logger.warning(
                        f"Недостаточно истории цен для {ticker}: {len(ticker_history)}/{min_history_points} точек")
                    return 0, 0.0

                # Проверка минимальной цены тикера (защита от неликвида)
                min_price = self.config.get('min_ticker_price', 1.0)
                if price < min_price:
                    logger.warning(f"Цена {ticker} ({price:.2f}₽) ниже минимальной ({min_price}₽) — сделка отклонена")
                    return 0, 0.0

                # Текущая стоимость портфеля с fallback
                portfolio_value = self.portfolio_state.get('total_value')
                if not portfolio_value or portfolio_value <= 0:
                    portfolio_value = self.config.get('initial_capital_rub', 10000)
                    logger.debug(f"Использую начальный капитал для расчёта риска: {portfolio_value}")

                # 0. ПРОВЕРКА ДНЕВНОГО ЛИМИТА УБЫТКА
                max_daily_loss_pct = self.tw.get('trade_limits', {}).get('max_daily_loss_percent', 10.0)
                if max_daily_loss_pct > 0:
                    daily_start = self.config.get('daily_start_capital', None)
                    if daily_start and daily_start > 0:
                        current_value = self.portfolio_state.get('total_value', daily_start)
                        daily_pnl = current_value - daily_start
                        if daily_pnl < 0:
                            daily_loss_pct = abs(daily_pnl) / daily_start * 100
                            if daily_loss_pct > max_daily_loss_pct:
                                logger.warning(
                                    f"Дневной убыток {daily_loss_pct:.1f}% превысил лимит {max_daily_loss_pct}%. Торговля остановлена.")
                                self.trading_enabled = False
                                return 0, 0.0

                # 1. РАСЧЕТ СТОП-ЛОССА (приоритет ATR если включен и доступен)
                use_atr = self.config.get('use_atr_for_stops', True)
                stop_loss_pct = self.tw.get('risk_params', {}).get('stop_loss_percent', 6.0)
                min_risk_by_stop = price * (stop_loss_pct / 100)

                if use_atr and atr is not None and atr > 0:
                    # Используем ATR-based стоп
                    atr_multiplier = self.config.get('atr_multiplier', 2.0)
                    risk_per_share = atr * atr_multiplier
                    stop_loss_price = price - risk_per_share

                    # Защита от нулевого ATR: если ATR-стоп меньше процентного — используем процентный
                    if risk_per_share < min_risk_by_stop:
                        risk_per_share = min_risk_by_stop
                        stop_loss_price = price - risk_per_share
                        logger.debug(f"ATR-стоп {ticker} слишком мал (ATR={atr:.2f}), "
                                     f"использую процентный стоп {stop_loss_pct}%: {risk_per_share:.2f}₽")
                    else:
                        logger.debug(f"Использую ATR-стоп для {ticker}: ATR={atr:.2f}, стоп={stop_loss_price:.2f}")
                elif stop_loss is not None and stop_loss > 0:
                    # Используем переданный стоп
                    risk_per_share = abs(price - stop_loss)
                    stop_loss_price = stop_loss
                    # Защита: если переданный стоп меньше процентного — используем процентный
                    if risk_per_share < min_risk_by_stop:
                        risk_per_share = min_risk_by_stop
                        stop_loss_price = price - risk_per_share
                else:
                    # Дефолтный фиксированный процент из конфига
                    risk_per_share = min_risk_by_stop
                    stop_loss_price = price - risk_per_share
                    logger.debug(f"Нет ATR/стопа для {ticker}, использую фиксированный {stop_loss_pct}%")

                # 2. ПРОВЕРКА МИНИМАЛЬНОГО РИСКА (из конфига)
                min_risk_pct = self.tw.get('risk_params', {}).get('min_risk_per_share_percent', 0.1) / 100
                min_risk_per_share = price * min_risk_pct

                if risk_per_share < min_risk_per_share:
                    logger.warning(
                        f"Слишком малый риск на акцию для {ticker}: {risk_per_share:.4f} < {min_risk_per_share:.4f}")
                    return 0, 0

                # 3. МАКСИМАЛЬНЫЙ РИСК НА СДЕЛКУ (из конфига)
                max_risk_pct = self.tw.get('risk_params', {}).get('risk_per_trade_percent', 3.0) / 100
                max_risk_rub = portfolio_value * max_risk_pct

                # 4. КОРРЕКТИРОВКА НА УВЕРЕННОСТЬ (исправленная логика из конфига)
                conf_weight_min = self.config.get('confidence_weight_min', 0.5)
                conf_weight_max = self.config.get('confidence_weight_max', 1.0)
                confidence_adjustment = conf_weight_min + (confidence * (conf_weight_max - conf_weight_min))
                adjusted_risk_rub = max_risk_rub * confidence_adjustment

                # 5. ЛИКВИДНОСТНЫЕ ОГРАНИЧЕНИЯ (из конфига)
                quantity_by_risk = int(adjusted_risk_rub / risk_per_share)

                if adv is not None and adv > 0:
                    max_adv_pct = self.tw.get('risk_params', {}).get('max_adv_percent', 5.0) / 100
                    max_quantity_by_adv = int(adv * max_adv_pct)
                    quantity_by_risk = min(quantity_by_risk, max_quantity_by_adv)
                    logger.debug(
                        f"Лимит ликвидности для {ticker}: {max_quantity_by_adv} акций ({max_adv_pct * 100:.1f}% от ADV)")

                # 6. ПРОВЕРКА МИНИМАЛЬНОЙ СДЕЛКИ (из конфига)
                min_trade_value = self.tw.get('trade_limits', {}).get('min_cash_per_trade', 1000)
                trade_value = quantity_by_risk * price

                if trade_value < min_trade_value:
                    # Проверяем, хватит ли кэша на минимальную сделку
                    available_cash = self.portfolio_state.get('cash', 0) - self.portfolio_state.get('reserved_cash', 0)
                    if available_cash < min_trade_value:
                        logger.warning(
                            f"Недостаточно кэша для минимальной сделки {ticker}: {available_cash:.0f}₽ < {min_trade_value:.0f}₽")
                        return 0, 0.0
                    quantity = max(1, math.ceil(min_trade_value / price))
                    trade_value = quantity * price

                    # Пересчитываем фактический риск
                    actual_risk_rub = quantity * risk_per_share

                    # Лимит превышения риска (из конфига)
                    risk_multiplier_limit = self.tw.get('risk_params', {}).get('risk_multiplier_on_high_confidence', 1.2)
                    if actual_risk_rub > max_risk_rub * risk_multiplier_limit:
                        logger.warning(f"Минимальная сделка {ticker} превышает лимит риска: "
                                       f"{actual_risk_rub:.0f}₽ > {max_risk_rub * risk_multiplier_limit:.0f}₽")
                        return 0, 0
                else:
                    quantity = quantity_by_risk
                    actual_risk_rub = quantity * risk_per_share


                    # ✅ ДОБАВЛЯЕМ: Корректировка на лотность
                    if lot_size > 1:
                        # Округляем до ближайшего кратного lot_size
                        quantity_by_lot = (quantity_by_risk // lot_size) * lot_size

                        # Если после округления получился 0, берем минимум 1 лот
                        if quantity_by_lot == 0 and quantity_by_risk >= lot_size / 2:
                            quantity_by_lot = lot_size
                        elif quantity_by_lot == 0:
                            logger.debug(f"Рассчитанное количество {quantity_by_risk} меньше половины лота {lot_size}")
                            return 0, 0

                        quantity = quantity_by_lot
                        logger.debug(f"Корректировка по лотности: {quantity_by_risk} → {quantity} (лот: {lot_size})")
                    else:
                        quantity = quantity_by_risk

                    # ✅ ДОБАВЛЯЕМ: Проверка минимального количества
                    if quantity < lot_size:
                        logger.debug(f"Количество {quantity} меньше размера лота {lot_size}")
                        # Пробуем взять 1 лот если это не превышает риск
                        if lot_size * price <= self.portfolio_value * max_risk_pct * 1.5:  # +50% допуск
                            quantity = lot_size
                            logger.debug(f"Установлено минимальное количество: {quantity} (1 лот)")
                        else:
                            logger.warning(f"1 лот {ticker} превышает лимит риска")
                            return 0, 0


                # 7. ПРОВЕРКА ДИВЕРСИФИКАЦИИ С УЧЕТОМ СЕКТОРА
                if not self.check_diversification(ticker, quantity * price, sector):
                    logger.warning(f"Сделка {ticker} нарушает диверсификацию")
                    return 0, 0

                # 8. ФИНАЛЬНАЯ ПРОВЕРКА
                if quantity < 1:
                    return 0, 0

                risk_pct = (actual_risk_rub / portfolio_value * 100) if portfolio_value > 0 else 0

                logger.info(f"Расчет позиции {ticker}: {quantity} акций, риск {actual_risk_rub:.0f}₽ "
                            f"({risk_pct:.1f}% от портфеля), стоп: {stop_loss_price:.2f}")

                return quantity, actual_risk_rub

            except Exception as e:
                logger.error(f"Ошибка расчета размера позиции для {ticker}: {e}")
                return 0, 0

    def check_diversification(self, ticker: str, new_position_value: float, sector: str = None) -> bool:
        """Проверка диверсификации портфеля - УЛУЧШЕННАЯ"""
        with self.lock:
            try:
                portfolio_value = self.portfolio_state.get('total_value',
                                                           self.config.get('initial_capital_rub', 10000))

                if portfolio_value <= 0:
                    return False

                # 1. ПРОВЕРКА МАКСИМАЛЬНОГО КОЛИЧЕСТВА ПОЗИЦИЙ (из конфига)
                max_positions = self.config.get('max_positions', 10)
                current_positions = len(self.portfolio_state.get('positions', {}))

                # Получаем текущие позиции
                positions = self.portfolio_state.get('positions', {})
                is_existing_position = ticker in positions

                # Проверка лимита позиций
                if current_positions >= max_positions:
                    # Если это НОВАЯ позиция - блокируем
                    if not is_existing_position:
                        logger.warning(f"Достигнут лимит позиций: {current_positions}/{max_positions}")
                        return False
                    # Если это СУЩЕСТВУЮЩАЯ позиция - разрешаем докупку
                    else:
                        logger.debug(f"Докупка существующей позиции {ticker} (всего позиций: {current_positions})")
                        # НЕ возвращаем False, продолжаем проверку дальше!

                # 2. ПРОВЕРКА МАКСИМАЛЬНОГО ВЕСА ПОЗИЦИИ (из конфига)
                max_weight_pct = self.tw.get('risk_params', {}).get('max_position_weight_percent', 30)
                max_weight = max_weight_pct / 100

                # Защита от расчётных ошибок: вес > 100% означает ошибку в quantity или price
                if new_position_value > portfolio_value:
                    logger.error(f"Ошибка расчёта позиции {ticker}: стоимость {new_position_value:.0f}₽ "
                                 f"превышает портфель {portfolio_value:.0f}₽")
                    return False

                if ticker in positions:
                    # Уже есть позиция
                    current_position = positions[ticker]
                    current_value = current_position.get('current_value',
                                                         current_position.get('qty', 0) *
                                                         current_position.get('avg_price', 0))
                    new_total_value = current_value + new_position_value
                    position_weight = new_total_value / portfolio_value

                    if position_weight > max_weight:
                        logger.warning(f"Превышен максимальный вес позиции {ticker}: "
                                       f"{position_weight * 100:.1f}% > {max_weight_pct}%")
                        return False
                else:
                    # Новая позиция
                    position_weight = new_position_value / portfolio_value

                    if position_weight > max_weight:
                        logger.warning(f"Новая позиция {ticker} превышает лимит веса: "
                                       f"{position_weight * 100:.1f}% > {max_weight_pct}%")
                        return False

                # 3. ПРОВЕРКА ЭКСПОЗИЦИИ НА СЕКТОР (из конфига)
                if sector:
                    sector_allocation = self.portfolio_state.get('sector_allocation', {})
                    current_sector_exposure = sector_allocation.get(sector, 0)
                    new_sector_exposure = current_sector_exposure + new_position_value
                    sector_weight = new_sector_exposure / portfolio_value

                    # Максимальный вес сектора (из конфига)
                    max_sector_weight_pct = self.config.get('max_sector_weight_percent', 40.0)
                    max_sector_weight = max_sector_weight_pct / 100

                    if sector_weight > max_sector_weight:
                        logger.warning(f"Превышена экспозиция на сектор {sector}: "
                                       f"{sector_weight * 100:.1f}% > {max_sector_weight_pct}%")
                        return False

                # 4. ПРОВЕРКА КОРРЕЛЯЦИИ (порог из конфига)
                correlation_threshold = self.config.get('correlation_threshold', 0.7)

                if ticker in self.portfolio_state.get('correlation_matrix', {}):
                    correlations = self.portfolio_state['correlation_matrix'][ticker]

                    # Проверяем корреляцию с существующими позициями
                    for existing_ticker in positions:
                        if existing_ticker in correlations:
                            corr = correlations[existing_ticker]
                            if abs(corr) > correlation_threshold:
                                logger.warning(f"Высокая корреляция {ticker} с {existing_ticker}: {corr:.2f}")
                                # Не блокируем по умолчанию, только предупреждаем
                                # return False  # Раскомментировать для строгой проверки

                return True

            except Exception as e:
                logger.error(f"Ошибка проверки диверсификации: {e}")
                return False  # При ошибке блокируем сделку

    def check_daily_limits(self) -> bool:
        """Проверка дневных лимитов - ИСПРАВЛЕННАЯ"""
        with self.lock:
            try:
                portfolio_value = self.portfolio_state.get('total_value',
                                                           self.config.get('initial_capital_rub', 10000))

                if portfolio_value <= 0:
                    return True

                # 1. ЛИМИТ УБЫТКОВ НА ОСНОВЕ ПРОСАДКИ (ИСПРАВЛЕНО!)
                daily_limit_pct = self.config.get('daily_loss_limit_percent', 5) / 100
                daily_limit_rub = portfolio_value * daily_limit_pct

                # Сравниваем максимальную просадку с лимитом
                if self.max_daily_drawdown >= daily_limit_rub:  # max_daily_drawdown - положительное число!
                    logger.critical(f"Достигнут дневной лимит просадки: {self.max_daily_drawdown:.0f}₽ "
                                    f"(лимит: {daily_limit_rub:.0f}₽)")
                    return False

                # 2. ЛИМИТ СДЕЛОК В ДЕНЬ (из конфига)
                max_daily_trades = self.tw.get('trade_limits', {}).get('max_daily_trades', 50)
                if self.daily_trades >= max_daily_trades:
                    logger.warning(f"Достигнут лимит сделок за день: {self.daily_trades}/{max_daily_trades}")
                    return False

                # 3. ЛИМИТ НА ПОСЛЕДОВАТЕЛЬНЫЕ УБЫТКИ (из конфига)
                consecutive_trades_window = self.config.get('consecutive_trades_window', 5)
                if len(self.trade_history) >= consecutive_trades_window:
                    recent_trades = list(self.trade_history)[-consecutive_trades_window:]
                    losing_trades = [t for t in recent_trades if t.get('pnl', 0) < 0]

                    consecutive_loss_limit = self.tw.get('trade_limits', {}).get('max_consecutive_losses', 8)
                    if len(losing_trades) >= consecutive_loss_limit:
                        logger.warning(
                            f"Слишком много убыточных сделок подряд: {len(losing_trades)}/{consecutive_trades_window}")
                        return False

                return True

            except Exception as e:
                logger.error(f"Ошибка проверки дневных лимитов: {e}")
                return False  # При ошибке блокируем!

    def update_trade_result(self,
                            ticker: str,
                            action: str,
                            quantity: int,
                            price: float,
                            pnl: float,
                            atr: float = None):
        """Обновление результатов сделки - РАСШИРЕННАЯ"""
        with self.lock:
            try:
                # Обновляем дневной PnL
                self.daily_pnl += pnl
                self.daily_trades += 1

                # Обновляем максимум портфеля за день
                portfolio_value = self.portfolio_state.get('total_value',
                                                           self.config.get('initial_capital_rub', 10000))

                if portfolio_value > self.daily_high_watermark:
                    self.daily_high_watermark = portfolio_value

                # ИСПРАВЛЕНО: Рассчитываем текущую просадку (положительное число!)
                current_drawdown = self.daily_high_watermark - portfolio_value
                if current_drawdown > self.max_daily_drawdown:
                    self.max_daily_drawdown = current_drawdown

                # Записываем в историю
                trade_record = {
                    'timestamp': datetime.now().isoformat(),
                    'ticker': ticker,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'pnl': pnl,
                    'atr': atr,
                    'pnl_percent': (pnl / (quantity * price) * 100) if quantity * price > 0 else 0,
                    'daily_pnl': self.daily_pnl,
                    'daily_drawdown': self.max_daily_drawdown
                }

                self.trade_history.append(trade_record)

                logger.info(f"Trade update: {ticker} {action} {quantity} @ {price:.2f}, "
                            f"PnL: {pnl:+.0f}₽ ({trade_record['pnl_percent']:+.1f}%), "
                            f"Daily: {self.daily_pnl:+.0f}₽, Drawdown: {self.max_daily_drawdown:.0f}₽")

            except Exception as e:
                logger.error(f"Ошибка обновления результатов сделки: {e}")

    def get_risk_metrics(self) -> Dict:
        """Получение метрик риска - УЛУЧШЕННЫЕ"""
        with self.lock:
            try:
                portfolio_value = self.portfolio_state.get('total_value',
                                                           self.config.get('initial_capital_rub', 10000))

                if portfolio_value <= 0:
                    return {'error': 'Invalid portfolio value'}

                metrics = {
                    'portfolio_value': portfolio_value,
                    'cash': self.portfolio_state.get('cash', 0),
                    'daily_pnl': self.daily_pnl,
                    'daily_trades': self.daily_trades,
                    'max_daily_drawdown': self.max_daily_drawdown,  # ИСПРАВЛЕНО: просадка, а не loss
                    'daily_high_watermark': self.daily_high_watermark,
                    'current_drawdown_pct': (self.max_daily_drawdown / portfolio_value * 100)
                    if portfolio_value > 0 else 0,
                    'risk_per_trade_pct': self.config.get('risk_per_trade_percent', 3),
                    'positions_count': len(self.portfolio_state.get('positions', {})),
                    'max_positions': self.config.get('max_positions', 10),
                    'can_trade': self.check_daily_limits(),
                    'last_update': datetime.now().isoformat()
                }

                # Рассчитываем концентрацию портфеля
                positions = self.portfolio_state.get('positions', {})
                if positions:
                    position_values = []
                    for pos in positions.values():
                        value = pos.get('current_value',
                                        pos.get('qty', 0) * pos.get('avg_price', 0))
                        position_values.append(value)

                    total_positions_value = sum(position_values)

                    if total_positions_value > 0 and portfolio_value > 0:
                        # Доля портфеля в позициях
                        metrics['portfolio_in_positions_pct'] = (total_positions_value / portfolio_value * 100)

                        # Максимальный вес позиции
                        max_weight = max(position_values) / portfolio_value * 100
                        metrics['max_position_weight_pct'] = max_weight

                        # Коэффициент Джини для концентрации (упрощенный)
                        if len(position_values) > 1:
                            sorted_values = sorted(position_values)
                            n = len(sorted_values)
                            index = np.arange(1, n + 1)
                            gini_numerator = np.sum(index * sorted_values)
                            gini_denominator = n * np.sum(sorted_values)
                            metrics['gini_coefficient'] = (2 * gini_numerator / gini_denominator) - (n + 1) / n

                        # Win rate по последним сделкам
                        if len(self.trade_history) >= 20:
                            recent_trades = list(self.trade_history)[-20:]
                            winning_trades = [t for t in recent_trades if t.get('pnl', 0) > 0]
                            metrics['recent_win_rate'] = len(winning_trades) / len(recent_trades)
                            metrics['avg_win'] = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                            losing_trades = [t for t in recent_trades if t.get('pnl', 0) < 0]
                            metrics['avg_loss'] = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

                # Добавляем конфигурационные параметры для прозрачности
                metrics['config'] = {
                    'daily_loss_limit_pct': self.tw.get('trade_limits', {}).get('max_daily_loss_percent', 10.0),
                    'max_position_weight_pct': self.tw.get('risk_params', {}).get('max_position_weight_percent', 30),
                    'use_atr_for_stops': self.config.get('use_atr_for_stops', True),
                    'atr_multiplier': self.config.get('atr_multiplier', 2.0),
                    'max_adv_percent': self.tw.get('risk_params', {}).get('max_adv_percent', 5.0),
                    'max_sector_weight_pct': self.config.get('max_sector_weight_percent', 40.0),
                    'correlation_threshold': self.config.get('correlation_threshold', 0.7)
                }

                return metrics

            except Exception as e:
                logger.error(f"Ошибка расчета метрик риска: {e}")
                return {'error': str(e)}

    def reset_daily_metrics(self):
        """Сброс дневных метрик (вызывается в начале дня)"""
        with self.lock:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.max_daily_drawdown = 0.0
            self.daily_high_watermark = self.portfolio_state.get('total_value', 0)
            logger.info("Дневные метрики риска сброшены")

    def update_portfolio_state(self, new_state: Dict):
        """Обновление состояния портфеля с потокобезопасностью"""
        with self.lock:
            try:
                # Обновляем только необходимые поля для обратной совместимости
                update_fields = {
                    'total_value': new_state.get('total_value'),
                    'cash': new_state.get('cash'),
                    'positions': new_state.get('positions'),
                    'last_update': datetime.now().isoformat()
                }

                # Удаляем None значения
                update_fields = {k: v for k, v in update_fields.items() if v is not None}
                self.portfolio_state.update(update_fields)

                # Сохраняем состояние
                self._save_portfolio_state()

                logger.debug("Состояние портфеля обновлено")

            except Exception as e:
                logger.error(f"Ошибка обновления состояния портфеля: {e}")

    def _save_portfolio_state(self):
        """Сохранение состояния портфеля"""
        try:
            with open('data/portfolio_state.json', 'w', encoding='utf-8') as f:
                json.dump(self.portfolio_state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния портфеля: {e}")

    # Новые методы для расширенной функциональности (обратная совместимость)

    def update_correlation_matrix(self, ticker: str, correlations: Dict[str, float]):
        """Обновление матрицы корреляций для тикера"""
        with self.lock:
            try:
                if 'correlation_matrix' not in self.portfolio_state:
                    self.portfolio_state['correlation_matrix'] = {}

                self.portfolio_state['correlation_matrix'][ticker] = correlations
                self._save_portfolio_state()

                logger.debug(f"Обновлены корреляции для {ticker}: {len(correlations)} записей")

            except Exception as e:
                logger.error(f"Ошибка обновления корреляций для {ticker}: {e}")

    def update_sector_allocation(self, sector: str, value: float):
        """Обновление распределения по секторам"""
        with self.lock:
            try:
                if 'sector_allocation' not in self.portfolio_state:
                    self.portfolio_state['sector_allocation'] = {}

                self.portfolio_state['sector_allocation'][sector] = value
                self._save_portfolio_state()

            except Exception as e:
                logger.error(f"Ошибка обновления распределения по секторам: {e}")

    def get_trade_statistics(self, lookback_trades: int = 20) -> Dict:
        """Получение статистики сделок за период"""
        with self.lock:
            try:
                if not self.trade_history:
                    return {'error': 'No trade history'}

                # Безопасное копирование под локом
                recent_trades = list(self.trade_history)

                if len(recent_trades) > lookback_trades:
                    recent_trades = recent_trades[-lookback_trades:]

                if not recent_trades:
                    return {'error': 'No recent trades'}

                # Базовая статистика
                pnls = [t['pnl'] for t in recent_trades]
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0

                # Упрощенный Sharpe ratio (для внутридневной торговли)
                avg_return = np.mean(pnls) if pnls else 0
                std_return = np.std(pnls) if len(pnls) > 1 else 0

                # 252 торговых дня в году (стандарт)
                trading_days_per_year = 252
                sharpe_ratio = (avg_return / std_return * np.sqrt(trading_days_per_year)) \
                    if std_return > 0 else 0

                return {
                    'total_trades': len(recent_trades),
                    'win_rate': win_rate,
                    'total_pnl': sum(pnls),
                    'avg_pnl': avg_return,
                    'std_pnl': std_return,
                    'max_win': max(pnls) if pnls else 0,
                    'max_loss': min(pnls) if pnls else 0,
                    'sharpe_ratio': sharpe_ratio,
                    'profit_factor': abs(sum(p for p in pnls if p > 0) / sum(p for p in pnls if p < 0))
                    if sum(p for p in pnls if p < 0) < 0 else float('inf')
                }

            except Exception as e:
                logger.error(f"Ошибка расчета статистики сделок: {e}")
                return {'error': str(e)}