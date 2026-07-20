"""
Управление портфелем и позициями
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from utils.logger import get_logger
from utils.lot_validator import LotValidator

logger = get_logger("PORTFOLIO")


class PortfolioManager:
    """Класс для управления торговым портфелем"""

    def __init__(self, portfolio_file: str = "data/portfolio_state.json"):
        self.portfolio_file = portfolio_file
        self.positions = {}
        self.cash = 0.0
        self.reserved_cash = 0.0
        self.pending_commissions = []
        self.trade_history = []
        self.initial_capital = 0.0
        self.strategy_positions = defaultdict(list)

        self.daily_trades = []
        self.daily_reset_time = "19:00"
        self.preserve_buy_time = True

        # ========== ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ (БУДУТ ПЕРЕЗАПИСАНЫ ИЗ КОНФИГА) ==========
        self.settings = self._load_settings()
        self.training_wheels = self._load_training_wheels()
        self.commission_rate = 0.003  # 0.3% - тариф Т-Банка "Инвестор"
        self.min_commission = 0.01  # минимальная комиссия 0.01₽
        self.rounding = 2  # округление до 2 знаков
        self.max_positions = 10
        self.max_trades_per_hour = 10
        self.daily_commission_limit = 100.0

        # Поля для комиссионных признаков
        self.commission_reserve = 0.0
        self.commission_spent_today = 0.0
        self.total_commission = 0.0
        self.total_trades = 0
        self.total_pnl = 0.0

        # 🆕 v16 Фаза 1.2: Anti-overtrading
        # Загружаем из конфига, дефолты — консервативные
        anti_overtrading_cfg = self.settings.get('anti_overtrading', {})
        self.max_trades_per_hour = anti_overtrading_cfg.get('max_trades_per_hour', 5)
        self.min_seconds_between_trades = anti_overtrading_cfg.get('min_seconds_between_trades', 600)
        self._last_trade_timestamp = 0.0  # timestamp последней сделки

        # 🆕 v16 Фаза 1.5: Limit-ордера (проскальзывание)
        execution_cfg = self.settings.get('execution', {})
        self.buy_price_markup = execution_cfg.get('buy_price_markup', 0.001)   # +0.1% к цене
        self.sell_price_markdown = execution_cfg.get('sell_price_markdown', 0.001)  # -0.1%
        self.log_slippage = execution_cfg.get('log_slippage', True)
        self.slippage_warning_threshold = execution_cfg.get('slippage_warning_threshold', 0.003)

        # 🆕 v16 Фаза 1.4: Stop-loss на позициях (читается в smart_broker)
        risk_cfg = self.settings.get('risk_management', {})
        self.max_position_loss_pct = risk_cfg.get('max_position_loss_pct', 5.0)
        self.no_averaging_below_loss_pct = risk_cfg.get('no_averaging_below_loss_pct', 3.0)

        # ========== ЗАГРУЗКА КОНФИГА (ПЕРЕЗАПИСЫВАЕТ ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ) ==========
        self._load_commission_config()

        # Загрузка состояния портфеля
        self.load_portfolio()

        logger.info(f"Инициализирован PortfolioManager. Комиссия: {self.commission_rate * 100}%, "
                    f"мин: {self.min_commission}₽, округление: {self.rounding} знаков. "
                    f"Капитал: {self.cash:,.0f}₽")

    # ⚠️ ИСПРАВЛЕНО: НОВЫЙ МЕТОД для загрузки комиссионных настроек




    def _load_settings(self) -> Dict:
        """Загрузка settings.json"""
        try:
            with open("config/settings.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_commission_config(self):
        """Загрузка настроек комиссий из конфигурационного файла"""
        try:
            with open("config/settings.json", "r", encoding="utf-8") as f:
                settings = json.load(f)

            # Ищем секцию commission (новый формат)
            commission_config = settings.get('commission', {})

            if commission_config:
                # Новый формат: commission.equity_rate_decimal
                self.commission_rate = commission_config.get('equity_rate_decimal', 0.003)
                self.min_commission = commission_config.get('min_commission_rub', 0.01)
                self.rounding = commission_config.get('rounding_decimals', 2)
            else:
                # Старый формат (обратная совместимость): moex_schedule.commission.rate_decimal
                moex_config = settings.get('moex_schedule', {})
                commission = moex_config.get('commission', {})
                self.commission_rate = commission.get('rate_decimal', 0.003)
                self.min_commission = 0.01  # Значение по умолчанию для старого формата
                self.rounding = 2  # Значение по умолчанию для старого формата

            # Загружаем остальные настройки
            self.max_positions = settings.get("max_positions", 10)
            self.max_trades_per_hour = settings.get("max_trades_per_hour", 10)
            self.daily_commission_limit = settings.get("daily_commission_limit", 100.0)

            # Настройки portfolio_manager
            pm_config = settings.get("portfolio_manager", {})
            self.preserve_buy_time = pm_config.get("preserve_buy_time_on_partial_sell", True)

            # Время сброса комиссий
            moex_config = settings.get('moex_schedule', {})
            commission = moex_config.get('commission', {})
            self.daily_reset_time = commission.get("charge_time", "19:00")

            logger.info(f"Загружены настройки комиссий: ставка={self.commission_rate * 100}%, "
                        f"мин={self.min_commission}₽, округление={self.rounding} знаков")

        except Exception as e:
            logger.warning(f"Не удалось загрузить настройки комиссий: {e}, "
                           f"использую значения по умолчанию")
            # Значения по умолчанию уже установлены в __init__

    def _load_training_wheels(self) -> Dict:
        """Загрузка учебных костылей"""
        try:
            with open("config/training_wheels.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def _calculate_commission(self, amount: float) -> float:
        """
        Расчёт комиссии с округлением и минимальным значением

        Args:
            amount: Сумма сделки (cost для покупки, revenue для продажи)

        Returns:
            Рассчитанная комиссия в рублях
        """
        commission = amount * self.commission_rate
        commission = round(commission, self.rounding)

        # Минимальная комиссия 0.01₽ (согласно тарифу Т-Банка)
        if commission < self.min_commission and commission > 0:
            commission = self.min_commission

        return commission


    def load_portfolio(self):
        """Загрузка состояния портфеля из файла"""
        try:
            with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.positions = data.get('positions', {})
            self.cash = data.get('cash', 0.0)
            self.initial_capital = data.get('initial_capital', self.cash)
            self.trade_history = data.get('trade_history', [])

            # Загрузка комиссионных полей
            self.total_commission = data.get('total_commission', 0.0)
            self.total_trades = data.get('total_trades', 0)
            self.total_pnl = data.get('total_pnl', 0.0)
            self.commission_spent_today = data.get('commission_spent_today', 0.0)

            # Загрузка стратегий
            self.strategy_positions = defaultdict(list)
            for ticker, pos in self.positions.items():
                strategy = pos.get('strategy')
                if strategy:
                    self.strategy_positions[strategy].append(ticker)

            # Конвертируем строковые даты обратно в timestamps
            for ticker, pos in self.positions.items():
                if 'buy_time' in pos and isinstance(pos['buy_time'], str):
                    try:
                        pos['buy_time'] = float(pos['buy_time'])
                    except:
                        pos['buy_time'] = time.time()

            logger.info(f"Загружен портфель: {len(self.positions)} позиций, "
                        f"{self.cash:,.0f}₽ кэша, {len(self.strategy_positions)} стратегий")

        except FileNotFoundError:
            logger.warning(f"Файл портфеля не найден: {self.portfolio_file}")
            self._create_initial_portfolio()
        except Exception as e:
            logger.error(f"Ошибка загрузки портфеля: {e}")
            self._create_initial_portfolio()

    def _create_initial_portfolio(self):
        """Создание начального состояния портфеля"""
        self.positions = {}
        self.cash = 10000.0
        self.initial_capital = self.cash
        self.trade_history = []
        self.strategy_positions = defaultdict(list)
        self.save_portfolio()
        logger.info(f"Создан начальный портфель с капиталом {self.cash:,.0f}₽")

    def _load_lot_size_directory(self) -> Dict[str, Dict]:
        """Загрузка справочника лотностей из tickers.json"""
        try:
            with open('config/tickers.json', 'r', encoding='utf-8') as f:
                tickers_config = json.load(f)

            directory = {}
            for item in tickers_config.get('watchlist', []):
                directory[item['ticker']] = {
                    'lot_size': item.get('lot_size', 1),
                    'min_step': item.get('min_step', 0.01)
                }

            logger.info(f"Загружен справочник лотностей: {len(directory)} тикеров")
            return directory

        except Exception as e:
            logger.error(f"Ошибка загрузки справочника лотностей: {e}")
            return {}

    def save_portfolio(self):
        """Сохранение состояния портфеля в файл"""
        try:
            strategy_positions_dict = dict(self.strategy_positions)

            data = {
                'positions': self.positions,
                'cash': self.cash,
                'initial_capital': self.initial_capital,
                'trade_history': self.trade_history[-100:],
                'strategy_positions': strategy_positions_dict,
                'total_value': self.get_total_value({}),
                'last_update': datetime.now().isoformat(),
                'stats': self.get_portfolio_stats(),
                'total_commission': self.total_commission,
                'total_trades': self.total_trades,
                'total_pnl': self.total_pnl,
                'commission_spent_today': self.commission_spent_today
            }

            with open(self.portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Портфель сохранен: {len(self.positions)} позиций, "
                         f"{self.cash:,.0f}₽ кэша")

            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения портфеля: {e}")
            return False

    def _update_commission_stats(self, commission_amount: float):
        """Обновление статистики комиссий"""
        self.commission_spent_today += commission_amount
        self.total_commission += commission_amount
        self.commission_reserve = self.commission_spent_today + (self.commission_rate * 1000)

    def buy(self, ticker: str, quantity: int, price: float, strategy: str = None,
            time_horizon: str = 'balanced', **kwargs) -> bool:
        """Покупка акций с указанием стратегии"""
        try:
            # Получение информации о бумаге
            lot_size = kwargs.get('lot_size', 1)
            min_step = kwargs.get('min_step', 0.01)

            # 🆕 v16 Фаза 1.2: Anti-overtrading — проверка интервала между сделками
            now_ts = time.time()
            seconds_since_last = now_ts - self._last_trade_timestamp
            if self._last_trade_timestamp > 0 and seconds_since_last < self.min_seconds_between_trades:
                logger.warning(
                    f"Anti-overtrading: прошло {seconds_since_last:.0f}с < {self.min_seconds_between_trades}с "
                    f"с последней сделки — BUY {ticker} отменён"
                )
                return False

            # 🆕 v16 Фаза 1.5: Limit-ордер — корректируем цену с markup
            # (эмулируем лимит-ордер чуть выше рынка для гарантии исполнения)
            original_price = price
            limit_price = price * (1.0 + self.buy_price_markup)
            # Округляем до min_step
            if min_step > 0:
                limit_price = round(limit_price / min_step) * min_step
            if self.log_slippage and abs(limit_price - original_price) / original_price > 1e-6:
                logger.info(f"Limit BUY: market={original_price:.4f} → limit={limit_price:.4f} "
                           f"(markup {self.buy_price_markup*100:.2f}%)")
            price = limit_price

            # 🆕 v16 Фаза 1.1: LotValidator — корректная проверка лотности (работает для float)
            adjusted_qty, was_adjusted = LotValidator.validate_and_adjust_quantity(int(quantity), lot_size)
            if was_adjusted:
                logger.info(f"Лотность: qty {quantity} → {adjusted_qty} (lot={lot_size})")
                quantity = adjusted_qty
            if quantity == 0:
                logger.error(f"BUY {ticker}: после проверки лотности qty=0 (lot={lot_size}) — отмена")
                return False

            # Проверка кратности цены
            if min_step > 0 and price % min_step != 0:
                price = round(price / min_step) * min_step
                logger.info(f"Цена скорректирована до {price:.4f}")

            # (старая проверка лотности удалена — заменена на LotValidator выше)

            # Проверка входных данных
            if quantity <= 0 or price <= 0:
                logger.error(f"Некорректные данные для покупки {ticker}: qty={quantity}, price={price}")
                return False

            if quantity < lot_size:
                logger.error(f"Количество меньше размера лота {lot_size}: {quantity}")
                return False

            # ========== ПРОВЕРКА ЛИМИТА ПОЗИЦИЙ ==========
            if ticker not in self.positions and len(self.positions) >= self.max_positions:
                logger.warning(f"Достигнут лимит позиций: {len(self.positions)}/{self.max_positions}")
                return False

            # ========== ПРОВЕРКА ЛИМИТОВ ГОРИЗОНТА ==========
            horizon_positions = sum(
                1 for p in self.positions.values()
                if p.get('time_horizon') == time_horizon
            )
            max_per_horizon = self.settings.get('max_positions_per_horizon', {}).get(
                time_horizon,
                self.settings.get('max_positions_per_horizon', {}).get('week', 10)
            )

            if ticker not in self.positions and horizon_positions >= max_per_horizon:
                logger.warning(f"Лимит позиций горизонта {time_horizon}: "
                               f"{horizon_positions}/{max_per_horizon}")
                return False

            # ========== ПРОВЕРКА ЛИМИТА СДЕЛОК В ЧАС ==========
            now = time.time()
            lookback = self.settings.get('trades_lookback_seconds', 3600)
            trades_last_hour = 0
            for t in self.trade_history:
                ts = t.get('timestamp')
                if ts:
                    if isinstance(ts, str):
                        try:
                            from datetime import datetime as dt
                            ts_float = dt.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                        except:
                            continue
                    else:
                        ts_float = float(ts)
                    if now - ts_float < lookback:
                        trades_last_hour += 1

            if trades_last_hour >= self.max_trades_per_hour:
                logger.warning(f"Лимит сделок в час: {trades_last_hour}/{self.max_trades_per_hour}")
                return False

            # ========== РАСЧЁТ СТОИМОСТИ И КОМИССИИ ==========
            cost = quantity * price
            commission_buy = self._calculate_commission(cost)

            # ========== ПРОВЕРКА ЛИМИТА ДНЕВНЫХ КОМИССИЙ ==========
            if self.commission_spent_today + commission_buy > self.daily_commission_limit:
                logger.warning(f"Лимит дневных комиссий: "
                               f"{self.commission_spent_today:.2f} + {commission_buy:.2f} > "
                               f"{self.daily_commission_limit:.2f}")
                return False

            total_required = cost + commission_buy

            # ========== ПРОВЕРКА ДОСТУПНОСТИ СРЕДСТВ ==========
            available_cash = self.cash - self.reserved_cash
            if total_required > available_cash:
                logger.error(f"Недостаточно средств: нужно {total_required:,.0f}₽, "
                             f"доступно {available_cash:,.0f}₽ (cash: {self.cash:,.0f}₽, "
                             f"резерв: {self.reserved_cash:,.0f}₽)")
                return False

            # ========== ВЫПОЛНЕНИЕ ПОКУПКИ ==========
            if ticker in self.positions:
                # Усреднение позиции
                pos = self.positions[ticker]
                total_qty = pos['qty'] + quantity
                total_cost = (pos['qty'] * pos['avg_price']) + cost

                pos['qty'] = total_qty
                pos['avg_price'] = total_cost / total_qty
                pos['buy_time'] = time.time()

                old_commission = pos.get('commission_buy', 0.0)
                pos['commission_buy'] = old_commission + commission_buy

                if strategy and 'strategy' not in pos:
                    pos['strategy'] = strategy

                if time_horizon and 'time_horizon' not in pos:
                    pos['time_horizon'] = time_horizon

                if kwargs:
                    for key, value in kwargs.items():
                        pos[key] = value

                logger.debug(f"Усреднена позиция {ticker}: +{quantity} @ {price:.2f}")
            else:
                # Новая позиция
                self.positions[ticker] = {
                    'qty': quantity,
                    'avg_price': price,
                    'buy_time': time.time(),
                    'total_cost': cost,
                    'commission_buy': commission_buy,
                    'time_horizon': time_horizon
                }

                if strategy:
                    self.positions[ticker]['strategy'] = strategy
                    self.strategy_positions[strategy].append(ticker)

                if kwargs:
                    for key, value in kwargs.items():
                        self.positions[ticker][key] = value

            # ========== СПИСАНИЕ СРЕДСТВ И РЕЗЕРВИРОВАНИЕ КОМИССИИ ==========
            self.cash -= cost
            self.reserved_cash += commission_buy
            self._update_commission_stats(commission_buy)

            # ========== ЗАПИСЬ В ИСТОРИЮ ==========
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'BUY',
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'cost': cost,
                'commission': commission_buy,
                'cash_after': self.cash,
                'reserved_after': self.reserved_cash,
                'position_after': self.positions[ticker].copy(),
                'strategy': strategy
            }

            if kwargs:
                trade_record['params'] = kwargs

            self.trade_history.append(trade_record)

            self.daily_trades.append({
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'value': cost,
                'commission_reserved': commission_buy,
                'strategy': strategy
            })

            self.save_portfolio()

            # 🆕 v16 Фаза 1.2: Anti-overtrading — обновляем timestamp последней сделки
            self._last_trade_timestamp = time.time()

            logger.info(f"КУПЛЕНО: {ticker} {quantity} @ {price:.2f} = {cost:,.0f}₽, "
                        f"комиссия: {commission_buy:.2f}₽, "
                        f"кэш: {self.cash:,.0f}₽, резерв: {self.reserved_cash:,.0f}₽")

            return True

        except Exception as e:
            logger.error(f"Ошибка покупки {ticker}: {e}")
            return False

    def sell(self, ticker: str, quantity: int, price: float) -> Tuple[bool, float]:
        """Продажа акций. Возвращает (успех, pnl)"""
        try:
            # Получение информации о бумаге из позиции
            if ticker not in self.positions:
                logger.error(f"Нет позиции для продажи: {ticker}")
                return False, 0.0

            pos = self.positions[ticker]
            lot_size = pos.get('lot_size', 1)
            strategy = pos.get('strategy')

            # 🆕 v16 Фаза 1.5: Limit-ордер для SELL — markdown для гарантии исполнения
            original_price = price
            min_step = pos.get('min_step', 0.01)
            limit_price = price * (1.0 - self.sell_price_markdown)
            if min_step > 0:
                limit_price = round(limit_price / min_step) * min_step
            if self.log_slippage and abs(limit_price - original_price) / original_price > 1e-6:
                logger.info(f"Limit SELL: market={original_price:.4f} → limit={limit_price:.4f} "
                           f"(markdown {self.sell_price_markdown*100:.2f}%)")
            price = limit_price

            # 🆕 v16 Фаза 1.1: LotValidator для SELL (работает с float qty)
            # Если остаток позиции < 1 лота — продаём всё (закрытие позиции)
            current_qty = pos['qty']
            if lot_size > 1 and current_qty < lot_size:
                # Мусорная позиция (< 1 лота) — закрываем полностью
                logger.info(f"SELL {ticker}: остаток {current_qty} < 1 лота ({lot_size}) — закрываем позицию")
                quantity = current_qty
            else:
                adjusted_qty, was_adjusted = LotValidator.validate_and_adjust_quantity(int(quantity), lot_size)
                if was_adjusted:
                    logger.info(f"Лотность SELL: qty {quantity} → {adjusted_qty} (lot={lot_size})")
                    quantity = adjusted_qty
                if quantity == 0 and current_qty >= lot_size:
                    # Если qty_to_sell < 0.5 лота, но позиция больше — пропускаем продажу
                    logger.warning(f"SELL {ticker}: qty после лотности = 0, пропуск "
                                  f"(requested={quantity}, lot={lot_size})")
                    return False, 0.0

            # Проверка входных данных
            if quantity <= 0 or price <= 0:
                logger.error(f"Некорректные данные для продажи {ticker}")
                return False, 0.0

            if quantity > pos['qty']:
                logger.error(f"Недостаточно акций: нужно {quantity}, есть {pos['qty']}")
                return False, 0.0

            # 🆕 v16 Фаза 1.3: Min hold time из settings (не training_wheels)
            risk_cfg = self.settings.get('risk_management', {})
            min_hold_hours = risk_cfg.get('minimum_hold_hours_before_sell', 0)
            if min_hold_hours > 0:
                buy_time = pos.get('buy_time', 0)
                hold_seconds = time.time() - buy_time
                if hold_seconds < min_hold_hours * 3600:
                    logger.warning(
                        f"SELL {ticker}: hold {hold_seconds/60:.1f}мин < {min_hold_hours}ч — отмена "
                        f"(phase exit слишком рано)"
                    )
                    return False, 0.0

            # Проверка минимального времени удержания (учебный костыль) — оставлено для обратной совместимости
            min_hold = self.training_wheels.get('trade_limits', {}).get('min_hold_time_seconds', 0)
            if min_hold > 0:
                buy_time = pos.get('buy_time', 0)
                if time.time() - buy_time < min_hold:
                    logger.debug(f"Сделка отклонена: позиция {ticker} удерживается менее {min_hold}с")
                    return False, 0.0

            # ========== РАСЧЁТ КОМИССИЙ ==========
            revenue = quantity * price
            commission_sell = self._calculate_commission(revenue)

            # Комиссия покупки (пропорционально продаваемой части)
            total_commission_buy = pos.get('commission_buy', 0.0)
            total_qty = pos['qty']
            commission_buy_for_part = total_commission_buy * (quantity / total_qty)

            # ========== РАСЧЁТ PnL ==========
            entry_cost = quantity * pos['avg_price']
            pnl = revenue - entry_cost - commission_buy_for_part - commission_sell

            # ========== ОБНОВЛЕНИЕ ПОЗИЦИИ ==========
            if quantity == pos['qty']:
                # Полностью закрываем позицию
                del self.positions[ticker]
                if strategy:
                    self.remove_strategy_from_tracker(ticker, strategy)
                logger.debug(f"Закрыта позиция {ticker}")
            else:
                # Частичная продажа
                pos['qty'] -= quantity

                # Уменьшаем накопленную комиссию покупки
                remaining_commission_buy = total_commission_buy - commission_buy_for_part
                if remaining_commission_buy > 0:
                    pos['commission_buy'] = remaining_commission_buy
                else:
                    pos.pop('commission_buy', None)

                logger.debug(f"Частичная продажа {ticker}: -{quantity}, осталось {pos['qty']}")

            # ========== СПИСАНИЕ КОМИССИЙ И ОБНОВЛЕНИЕ СРЕДСТВ ==========
            # Зачисляем выручку от продажи
            self.cash += revenue

            # Списание комиссии продажи
            self.cash -= commission_sell

            # Освобождаем зарезервированную комиссию покупки и продажи
            total_commission_to_release = commission_buy_for_part + commission_sell

            if self.reserved_cash >= total_commission_to_release:
                self.reserved_cash -= total_commission_to_release
            else:
                # Защита от отрицательного резерва (на всякий случай)
                logger.warning(f"Недостаточно резерва для {ticker}: "
                               f"нужно {total_commission_to_release:.2f}₽, "
                               f"есть {self.reserved_cash:.2f}₽")
                total_commission_to_release = self.reserved_cash
                self.reserved_cash = 0

            # ========== ОБНОВЛЕНИЕ СТАТИСТИКИ ==========
            self._update_commission_stats(commission_sell)
            self.total_trades += 1
            self.total_pnl += pnl

            # ========== ЗАПИСЬ В ИСТОРИЮ ==========
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'SELL',
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'revenue': revenue,
                'commission_buy': commission_buy_for_part,
                'commission_sell': commission_sell,
                'commission_total': commission_buy_for_part + commission_sell,
                'pnl': pnl,
                'pnl_percent': (pnl / entry_cost * 100) if entry_cost > 0 else 0,
                'cash_after': self.cash,
                'reserved_after': self.reserved_cash,
                'position_after': self.positions.get(ticker, {}).copy(),
                'strategy': strategy
            }

            self.trade_history.append(trade_record)
            self.daily_trades.append({
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'action': 'SELL',
                'quantity': quantity,
                'price': price,
                'value': revenue,
                'commission': commission_sell,
                'pnl': pnl,
                'strategy': strategy
            })

            self.save_portfolio()

            # 🆕 v16 Фаза 1.2: Anti-overtrading — обновляем timestamp последней сделки
            self._last_trade_timestamp = time.time()

            logger.info(f"ПРОДАНО: {ticker} {quantity} @ {price:.2f} = {revenue:,.0f}₽, "
                        f"комиссия покупки: {commission_buy_for_part:.2f}₽, "
                        f"комиссия продажи: {commission_sell:.2f}₽, "
                        f"PnL: {pnl:+,.0f}₽ ({trade_record['pnl_percent']:+.1f}%), "
                        f"кэш: {self.cash:,.0f}₽, резерв: {self.reserved_cash:,.0f}₽")

            return True, pnl

        except Exception as e:
            logger.error(f"Ошибка продажи {ticker}: {e}")
            return False, 0.0

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Расчет общей стоимости портфеля"""
        try:
            positions_value = 0.0
            for ticker, pos in self.positions.items():
                current_price = current_prices.get(ticker, pos.get('avg_price', 0))
                positions_value += pos['qty'] * current_price
            return self.cash + positions_value
        except Exception as e:
            logger.error(f"Ошибка расчета стоимости портфеля: {e}")
            return self.cash

    def get_portfolio_stats(self, current_prices: Dict[str, float] = None) -> Dict:
        """Получение статистики портфеля"""
        if current_prices is None:
            current_prices = {}

        stats = {
            'cash': self.cash,
            'positions_count': len(self.positions),
            'total_trades': len(self.trade_history),
            'initial_capital': self.initial_capital,
            'current_capital': self.cash + sum(
                p.get('qty', 0) * p.get('avg_price', 0)
                for p in self.positions.values()
            ),
            'last_update': datetime.now().isoformat(),
            'strategies_count': len(self.strategy_positions)
        }

        if current_prices:
            total_value = self.get_total_value(current_prices)
            stats['total_value_with_current_prices'] = total_value
            stats['total_pnl_with_current_prices'] = total_value - self.initial_capital

        return stats

    def calculate_projected_weight(self,
                                   ticker: str,
                                   quantity: int,
                                   price: float,
                                   current_prices: Dict[str, float],
                                   lot_size: int = 1) -> float:
        """Расчет проектируемого веса позиции в портфеле"""
        try:
            # Текущая стоимость портфеля
            current_value = self.get_total_value(current_prices)

            if current_value <= 0:
                return 0.0

            # ✅ ДОБАВЛЯЕМ: Корректировка количества по лотности
            if lot_size > 1 and quantity % lot_size != 0:
                # Округляем до ближайшего кратного
                quantity = (quantity // lot_size) * lot_size
                if quantity == 0:
                    quantity = lot_size  # Минимум 1 лот


            # Стоимость новой позиции
            new_position_value = quantity * price

            # Текущая стоимость этой позиции (если уже есть)
            current_position_value = 0
            if ticker in self.positions:
                current_qty = self.positions[ticker]['qty']
                current_position_value = current_qty * current_prices.get(ticker, price)

            # Общая стоимость позиции после покупки
            total_position_value = current_position_value + new_position_value

            # Проектируемый вес
            projected_weight = total_position_value / current_value

            return projected_weight

        except Exception as e:
            logger.error(f"Ошибка расчета проектируемого веса для {ticker}: {e}")
            return 0.0

    def calculate_position_weight_with_sentiment(self,
                                                 ticker: str,
                                                 quantity: int,
                                                 price: float,
                                                 current_prices: Dict[str, float],
                                                 ticker_sentiment: float,
                                                 market_sentiment: float) -> Dict:
        """Расчет веса позиции с учетом тональности"""
        try:
            # Базовый расчет веса
            base_weight = self.calculate_projected_weight(ticker, quantity, price, current_prices)

            # Корректировка на сентимент
            sentiment_adjustment = 0.0

            # Положительный сентимент тикера увеличивает допустимый вес
            if ticker_sentiment > 0.3:
                sentiment_adjustment += 0.1  # +10% к весу
            elif ticker_sentiment < -0.3:
                sentiment_adjustment -= 0.1  # -10% к весу

            # Рыночный сентимент влияет на общую агрессивность
            if market_sentiment > 0.3:
                sentiment_adjustment += 0.05  # Более агрессивный рынок
            elif market_sentiment < -0.3:
                sentiment_adjustment -= 0.05  # Более консервативный рынок

            final_weight = base_weight * (1 + sentiment_adjustment)

            return {
                'base_weight': base_weight,
                'sentiment_adjustment': sentiment_adjustment,
                'final_weight': final_weight,
                'ticker_sentiment': ticker_sentiment,
                'market_sentiment': market_sentiment
            }

        except Exception as e:
            logger.error(f"Ошибка расчета веса с сентиментом: {e}")
            return {'base_weight': 0, 'final_weight': 0}

    def get_position_details(self, ticker: str) -> Optional[Dict]:
        """Получение деталей по позиции"""
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]

        details = {
            'ticker': ticker,
            'quantity': pos['qty'],
            'avg_price': pos['avg_price'],
            'buy_time': datetime.fromtimestamp(pos['buy_time']).isoformat() if 'buy_time' in pos else None,
            'total_cost': pos['qty'] * pos['avg_price'],
            'days_held': (time.time() - pos.get('buy_time', time.time())) / 86400 if 'buy_time' in pos else 0
        }

        return details

    def get_positions_by_strategy(self, strategy: str) -> List[Dict]:
        """Получение позиций по конкретной стратегии"""
        positions = []

        for ticker, pos in self.positions.items():
            if pos.get('strategy') == strategy:
                positions.append({
                    'ticker': ticker,
                    'quantity': pos['qty'],
                    'avg_price': pos['avg_price'],
                    'buy_time': datetime.fromtimestamp(pos['buy_time']).isoformat() if 'buy_time' in pos else None,
                    'strategy': pos.get('strategy', 'unknown'),
                    'stop_loss': pos.get('stop_loss'),
                    'take_profit': pos.get('take_profit')
                })

        return positions

    def get_strategy_stats(self, current_prices: Dict[str, float]) -> Dict[str, Dict]:
        """Статистика по стратегиям"""
        strategy_stats = {}

        for strategy, tickers in self.strategy_positions.items():
            total_value = 0
            total_pnl = 0
            positions_count = 0

            for ticker in tickers:
                if ticker in self.positions:
                    pos = self.positions[ticker]
                    current_price = current_prices.get(ticker, pos['avg_price'])
                    position_value = pos['qty'] * current_price
                    pnl = (current_price - pos['avg_price']) * pos['qty']

                    total_value += position_value
                    total_pnl += pnl
                    positions_count += 1

            if positions_count > 0:
                strategy_stats[strategy] = {
                    'positions_count': positions_count,
                    'total_value': total_value,
                    'total_pnl': total_pnl,
                    'avg_pnl_per_position': total_pnl / positions_count if positions_count > 0 else 0
                }

        return strategy_stats

    def get_all_positions_details(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Получение деталей по всем позициям"""
        positions_details = []

        for ticker, pos in self.positions.items():
            current_price = current_prices.get(ticker, pos.get('avg_price', 0))

            position_value = pos['qty'] * current_price
            pnl = (current_price - pos['avg_price']) * pos['qty']
            pnl_percent = (pnl / (pos['qty'] * pos['avg_price']) * 100) if pos['qty'] * pos['avg_price'] > 0 else 0

            details = {
                'ticker': ticker,
                'quantity': pos['qty'],
                'avg_price': pos['avg_price'],
                'current_price': current_price,
                'position_value': position_value,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'buy_time': datetime.fromtimestamp(pos['buy_time']).isoformat() if 'buy_time' in pos else None,
                'days_held': (time.time() - pos.get('buy_time', time.time())) / 86400 if 'buy_time' in pos else 0
            }

            positions_details.append(details)

        # Сортировка по стоимости позиции
        positions_details.sort(key=lambda x: x['position_value'], reverse=True)

        return positions_details

    def get_trade_history_summary(self, limit: int = 20) -> List[Dict]:
        """Получение истории сделок"""
        return self.trade_history[-limit:] if self.trade_history else []

    def reset_portfolio(self, new_cash: float = 10000.0):
        """Сброс портфеля к начальному состоянию"""
        confirmation = input(f"Вы уверены, что хотите сбросить портфель? "
                             f"Все позиции будут закрыты. (yes/no): ")

        if confirmation.lower() == 'yes':
            old_positions = len(self.positions)

            # Продаем все позиции по текущим ценам (в симуляции)
            self.positions = {}
            self.cash = new_cash
            self.initial_capital = new_cash
            self.trade_history = []

            self.save_portfolio()

            logger.warning(f"Портфель сброшен: закрыто {old_positions} позиций, "
                           f"новый капитал: {new_cash:,.0f}₽")

            return True
        else:
            logger.info("Сброс портфеля отменен")
            return False

    def add_cash(self, amount: float):
        """Пополнение счета"""
        if amount <= 0:
            logger.error(f"Некорректная сумма пополнения: {amount}")
            return False

        self.cash += amount
        self.save_portfolio()

        logger.info(f"Счет пополнен на {amount:,.0f}₽, новый баланс: {self.cash:,.0f}₽")
        return True

    def withdraw_cash(self, amount: float):
        """Снятие средств"""
        if amount <= 0:
            logger.error(f"Некорректная сумма снятия: {amount}")
            return False

        if amount > self.cash:
            logger.error(f"Недостаточно средств для снятия: нужно {amount:,.0f}₽, "
                         f"есть {self.cash:,.0f}₽")
            return False

        self.cash -= amount
        self.save_portfolio()

        logger.info(f"Снято {amount:,.0f}₽, новый баланс: {self.cash:,.0f}₽")
        return True

    def get_available_cash_for_trade(self, risk_percent: float = 0.1) -> float:
        """Расчет доступного кэша для торговли с учетом риска"""
        try:
            # Общая стоимость портфеля (оценочная)
            total_value = self.get_total_value({})

            # Доступный кэш с учетом риска
            available = min(self.cash, total_value * risk_percent)

            return max(0, available)

        except Exception as e:
            logger.error(f"Ошибка расчета доступного кэша: {e}")
            return self.cash * 0.1  # 10% от кэша по умолчанию

    def export_portfolio_report(self, export_path: str = "data/portfolio_report.json"):
        """Экспорт отчета по портфелю"""
        try:
            report = {
                'portfolio_summary': self.get_portfolio_stats(),
                'positions': self.get_all_positions_details({}),
                'trade_history': self.get_trade_history_summary(50),
                'export_time': datetime.now().isoformat()
            }

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Отчет по портфелю экспортирован в {export_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка экспорта отчета: {e}")
            return False

    def reset_daily_trades(self):
        """Сброс дневной статистики (вызывать в 19:00)"""
        self.daily_trades = []
        self.commission_spent_today = 0.0
        self.commission_reserve = 0.0
        logger.debug(f"Дневная статистика сделок и комиссий сброшена (время: {self.daily_reset_time})")

    def get_commission_stats(self) -> Dict[str, float]:
        """Получение статистики комиссий для модели"""
        return {
            'commission_reserve': self.commission_reserve,
            'commission_spent_today': self.commission_spent_today,
            'total_commission': self.total_commission,
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
            'daily_commission_limit': self.daily_commission_limit,
            'max_trades_per_hour': self.max_trades_per_hour,
            'max_positions': self.max_positions
        }


    def remove_strategy_from_tracker(self, ticker: str, strategy: str):
        """Удаление позиции из трекера стратегий"""
        if strategy in self.strategy_positions and ticker in self.strategy_positions[strategy]:
            self.strategy_positions[strategy].remove(ticker)
            if not self.strategy_positions[strategy]:
                del self.strategy_positions[strategy]
            logger.debug(f"Удален {ticker} из трекера стратегии {strategy}")