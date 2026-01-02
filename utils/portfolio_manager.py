"""
Управление портфелем и позициями
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from utils.logger import setup_logger

logger = setup_logger("PORTFOLIO")


class PortfolioManager:
    """Класс для управления торговым портфелем"""

    def __init__(self, portfolio_file: str = "data/portfolio_state.json"):
        self.portfolio_file = portfolio_file
        self.positions = {}  # {ticker: {qty, avg_price, current_value, buy_time}}
        self.cash = 0.0
        self.trade_history = []
        self.initial_capital = 0.0

        # Загрузка состояния
        self.load_portfolio()

        logger.info(f"Инициализирован PortfolioManager. Капитал: {self.cash:,.0f}₽")

    def load_portfolio(self):
        """Загрузка состояния портфеля из файла"""
        try:
            with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.positions = data.get('positions', {})
            self.cash = data.get('cash', 0.0)
            self.initial_capital = data.get('initial_capital', self.cash)
            self.trade_history = data.get('trade_history', [])

            # Конвертируем строковые даты обратно в timestamps
            for ticker, pos in self.positions.items():
                if 'buy_time' in pos and isinstance(pos['buy_time'], str):
                    try:
                        # Пробуем преобразовать строку во float (timestamp)
                        pos['buy_time'] = float(pos['buy_time'])
                    except:
                        # Если не получается, устанавливаем текущее время
                        pos['buy_time'] = time.time()

            logger.info(f"Загружен портфель: {len(self.positions)} позиций, "
                        f"{self.cash:,.0f}₽ кэша")

        except FileNotFoundError:
            logger.warning(f"Файл портфеля не найден: {self.portfolio_file}")
            self._create_initial_portfolio()
        except Exception as e:
            logger.error(f"Ошибка загрузки портфеля: {e}")
            self._create_initial_portfolio()

    def _create_initial_portfolio(self):
        """Создание начального состояния портфеля"""
        self.positions = {}
        self.cash = 10000.0  # Начальный капитал по умолчанию
        self.initial_capital = self.cash
        self.trade_history = []

        self.save_portfolio()
        logger.info(f"Создан начальный портфель с капиталом {self.cash:,.0f}₽")

    def save_portfolio(self):
        """Сохранение состояния портфеля в файл"""
        try:
            # Подготовка данных для сохранения
            data = {
                'positions': self.positions,
                'cash': self.cash,
                'initial_capital': self.initial_capital,
                'trade_history': self.trade_history[-100:],  # Сохраняем последние 100 сделок
                'total_value': self.get_total_value({}),
                'last_update': datetime.now().isoformat(),
                'stats': self.get_portfolio_stats()
            }

            with open(self.portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Портфель сохранен: {len(self.positions)} позиций, "
                         f"{self.cash:,.0f}₽ кэша")

            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения портфеля: {e}")
            return False

    def buy(self, ticker: str, quantity: int, price: float) -> bool:
        """Покупка акций"""
        try:
            # Проверка входных данных
            if quantity <= 0 or price <= 0:
                logger.error(f"Некорректные данные для покупки {ticker}: qty={quantity}, price={price}")
                return False

            # Расчет стоимости покупки
            cost = quantity * price

            # Проверка доступности средств
            if cost > self.cash:
                logger.error(f"Недостаточно средств для покупки {ticker}: нужно {cost:,.0f}₽, есть {self.cash:,.0f}₽")
                return False

            # Выполнение покупки
            if ticker in self.positions:
                # Уже есть позиция - усредняем
                pos = self.positions[ticker]
                total_qty = pos['qty'] + quantity
                total_cost = (pos['qty'] * pos['avg_price']) + cost

                pos['qty'] = total_qty
                pos['avg_price'] = total_cost / total_qty
                pos['buy_time'] = time.time()  # Обновляем время покупки

                logger.debug(f"Усреднена позиция {ticker}: +{quantity} @ {price:.2f}, "
                             f"итого {total_qty} @ {pos['avg_price']:.2f}")
            else:
                # Новая позиция
                self.positions[ticker] = {
                    'qty': quantity,
                    'avg_price': price,
                    'buy_time': time.time(),
                    'total_cost': cost
                }

            # Списание средств
            self.cash -= cost

            # Запись в историю
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'BUY',
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'cost': cost,
                'cash_after': self.cash,
                'position_after': self.positions[ticker].copy()
            }

            self.trade_history.append(trade_record)

            # Сохранение
            self.save_portfolio()

            logger.info(f"КУПЛЕНО: {ticker} {quantity} @ {price:.2f} = {cost:,.0f}₽, "
                        f"остаток кэша: {self.cash:,.0f}₽")

            return True

        except Exception as e:
            logger.error(f"Ошибка покупки {ticker}: {e}")
            return False

    def sell(self, ticker: str, quantity: int, price: float) -> bool:
        """Продажа акций"""
        try:
            # Проверка входных данных
            if quantity <= 0 or price <= 0:
                logger.error(f"Некорректные данные для продажи {ticker}: qty={quantity}, price={price}")
                return False

            # Проверка наличия позиции
            if ticker not in self.positions:
                logger.error(f"Нет позиции для продажи: {ticker}")
                return False

            pos = self.positions[ticker]

            # Проверка количества
            if quantity > pos['qty']:
                logger.error(f"Недостаточно акций для продажи {ticker}: "
                             f"нужно {quantity}, есть {pos['qty']}")
                return False

            # Расчет выручки
            revenue = quantity * price

            # Расчет PnL
            entry_cost = quantity * pos['avg_price']
            pnl = revenue - entry_cost
            pnl_percent = (pnl / entry_cost * 100) if entry_cost > 0 else 0

            # Выполнение продажи
            if quantity == pos['qty']:
                # Продажа всей позиции
                del self.positions[ticker]
                logger.debug(f"Закрыта позиция {ticker}")
            else:
                # Частичная продажа
                pos['qty'] -= quantity
                pos['buy_time'] = time.time()  # Обновляем время покупки для оставшихся
                logger.debug(f"Частичная продажа {ticker}: -{quantity}, осталось {pos['qty']}")

            # Зачисление средств
            self.cash += revenue

            # Запись в историю
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'SELL',
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'revenue': revenue,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'cash_after': self.cash,
                'position_after': self.positions.get(ticker, {}).copy()
            }

            self.trade_history.append(trade_record)

            # Сохранение
            self.save_portfolio()

            logger.info(f"ПРОДАНО: {ticker} {quantity} @ {price:.2f} = {revenue:,.0f}₽, "
                        f"PnL: {pnl:+,.0f}₽ ({pnl_percent:+.1f}%), "
                        f"кэш: {self.cash:,.0f}₽")

            return True

        except Exception as e:
            logger.error(f"Ошибка продажи {ticker}: {e}")
            return False

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Расчет общей стоимости портфеля"""
        try:
            # Стоимость позиций
            positions_value = 0.0

            for ticker, pos in self.positions.items():
                current_price = current_prices.get(ticker, pos.get('avg_price', 0))
                positions_value += pos['qty'] * current_price

            # Общая стоимость
            total_value = self.cash + positions_value

            return total_value

        except Exception as e:
            logger.error(f"Ошибка расчета стоимости портфеля: {e}")
            return self.cash

    def get_portfolio_stats(self) -> Dict:
        """Получение статистики портфеля"""
        stats = {
            'cash': self.cash,
            'positions_count': len(self.positions),
            'total_trades': len(self.trade_history),
            'initial_capital': self.initial_capital,
            'current_capital': self.cash + sum(
                p.get('qty', 0) * p.get('avg_price', 0)
                for p in self.positions.values()
            ),
            'last_update': datetime.now().isoformat()
        }

        # Расчет PnL
        if self.trade_history:
            # Только завершенные сделки (SELL)
            sell_trades = [t for t in self.trade_history if t['action'] == 'SELL']
            if sell_trades:
                total_pnl = sum(t.get('pnl', 0) for t in sell_trades)
                stats['total_pnl'] = total_pnl
                stats['total_pnl_percent'] = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0

        # Анализ позиций
        if self.positions:
            stats['largest_position'] = max(
                self.positions.items(),
                key=lambda x: x[1]['qty'] * x[1].get('avg_price', 0)
            )[0] if self.positions else None

        return stats

    def calculate_projected_weight(self,
                                   ticker: str,
                                   quantity: int,
                                   price: float,
                                   current_prices: Dict[str, float]) -> float:
        """Расчет проектируемого веса позиции в портфеле"""
        try:
            # Текущая стоимость портфеля
            current_value = self.get_total_value(current_prices)

            if current_value <= 0:
                return 0.0

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