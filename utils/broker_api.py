"""
API для подключения к брокерским системам
"""

import json
import time
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
import websocket
import threading

from utils.logger import setup_logger

logger = setup_logger("BROKER_API")


class BrokerAPI:
    """Базовый класс для брокерских API"""

    def __init__(self, config: Dict):
        self.config = config
        self.broker_name = config.get('broker_name', 'unknown')
        self.paper_trading = config.get('paper_trading', True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Статистика запросов
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'last_request': None
        }

        logger.info(f"Инициализирован API для {self.broker_name} (paper: {self.paper_trading})")

    def make_request(self,
                     method: str,
                     url: str,
                     **kwargs) -> Optional[requests.Response]:
        """Выполнение HTTP запроса с обработкой ошибок"""
        try:
            self.request_stats['total_requests'] += 1
            self.request_stats['last_request'] = datetime.now().isoformat()

            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()

            self.request_stats['successful_requests'] += 1

            # Логирование для отладки
            logger.debug(f"{method} {url} - {response.status_code}")

            return response

        except requests.exceptions.RequestException as e:
            self.request_stats['failed_requests'] += 1
            logger.error(f"Ошибка запроса {method} {url}: {e}")
            return None
        except Exception as e:
            self.request_stats['failed_requests'] += 1
            logger.error(f"Неожиданная ошибка запроса: {e}")
            return None

    def get_account_info(self) -> Optional[Dict]:
        """Получение информации о счете"""
        raise NotImplementedError

    def get_portfolio(self) -> Optional[Dict]:
        """Получение портфеля"""
        raise NotImplementedError

    def get_positions(self) -> List[Dict]:
        """Получение позиций"""
        raise NotImplementedError

    def get_orders(self) -> List[Dict]:
        """Получение списка заявок"""
        raise NotImplementedError

    def place_order(self,
                    ticker: str,
                    quantity: int,
                    price: float,
                    direction: str = 'buy',
                    order_type: str = 'limit') -> Optional[str]:
        """Размещение заявки"""
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> bool:
        """Отмена заявки"""
        raise NotImplementedError

    def get_market_data(self, ticker: str) -> Optional[Dict]:
        """Получение рыночных данных"""
        raise NotImplementedError

    def get_stats(self) -> Dict:
        """Получение статистики API"""
        return {
            'broker_name': self.broker_name,
            'paper_trading': self.paper_trading,
            'request_stats': self.request_stats,
            'is_connected': self.check_connection()
        }

    def check_connection(self) -> bool:
        """Проверка соединения с брокером"""
        try:
            # Базовая проверка
            account_info = self.get_account_info()
            return account_info is not None
        except:
            return False


class TinkoffAPI(BrokerAPI):
    """API для Тинькофф Инвестиций"""

    def __init__(self, config: Dict):
        super().__init__(config)

        tinkoff_config = config.get('tinkoff', {})
        self.token = tinkoff_config.get('token', '')
        self.account_id = tinkoff_config.get('account_id', '')
        self.sandbox = tinkoff_config.get('sandbox', True)

        # Базовые URL
        if self.sandbox:
            self.base_url = "https://api-invest.tinkoff.ru/openapi/sandbox"
            logger.info("Используется Sandbox режим Тинькофф")
        else:
            self.base_url = "https://api-invest.tinkoff.ru/openapi"
            logger.info("Используется боевой режим Тинькофф")

        # Настройка заголовков
        self.session.headers.update({
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        })

    def make_tinkoff_request(self,
                             method: str,
                             endpoint: str,
                             **kwargs) -> Optional[Dict]:
        """Выполнение запроса к API Тинькофф"""
        url = f"{self.base_url}{endpoint}"
        response = self.make_request(method, url, **kwargs)

        if response:
            try:
                data = response.json()
                if data.get('status') == 'Error':
                    logger.error(f"Ошибка API Тинькофф: {data.get('payload', {}).get('message', 'Unknown')}")
                    return None
                return data.get('payload')
            except Exception as e:
                logger.error(f"Ошибка парсинга ответа Тинькофф: {e}")

        return None

    def get_account_info(self) -> Optional[Dict]:
        """Получение информации о счетах"""
        endpoint = "/user/accounts"
        data = self.make_tinkoff_request('GET', endpoint)

        if data and 'accounts' in data:
            # Ищем нужный счет
            for account in data['accounts']:
                if account.get('brokerAccountId') == self.account_id:
                    return account

            # Если счет не найден, берем первый
            if data['accounts']:
                return data['accounts'][0]

        return None

    def get_portfolio(self) -> Optional[Dict]:
        """Получение портфеля"""
        if not self.account_id:
            logger.error("Account ID не установлен для Тинькофф API")
            return None

        endpoint = f"/portfolio"
        params = {'brokerAccountId': self.account_id}

        data = self.make_tinkoff_request('GET', endpoint, params=params)
        return data

    def get_positions(self) -> List[Dict]:
        """Получение позиций"""
        portfolio = self.get_portfolio()

        if not portfolio:
            return []

        positions = []

        # Обработка разных типов позиций
        for position_type in ['positions', 'currencies']:
            if position_type in portfolio:
                for item in portfolio[position_type]:
                    if position_type == 'positions':
                        positions.append({
                            'ticker': item.get('ticker'),
                            'figi': item.get('figi'),
                            'name': item.get('name'),
                            'quantity': item.get('quantity', {}).get('lots', 0),
                            'average_price': item.get('averagePositionPrice', {}).get('value', 0),
                            'current_price': item.get('expectedYield', {}).get('value', 0),
                            'instrument_type': item.get('instrumentType')
                        })
                    elif position_type == 'currencies':
                        if item.get('currency') == 'RUB':
                            positions.append({
                                'ticker': 'RUB',
                                'name': 'Рубли',
                                'quantity': item.get('balance', 0),
                                'average_price': 1.0,
                                'current_price': 1.0,
                                'instrument_type': 'currency'
                            })

        return positions

    def get_market_data(self, ticker: str) -> Optional[Dict]:
        """Получение рыночных данных"""
        # Сначала получаем FIGI инструмента
        endpoint = "/market/search/by-ticker"
        params = {'ticker': ticker}

        search_data = self.make_tinkoff_request('GET', endpoint, params=params)

        if not search_data or 'instruments' not in search_data:
            logger.error(f"Инструмент {ticker} не найден")
            return None

        instruments = search_data['instruments']
        if not instruments:
            return None

        # Берем первый подходящий инструмент
        instrument = instruments[0]
        figi = instrument.get('figi')

        if not figi:
            return None

        # Получаем свечи
        to_date = datetime.now()
        from_date = to_date - timedelta(days=1)

        endpoint = f"/market/candles"
        params = {
            'figi': figi,
            'from': from_date.isoformat() + 'Z',
            'to': to_date.isoformat() + 'Z',
            'interval': 'hour'
        }

        candles_data = self.make_tinkoff_request('GET', endpoint, params=params)

        if not candles_data or 'candles' not in candles_data:
            return None

        candles = candles_data['candles']
        if not candles:
            return None

        # Берем последнюю свечу
        last_candle = candles[-1]

        return {
            'figi': figi,
            'ticker': ticker,
            'open': last_candle.get('o', 0),
            'high': last_candle.get('h', 0),
            'low': last_candle.get('l', 0),
            'close': last_candle.get('c', 0),
            'volume': last_candle.get('v', 0),
            'time': last_candle.get('time'),
            'instrument_info': instrument
        }

    def place_order(self,
                    ticker: str,
                    quantity: int,
                    price: float,
                    direction: str = 'buy',
                    order_type: str = 'limit') -> Optional[str]:
        """Размещение заявки"""
        if not self.account_id:
            logger.error("Account ID не установлен")
            return None

        # Получаем FIGI инструмента
        market_data = self.get_market_data(ticker)
        if not market_data:
            logger.error(f"Не удалось получить FIGI для {ticker}")
            return None

        figi = market_data['figi']
        operation = 'Buy' if direction.lower() == 'buy' else 'Sell'

        endpoint = f"/orders/{'limit' if order_type == 'limit' else 'market'}"

        payload = {
            'figi': figi,
            'lots': quantity,
            'operation': operation,
            'price': price
        }

        params = {'brokerAccountId': self.account_id}

        data = self.make_tinkoff_request('POST', endpoint, params=params, json=payload)

        if data and 'orderId' in data:
            order_id = data['orderId']
            logger.info(f"Заявка размещена: {ticker} {operation} {quantity} @ {price}, ID: {order_id}")
            return order_id

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Отмена заявки"""
        if not self.account_id:
            return False

        endpoint = f"/orders/cancel"
        params = {
            'orderId': order_id,
            'brokerAccountId': self.account_id
        }

        data = self.make_tinkoff_request('POST', endpoint, params=params)

        if data and 'status' in data and data['status'] == 'Ok':
            logger.info(f"Заявка {order_id} отменена")
            return True

        return False

    def get_orders(self) -> List[Dict]:
        """Получение активных заявок"""
        if not self.account_id:
            return []

        endpoint = "/orders"
        params = {'brokerAccountId': self.account_id}

        data = self.make_tinkoff_request('GET', endpoint, params=params)

        if data:
            return data

        return []


class AlorAPI(BrokerAPI):
    """API для Alor"""

    def __init__(self, config: Dict):
        super().__init__(config)

        alor_config = config.get('alor', {})
        self.refresh_token = alor_config.get('refresh_token', '')
        self.account = alor_config.get('account', '')

        # Базовые URL
        self.base_url = "https://api.alor.ru"
        self.ws_url = "wss://api.alor.ru/ws"

        # Получение access token
        self.access_token = self._get_access_token()

        if self.access_token:
            self.session.headers.update({
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            })
            logger.info("Успешно авторизован в Alor API")
        else:
            logger.error("Не удалось авторизоваться в Alor API")

    def _get_access_token(self) -> Optional[str]:
        """Получение access token"""
        try:
            url = f"{self.base_url}/refresh"
            payload = {
                'token': self.refresh_token
            }

            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data.get('AccessToken')

        except Exception as e:
            logger.error(f"Ошибка получения токена Alor: {e}")
            return None

    def make_alor_request(self,
                          method: str,
                          endpoint: str,
                          **kwargs) -> Optional[Dict]:
        """Выполнение запроса к API Alor"""
        url = f"{self.base_url}{endpoint}"
        response = self.make_request(method, url, **kwargs)

        if response:
            try:
                return response.json()
            except Exception as e:
                logger.error(f"Ошибка парсинга ответа Alor: {e}")

        return None

    def get_account_info(self) -> Optional[Dict]:
        """Получение информации о счете"""
        endpoint = "/md/v2/clients"
        params = {
            'portfolio': self.account,
            'exchange': 'MOEX',
            'format': 'Simple'
        }

        data = self.make_alor_request('GET', endpoint, params=params)
        return data

    def get_portfolio(self) -> Optional[Dict]:
        """Получение портфеля"""
        return self.get_account_info()

    def get_positions(self) -> List[Dict]:
        """Получение позиций"""
        portfolio = self.get_portfolio()

        if not portfolio:
            return []

        positions = []

        # Обработка позиций
        if 'portfolio' in portfolio:
            for item in portfolio['portfolio']:
                if item.get('qty', 0) != 0:
                    positions.append({
                        'ticker': item.get('ticker'),
                        'symbol': item.get('symbol'),
                        'quantity': item.get('qty', 0),
                        'average_price': item.get('avg_price', 0),
                        'current_price': item.get('last_price', 0),
                        'qty_batch': item.get('qty_batch', 1)
                    })

        return positions

    def get_market_data(self, ticker: str) -> Optional[Dict]:
        """Получение рыночных данных"""
        endpoint = f"/md/v2/securities/{ticker}"
        params = {'exchange': 'MOEX'}

        data = self.make_alor_request('GET', endpoint, params=params)

        if data:
            return {
                'ticker': ticker,
                'last_price': data.get('lastPrice', 0),
                'open_price': data.get('openPrice', 0),
                'high_price': data.get('highPrice', 0),
                'low_price': data.get('lowPrice', 0),
                'close_price': data.get('closePrice', 0),
                'volume': data.get('volume', 0),
                'bid': data.get('bid', 0),
                'ask': data.get('ask', 0),
                'lot_size': data.get('lotSize', 1),
                'min_step': data.get('minStep', 0.01)
            }

        return None

    def place_order(self,
                    ticker: str,
                    quantity: int,
                    price: float,
                    direction: str = 'buy',
                    order_type: str = 'limit') -> Optional[str]:
        """Размещение заявки"""
        endpoint = "/commandapi/warptrans/TRADE/v2/client/orders/actions/market/limit"

        side = 'buy' if direction.lower() == 'buy' else 'sell'

        payload = {
            'side': side,
            'type': 'limit',
            'quantity': quantity,
            'price': price,
            'instrument': {
                'symbol': ticker,
                'exchange': 'MOEX'
            },
            'user': {
                'portfolio': self.account,
                'exchange': 'MOEX'
            }
        }

        data = self.make_alor_request('POST', endpoint, json=payload)

        if data and 'orderNumber' in data:
            order_id = data['orderNumber']
            logger.info(f"Заявка размещена: {ticker} {side} {quantity} @ {price}, ID: {order_id}")
            return order_id

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Отмена заявки"""
        endpoint = f"/commandapi/warptrans/TRADE/v2/client/orders/{order_id}/cancel"

        payload = {
            'portfolio': self.account,
            'exchange': 'MOEX'
        }

        data = self.make_alor_request('POST', endpoint, json=payload)

        if data and 'message' in data and data['message'] == 'Ok':
            logger.info(f"Заявка {order_id} отменена")
            return True

        return False

    def get_orders(self) -> List[Dict]:
        """Получение активных заявок"""
        endpoint = "/commandapi/warptrans/TRADE/v2/client/orders"
        params = {
            'portfolio': self.account,
            'exchange': 'MOEX',
            'format': 'Simple'
        }

        data = self.make_alor_request('GET', endpoint, params=params)

        if data and 'orders' in data:
            return data['orders']

        return []


class PaperTradingAPI(BrokerAPI):
    """API для бумажной торговли (симуляции)"""

    def __init__(self, config: Dict):
        super().__init__(config)

        # Состояние симуляции
        self.simulation_portfolio = {
            'cash': config.get('initial_capital', 10000.0),
            'positions': {},  # {ticker: {quantity, avg_price}}
            'orders': [],  # Активные заявки
            'trade_history': []
        }

        # Настройки комиссий
        self.commission_rate = config.get('commission_percent', 0.05) / 100
        self.slippage_percent = config.get('slippage_percent', 0.1) / 100

        logger.info(f"Инициализирована бумажная торговля с капиталом {self.simulation_portfolio['cash']:,.0f}₽")

    def get_account_info(self) -> Optional[Dict]:
        """Информация о счете"""
        return {
            'account_id': 'PAPER_TRADING',
            'account_type': 'paper',
            'currency': 'RUB',
            'cash': self.simulation_portfolio['cash'],
            'total_value': self._get_total_value(),
            'last_update': datetime.now().isoformat()
        }

    def get_portfolio(self) -> Optional[Dict]:
        """Портфель бумажной торговли"""
        return {
            'total_value': self._get_total_value(),
            'cash': self.simulation_portfolio['cash'],
            'positions': self.simulation_portfolio['positions'],
            'orders': self.simulation_portfolio['orders'],
            'statistics': {
                'total_trades': len(self.simulation_portfolio['trade_history']),
                'total_commission': sum(t.get('commission', 0) for t in self.simulation_portfolio['trade_history']),
                'total_pnl': sum(t.get('pnl', 0) for t in self.simulation_portfolio['trade_history'])
            }
        }

    def get_positions(self) -> List[Dict]:
        """Позиции бумажной торговли"""
        positions = []

        for ticker, pos in self.simulation_portfolio['positions'].items():
            positions.append({
                'ticker': ticker,
                'quantity': pos['quantity'],
                'avg_price': pos['avg_price'],
                'current_price': pos.get('current_price', pos['avg_price']),
                'market_value': pos['quantity'] * pos.get('current_price', pos['avg_price']),
                'pnl': pos['quantity'] * (pos.get('current_price', pos['avg_price']) - pos['avg_price'])
            })

        return positions

    def get_market_data(self, ticker: str) -> Optional[Dict]:
        """Рыночные данные (симуляция)"""
        # В реальной системе здесь был бы запрос к MOEX
        # Для симуляции возвращаем фиктивные данные
        import random

        base_price = 100.0 + random.random() * 900.0  # 100-1000 рублей

        return {
            'ticker': ticker,
            'last_price': base_price,
            'open_price': base_price * (0.98 + random.random() * 0.04),
            'high_price': base_price * (1.0 + random.random() * 0.05),
            'low_price': base_price * (0.95 + random.random() * 0.05),
            'volume': random.randint(1000, 1000000),
            'bid': base_price * (0.999 - random.random() * 0.002),
            'ask': base_price * (1.001 + random.random() * 0.002),
            'lot_size': 1,
            'min_step': 0.01,
            'is_simulated': True
        }

    def place_order(self,
                    ticker: str,
                    quantity: int,
                    price: float,
                    direction: str = 'buy',
                    order_type: str = 'limit') -> Optional[str]:
        """Размещение заявки в симуляции"""
        import random
        import uuid

        # Генерируем ID заявки
        order_id = str(uuid.uuid4())[:8]

        # Применяем проскальзывание
        if direction == 'buy':
            executed_price = price * (1 + self.slippage_percent)
        else:
            executed_price = price * (1 - self.slippage_percent)

        # Комиссия
        trade_value = quantity * executed_price
        commission = trade_value * self.commission_rate

        # Исполнение заявки
        if direction == 'buy':
            # Проверка доступности средств
            total_cost = trade_value + commission

            if total_cost > self.simulation_portfolio['cash']:
                logger.error(
                    f"Недостаточно средств для покупки: нужно {total_cost:,.0f}₽, есть {self.simulation_portfolio['cash']:,.0f}₽")
                return None

            # Исполнение покупки
            self.simulation_portfolio['cash'] -= total_cost

            # Обновление позиции
            if ticker in self.simulation_portfolio['positions']:
                pos = self.simulation_portfolio['positions'][ticker]
                total_qty = pos['quantity'] + quantity
                total_cost_prev = pos['quantity'] * pos['avg_price']
                new_avg_price = (total_cost_prev + trade_value) / total_qty

                pos['quantity'] = total_qty
                pos['avg_price'] = new_avg_price
            else:
                self.simulation_portfolio['positions'][ticker] = {
                    'quantity': quantity,
                    'avg_price': executed_price
                }

            logger.info(f"СИМУЛЯЦИЯ: КУПЛЕНО {ticker} {quantity} @ {executed_price:.2f}, комиссия: {commission:.1f}₽")

        else:  # sell
            # Проверка наличия позиции
            if ticker not in self.simulation_portfolio['positions']:
                logger.error(f"Нет позиции для продажи: {ticker}")
                return None

            pos = self.simulation_portfolio['positions'][ticker]

            if quantity > pos['quantity']:
                logger.error(f"Недостаточно акций для продажи: нужно {quantity}, есть {pos['quantity']}")
                return None

            # Исполнение продажи
            revenue = trade_value - commission
            self.simulation_portfolio['cash'] += revenue

            # Расчет PnL
            entry_cost = quantity * pos['avg_price']
            pnl = revenue - entry_cost

            # Обновление позиции
            if quantity == pos['quantity']:
                del self.simulation_portfolio['positions'][ticker]
            else:
                pos['quantity'] -= quantity

            logger.info(f"СИМУЛЯЦИЯ: ПРОДАНО {ticker} {quantity} @ {executed_price:.2f}, "
                        f"комиссия: {commission:.1f}₽, PnL: {pnl:+,.0f}₽")

        # Запись в историю
        trade_record = {
            'order_id': order_id,
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'direction': direction,
            'quantity': quantity,
            'price': executed_price,
            'commission': commission,
            'pnl': pnl if direction == 'sell' else 0,
            'cash_after': self.simulation_portfolio['cash']
        }

        self.simulation_portfolio['trade_history'].append(trade_record)

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """Отмена заявки в симуляции"""
        # В симуляции заявки исполняются мгновенно
        logger.warning(f"В симуляции заявки исполняются мгновенно, отмена невозможна")
        return False

    def get_orders(self) -> List[Dict]:
        """Активные заявки в симуляции"""
        # В симуляции нет активных заявок
        return []

    def _get_total_value(self) -> float:
        """Расчет общей стоимости портфеля"""
        total = self.simulation_portfolio['cash']

        for ticker, pos in self.simulation_portfolio['positions'].items():
            # Получаем текущую цену
            market_data = self.get_market_data(ticker)
            current_price = market_data['last_price'] if market_data else pos['avg_price']

            total += pos['quantity'] * current_price

        return total

    def reset_portfolio(self, new_capital: float = 10000.0):
        """Сброс портфеля симуляции"""
        self.simulation_portfolio = {
            'cash': new_capital,
            'positions': {},
            'orders': [],
            'trade_history': []
        }

        logger.info(f"Портфель симуляции сброшен, новый капитал: {new_capital:,.0f}₽")


# Фабрика для создания API
def create_broker_api(config: Dict) -> Optional[BrokerAPI]:
    """Создание API для брокера"""
    broker_name = config.get('broker_name', '').lower()

    if broker_name == 'tinkoff':
        return TinkoffAPI(config)
    elif broker_name == 'alor':
        return AlorAPI(config)
    elif broker_name == 'paper' or config.get('paper_trading', True):
        return PaperTradingAPI(config)
    else:
        logger.error(f"Неподдерживаемый брокер: {broker_name}")
        return None