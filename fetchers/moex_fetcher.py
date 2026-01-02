"""
Получение данных с Московской биржи (MOEX)
"""

import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger("MOEX_FETCHER")


class MoexFetcher:
    """Класс для получения данных с MOEX"""

    def __init__(self, use_cache: bool = True, cache_ttl: int = 300):
        self.base_url = "https://iss.moex.com/iss"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Кэширование
        self.use_cache = use_cache
        self.cache_ttl = cache_ttl
        self.cache = {}

        logger.info("Инициализирован MOEX Fetcher")

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Получение данных из кэша"""
        if not self.use_cache or key not in self.cache:
            return None

        cached_data, timestamp = self.cache[key]

        if time.time() - timestamp > self.cache_ttl:
            del self.cache[key]
            return None

        return cached_data

    def _save_to_cache(self, key: str, data: Any):
        """Сохранение данных в кэш"""
        if self.use_cache:
            self.cache[key] = (data, time.time())

    def get_all_securities(self) -> Dict[str, Dict]:
        """Получение списка всех ликвидных бумаг"""
        cache_key = "all_securities"
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            logger.debug("Использую кэшированные данные по бумагам")
            return cached

        try:
            # Используем эндпоинт для получения списка бумаг с основными параметрами
            url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities.json"
            params = {
                'iss.meta': 'off',
                'iss.only': 'securities',
                'securities.columns': 'SECID,SHORTNAME,SECNAME,LOTSIZE,PREVPRICE',
                'limit': 100
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Обработка данных - только securities, без marketdata
            securities = {}

            if 'securities' not in data:
                logger.error("Нет данных по бумагам в ответе")
                return {}

            columns = data['securities']['columns']
            rows = data['securities']['data']

            for row in rows:
                try:
                    ticker = row[columns.index('SECID')]
                    name = row[columns.index('SHORTNAME')]

                    # Получаем предыдущую цену, если есть
                    prev_price_idx = columns.index('PREVPRICE') if 'PREVPRICE' in columns else -1
                    prev_price = row[prev_price_idx] if prev_price_idx != -1 and prev_price_idx < len(row) else 0

                    # Получаем текущую цену отдельным запросом
                    current_price = self.get_price(ticker) or 0

                    # Простой расчет момента
                    momentum = 0.0
                    if prev_price and prev_price > 0 and current_price > 0:
                        momentum = (current_price / prev_price - 1) * 100

                    securities[ticker] = {
                        'name': name,
                        'price': current_price,
                        'prev_price': prev_price,
                        'volume': 0,  # Нужен отдельный запрос для объема
                        'change': current_price - prev_price if prev_price else 0,
                        'momentum': momentum,
                        'liquidity': 0.5,  # По умолчанию средняя ликвидность
                        'spread': 0.01,
                        'market_cap': 0,
                        'update_time': datetime.now().isoformat()
                    }

                except (IndexError, ValueError) as e:
                    logger.warning(f"Ошибка обработки строки данных: {e}")
                    continue

            logger.info(f"Загружено {len(securities)} бумаг")

            # Сохранение в кэш
            self._save_to_cache(cache_key, securities)

            return securities

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при получении бумаг: {e}")
            return {}
        except Exception as e:
            logger.error(f"Ошибка обработки данных бумаг: {e}")
            return {}

    def get_price(self, ticker: str) -> Optional[float]:
        """Получение текущей цены по тикеру"""
        cache_key = f"price_{ticker}"
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            return cached

        try:
            # Получаем данные по конкретному тикеру
            url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
            params = {
                'iss.meta': 'off'
            }

            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()

            # Ищем последнюю цену
            marketdata = data.get('marketdata', {})
            columns = marketdata.get('columns', [])
            rows = marketdata.get('data', [])

            if rows and columns:
                # Берем первую строку (последние данные)
                row = rows[0]

                # Ищем колонку с последней ценой
                price_columns = ['LAST', 'LCURRENTPRICE', 'OPEN', 'CLOSE']

                for price_col in price_columns:
                    if price_col in columns:
                        idx = columns.index(price_col)
                        price = row[idx] if idx < len(row) else None

                        if price and price > 0:
                            # Сохраняем в кэш
                            self._save_to_cache(cache_key, float(price))
                            return float(price)

            logger.warning(f"Не удалось получить цену для {ticker}")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при получении цены {ticker}: {e}")
            return None
        except Exception as e:
            logger.error(f"Ошибка обработки цены {ticker}: {e}")
            return None

    def get_candles(self,
                    ticker: str,
                    interval: int = 60,  # минуты
                    count: int = 100) -> Optional[pd.DataFrame]:
        """Получение исторических свечей"""
        cache_key = f"candles_{ticker}_{interval}_{count}"
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            return cached

        try:
            # Интервалы: 1, 10, 60 минут
            interval_str = str(interval)

            url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
            params = {
                'interval': interval_str,
                'count': count,
                'iss.meta': 'off'
            }

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            candles = data.get('candles', {})
            columns = candles.get('columns', [])
            rows = candles.get('data', [])

            if not rows:
                return None

            # Создаем DataFrame
            df = pd.DataFrame(rows, columns=columns)

            # Конвертируем даты
            if 'begin' in df.columns:
                df['datetime'] = pd.to_datetime(df['begin'])
                df.set_index('datetime', inplace=True)

            # Переименовываем колонки
            column_mapping = {
                'open': 'Open',
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume',
                'value': 'Value'
            }

            df.rename(columns={k.lower(): v for k, v in column_mapping.items()
                               if k.lower() in df.columns}, inplace=True)

            # Сохраняем в кэш
            self._save_to_cache(cache_key, df)

            logger.debug(f"Получено {len(df)} свечей для {ticker} (интервал {interval}мин)")

            return df

        except Exception as e:
            logger.error(f"Ошибка получения свечей для {ticker}: {e}")
            return None

    def get_orderbook(self, ticker: str, depth: int = 10) -> Optional[Dict]:
        """Получение стакана заявок"""
        try:
            url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/orderbook.json"
            params = {
                'depth': depth,
                'iss.meta': 'off'
            }

            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()
            orderbook = data.get('orderbook', {})
            columns = orderbook.get('columns', [])
            rows = orderbook.get('data', [])

            if not rows:
                return None

            # Обработка стакана
            bids = []
            asks = []

            for row in rows:
                if len(row) >= 4:
                    price = row[columns.index('PRICE')]
                    quantity = row[columns.index('QUANTITY')]
                    is_bid = row[columns.index('BUYSELL')] == 'B'

                    if is_bid:
                        bids.append({'price': price, 'quantity': quantity})
                    else:
                        asks.append({'price': price, 'quantity': quantity})

            # Сортируем
            bids.sort(key=lambda x: x['price'], reverse=True)  # По убыванию цены
            asks.sort(key=lambda x: x['price'])  # По возрастанию цены

            result = {
                'bids': bids[:depth],
                'asks': asks[:depth],
                'spread': asks[0]['price'] - bids[0]['price'] if bids and asks else 0,
                'total_bid_volume': sum(b['quantity'] for b in bids[:depth]),
                'total_ask_volume': sum(a['quantity'] for a in asks[:depth]),
                'timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Ошибка получения стакана для {ticker}: {e}")
            return None

    def get_market_indices(self) -> Dict[str, float]:
        """Получение основных рыночных индексов"""
        cache_key = "market_indices"
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            return cached

        try:
            indices = {}

            # Основные индексы MOEX
            index_tickers = ['IMOEX', 'RTSI', 'MOEXBMI', 'MOEXFN', 'MOEXOG', 'MOEXTL']

            for ticker in index_tickers:
                price = self.get_price(ticker)
                if price:
                    indices[ticker] = price

            # Добавляем расчетные метрики
            if indices:
                indices['market_mood'] = self._calculate_market_mood(indices)
                indices['update_time'] = datetime.now().isoformat()

            # Сохраняем в кэш
            self._save_to_cache(cache_key, indices)

            return indices

        except Exception as e:
            logger.error(f"Ошибка получения индексов: {e}")
            return {}

    def _calculate_market_mood(self, indices: Dict[str, float]) -> float:
        """Расчет общего настроения рынка"""
        try:
            # Простая эвристика на основе индексов
            if 'IMOEX' in indices and 'RTSI' in indices:
                # Сравнение текущих значений с предыдущими (в реальности нужно хранить историю)
                return 0.0  # По умолчанию нейтральное

            return 0.0
        except:
            return 0.0

    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Получение полной информации по тикеру"""
        try:
            # Получаем базовую информацию
            url = f"{self.base_url}/securities/{ticker}.json"
            params = {'iss.meta': 'off'}

            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()
            securities = data.get('securities', {})
            columns = securities.get('columns', [])
            rows = securities.get('data', [])

            if not rows:
                return None

            row = rows[0]
            info = {}

            # Важные поля
            fields_mapping = {
                'SHORTNAME': 'short_name',
                'SECNAME': 'full_name',
                'ISIN': 'isin',
                'REGNUMBER': 'reg_number',
                'ISSUESIZE': 'issue_size',
                'ISSUECAPITALIZATION': 'market_cap',
                'CURRENCYID': 'currency',
                'LOTSIZE': 'lot_size',
                'MINSTEP': 'min_step'
            }

            for col_ru, col_en in fields_mapping.items():
                if col_ru in columns:
                    idx = columns.index(col_ru)
                    if idx < len(row):
                        info[col_en] = row[idx]

            # Добавляем текущую цену
            price = self.get_price(ticker)
            if price:
                info['current_price'] = price

            # Добавляем свечи за день
            candles = self.get_candles(ticker, interval=60, count=24)
            if candles is not None and not candles.empty:
                info['daily_high'] = candles['High'].max()
                info['daily_low'] = candles['Low'].min()
                info['daily_volume'] = candles['Volume'].sum()
                info['daily_change'] = ((candles['Close'].iloc[-1] / candles['Close'].iloc[0]) - 1) * 100

            info['last_updated'] = datetime.now().isoformat()

            return info

        except Exception as e:
            logger.error(f"Ошибка получения информации по {ticker}: {e}")
            return None

    def get_market_status(self) -> Dict:
        """Получение статуса рынка"""
        try:
            # Простая проверка - пытаемся получить данные
            test_ticker = "SBER"
            price = self.get_price(test_ticker)

            status = {
                'is_available': price is not None,
                'last_check': datetime.now().isoformat(),
                'test_ticker': test_ticker,
                'test_price': price
            }

            if price is None:
                logger.warning("Рынок недоступен или закрыт")
            else:
                logger.debug(f"Рынок доступен, цена SBER: {price}")

            return status

        except Exception as e:
            logger.error(f"Ошибка проверки статуса рынка: {e}")
            return {'is_available': False, 'error': str(e)}

    def clear_cache(self):
        """Очистка кэша"""
        self.cache.clear()
        logger.info("Кэш MOEX очищен")