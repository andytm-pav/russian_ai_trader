"""
Получение данных с Московской биржи (MOEX) - оптимизированная версия
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

        # Rate limiting
        self.request_timestamps = []
        self.max_requests_per_minute = 50

        # Статистика
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"Инициализирован MOEX Fetcher (лимит: {self.max_requests_per_minute} запр/мин)")

    def _rate_limit(self):
        """Контроль частоты запросов"""
        now = time.time()

        # Удаляем старые записи (старше 1 минуты)
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]

        # Если достигли лимита, ждём
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            oldest = self.request_timestamps[0]
            sleep_time = 60 - (now - oldest) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.time()
                self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]

        self.request_timestamps.append(now)

    def _make_request(self, url: str, params: Dict = None, timeout: int = 10) -> Optional[Dict]:
        """Обёртка для выполнения запросов с rate limiting и обработкой ошибок"""
        self._rate_limit()

        try:
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при запросе {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Ошибка обработки ответа {url}: {e}")
            return None

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Получение данных из кэша"""
        if not self.use_cache or key not in self.cache:
            self.cache_misses += 1
            return None

        cached_data, timestamp = self.cache[key]

        if time.time() - timestamp > self.cache_ttl:
            del self.cache[key]
            self.cache_misses += 1
            return None

        self.cache_hits += 1
        return cached_data

    def _save_to_cache(self, key: str, data: Any):
        """Сохранение данных в кэш"""
        if self.use_cache:
            self.cache[key] = (data, time.time())

    def get_prices_batch(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Получение цен для нескольких тикеров одним запросом"""
        if not tickers:
            return {}

        results = {}
        batch_size = 50  # MOEX API поддерживает до 100 тикеров, но для надёжности 50

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]

            url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities.json"
            params = {
                'securities': ','.join(batch),
                'iss.meta': 'off',
                'iss.only': 'marketdata',
                'marketdata.columns': 'SECID,LAST,CHANGE,LASTTOPREVPRICE',
                'limit': len(batch)
            }

            data = self._make_request(url, params, timeout=15)
            if not data or 'marketdata' not in data:
                continue

            columns = data['marketdata']['columns']
            for row in data['marketdata']['data']:
                try:
                    ticker = row[columns.index('SECID')]
                    last_price = row[columns.index('LAST')] if 'LAST' in columns else None

                    if last_price and last_price > 0:
                        results[ticker] = float(last_price)
                        # Кэшируем отдельную цену
                        self._save_to_cache(f"price_{ticker}", float(last_price))
                except (IndexError, ValueError) as e:
                    logger.debug(f"Ошибка обработки цены для батча: {e}")
                    continue

        return results

    def get_all_securities(self) -> Dict[str, Dict]:
        """Получение списка всех ликвидных бумаг - исправленная оптимизированная версия"""
        cache_key = "all_securities"
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            logger.debug("Использую кэшированные данные по бумагам")
            return cached

        try:
            # 1. Получаем список бумаг (securities)
            url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities.json"
            params = {
                'iss.meta': 'off',
                'iss.only': 'securities',
                'securities.columns': 'SECID,SHORTNAME,SECNAME,LOTSIZE,PREVPRICE',
                'limit': 100
            }

            logger.debug(f"Запрос списка бумаг...")

            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if 'securities' not in data:
                logger.error("Нет ключа 'securities' в ответе")
                return {}

            # Обрабатываем список бумаг
            columns = data['securities']['columns']
            rows = data['securities']['data']
            logger.debug(f"Получено {len(rows)} строк с бумагами")

            tickers = []
            sec_dict = {}

            for row in rows:
                try:
                    ticker = row[columns.index('SECID')]
                    name = row[columns.index('SHORTNAME')]

                    tickers.append(ticker)

                    # Безопасное получение PREVPRICE
                    prev_price = 0
                    if 'PREVPRICE' in columns:
                        idx = columns.index('PREVPRICE')
                        if idx < len(row) and row[idx] is not None:
                            try:
                                prev_price = float(row[idx])
                            except (ValueError, TypeError):
                                prev_price = 0

                    sec_dict[ticker] = {
                        'name': name,
                        'full_name': row[columns.index('SECNAME')] if 'SECNAME' in columns else name,
                        'lot_size': row[columns.index('LOTSIZE')] if 'LOTSIZE' in columns else 1,
                        'prev_price': prev_price
                    }
                except (IndexError, ValueError) as e:
                    logger.debug(f"Ошибка обработки строки для {ticker}: {e}")
                    continue

            # 2. Получаем текущие цены batch-запросом (marketdata)
            logger.debug(f"Запрашиваем цены для {len(tickers)} тикеров...")

            # Разбиваем на батчи по 50
            all_prices = {}
            batch_size = 50

            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                logger.debug(f"Батч {i // batch_size + 1}: {len(batch)} тикеров")

                url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities.json"
                params = {
                    'securities': ','.join(batch),
                    'iss.meta': 'off',
                    'iss.only': 'marketdata',
                    'marketdata.columns': 'SECID,LAST',
                    'limit': len(batch)
                }

                try:
                    response = self.session.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    batch_data = response.json()

                    if 'marketdata' in batch_data:
                        md_columns = batch_data['marketdata']['columns']
                        for md_row in batch_data['marketdata']['data']:
                            try:
                                ticker = md_row[md_columns.index('SECID')]

                                # Безопасное получение LAST цены
                                last_price = 0
                                if 'LAST' in md_columns:
                                    idx = md_columns.index('LAST')
                                    if idx < len(md_row) and md_row[idx] is not None:
                                        try:
                                            last_price = float(md_row[idx])
                                        except (ValueError, TypeError):
                                            last_price = 0

                                all_prices[ticker] = last_price
                            except (IndexError, ValueError) as e:
                                logger.debug(f"Ошибка обработки marketdata для батча: {e}")
                                continue
                except Exception as e:
                    logger.warning(f"Ошибка batch-запроса для {len(batch)} тикеров: {e}")
                    # Fallback: получаем цены по одному для этого батча
                    for ticker in batch:
                        try:
                            price = self.get_price(ticker)
                            all_prices[ticker] = price if price else 0
                        except:
                            all_prices[ticker] = 0

            # 3. Формируем итоговый результат
            securities = {}
            for ticker in tickers:
                if ticker in sec_dict:
                    base_data = sec_dict[ticker]
                    current_price = all_prices.get(ticker, 0.0)
                    prev_price = base_data['prev_price']

                    # Расчет момента
                    momentum = 0.0
                    if prev_price and prev_price > 0 and current_price > 0:
                        momentum = ((current_price / prev_price) - 1) * 100

                    securities[ticker] = {
                        'name': base_data['name'],
                        'full_name': base_data['full_name'],
                        'lot_size': base_data['lot_size'],
                        'price': current_price,
                        'prev_price': prev_price,
                        'volume': 0,  # Можно добавить отдельным запросом VALTODAY
                        'change': current_price - prev_price if prev_price else 0,
                        'momentum': momentum,
                        'liquidity': 0.5,
                        'spread': 0.01,
                        'market_cap': 0,
                        'update_time': datetime.now().isoformat()
                    }

            logger.info(f"Загружено {len(securities)} бумаг (использованы batch-запросы)")

            # Сохранение в кэш
            self._save_to_cache(cache_key, securities)

            return securities

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка сети при получении бумаг: {e}")
            return {}
        except Exception as e:
            logger.error(f"Ошибка обработки данных бумаг: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def get_price(self, ticker: str) -> Optional[float]:
        """Более простая и надежная версия get_price"""
        cache_key = f"price_{ticker}"
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            return cached

        try:
            # Простой запрос к marketdata
            url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
            params = {'iss.meta': 'off'}

            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            marketdata = data.get('marketdata', {})
            columns = marketdata.get('columns', [])
            rows = marketdata.get('data', [])

            if rows and columns:
                row = rows[0]
                price_columns = ['LAST', 'LCURRENTPRICE', 'OPEN', 'CLOSE']

                for price_col in price_columns:
                    if price_col in columns:
                        idx = columns.index(price_col)
                        if idx < len(row) and row[idx] is not None:
                            try:
                                price = float(row[idx])
                                if price > 0:
                                    self._save_to_cache(cache_key, price)
                                    return price
                            except (ValueError, TypeError):
                                continue

            return None

        except Exception as e:
            logger.error(f"Ошибка получения цены {ticker}: {e}")
            return None

    def get_candles(self,
                    ticker: str,
                    interval: int = 60,  # минуты
                    count: int = 100) -> Optional[pd.DataFrame]:
        """Получение исторических свечей"""
        # Валидация интервала
        valid_intervals = {1: "1", 10: "10", 60: "60", 24: "24", 7: "7", 31: "31", 4: "4"}
        if interval not in valid_intervals:
            logger.warning(f"Некорректный интервал {interval}, используется 60")
            interval = 60

        cache_key = f"candles_{ticker}_{interval}_{count}"
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            return cached

        try:
            interval_str = valid_intervals[interval]

            # Используем параметры from и till для точного контроля периода
            till_date = datetime.now()
            from_date = till_date - timedelta(days=30 if interval in [24, 7, 31, 4] else 7)

            url = f"{self.base_url}/engines/stock/markets/shares/securities/{ticker}/candles.json"
            params = {
                'interval': interval_str,
                'from': from_date.strftime('%Y-%m-%d'),
                'till': till_date.strftime('%Y-%m-%d'),
                'iss.meta': 'off'
            }

            data = self._make_request(url, params, timeout=15)
            if not data or 'candles' not in data:
                return None

            candles = data['candles']
            columns = candles.get('columns', [])
            rows = candles.get('data', [])

            if not rows:
                return None

            # Создаем DataFrame
            df = pd.DataFrame(rows, columns=columns)

            # Конвертируем даты и устанавливаем индекс
            if 'begin' in df.columns:
                df['datetime'] = pd.to_datetime(df['begin'])
                df.set_index('datetime', inplace=True)

                # Сортируем по дате
                df.sort_index(inplace=True)

                # Ограничиваем количество строк если их больше запрошенного
                if len(df) > count:
                    df = df.iloc[-count:]

            # Стандартизируем названия колонок
            column_mapping = {
                'open': 'Open',
                'close': 'Close',
                'high': 'High',
                'low': 'Low',
                'volume': 'Volume',
                'value': 'Value'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)

            # Сохраняем в кэш
            self._save_to_cache(cache_key, df)

            logger.debug(f"Получено {len(df)} свечей для {ticker} (интервал {interval})")

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

            data = self._make_request(url, params, timeout=8)
            if not data or 'orderbook' not in data:
                return None

            orderbook = data['orderbook']
            columns = orderbook.get('columns', [])
            rows = orderbook.get('data', [])

            if not rows:
                return None

            # Обработка стакана
            bids = []
            asks = []

            for row in rows:
                if len(row) >= 4:
                    try:
                        price = row[columns.index('PRICE')]
                        quantity = row[columns.index('QUANTITY')]
                        is_bid = row[columns.index('BUYSELL')] == 'B'

                        if is_bid:
                            bids.append({'price': float(price), 'quantity': int(quantity)})
                        else:
                            asks.append({'price': float(price), 'quantity': int(quantity)})
                    except (IndexError, ValueError) as e:
                        logger.debug(f"Ошибка обработки строки стакана: {e}")
                        continue

            # Сортируем
            bids.sort(key=lambda x: x['price'], reverse=True)
            asks.sort(key=lambda x: x['price'])

            # Ограничиваем глубину
            bids = bids[:depth]
            asks = asks[:depth]

            # Рассчитываем спред если есть данные
            spread = 0
            if bids and asks:
                spread = asks[0]['price'] - bids[0]['price']

            result = {
                'bids': bids,
                'asks': asks,
                'spread': spread,
                'total_bid_volume': sum(b['quantity'] for b in bids),
                'total_ask_volume': sum(a['quantity'] for a in asks),
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker
            }

            return result

        except Exception as e:
            logger.error(f"Ошибка получения стакана для {ticker}: {e}")
            return None

    def get_market_indices(self) -> Dict[str, float]:
        """Оптимизированная версия с batch-запросом"""
        cache_key = "market_indices"
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            return cached

        index_tickers = ['IMOEX', 'RTSI', 'MOEXBMI', 'MOEXFN', 'MOEXOG', 'MOEXTL']

        # Используем batch-запрос вместо отдельных
        prices = {}
        if hasattr(self, 'get_prices_batch'):
            prices = self.get_prices_batch(index_tickers)
        else:
            # Fallback: по одному
            for ticker in index_tickers:
                price = self.get_price(ticker)
                if price:
                    prices[ticker] = price

        # Добавляем метрики
        if prices:
            prices['market_mood'] = self._calculate_market_mood(prices)
            prices['update_time'] = datetime.now().isoformat()
            self._save_to_cache(cache_key, prices)

        return prices

    def _calculate_market_mood(self, indices: Dict[str, float]) -> float:
        """Расчет общего настроения рынка"""
        try:
            if not indices:
                return 0.0

            # Простая эвристика: считаем процент положительных изменений
            # В реальном приложении нужно сравнивать с предыдущими значениями
            return 0.0  # По умолчанию нейтральное

        except:
            return 0.0

    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Получение полной информации по тикеру"""
        try:
            # Получаем базовую информацию
            url = f"{self.base_url}/securities/{ticker}.json"
            params = {'iss.meta': 'off'}

            data = self._make_request(url, params, timeout=8)
            if not data or 'securities' not in data:
                return None

            securities = data['securities']
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
                'MINSTEP': 'min_step',
                'PREVPRICE': 'prev_price'
            }

            for col_ru, col_en in fields_mapping.items():
                if col_ru in columns:
                    idx = columns.index(col_ru)
                    if idx < len(row) and row[idx] is not None:
                        info[col_en] = row[idx]

            # Добавляем текущую цену
            price = self.get_price(ticker)
            if price:
                info['current_price'] = price

            # Добавляем свечи за день
            candles = self.get_candles(ticker, interval=24, count=5)
            if candles is not None and not candles.empty:
                info['daily_high'] = candles['High'].max()
                info['daily_low'] = candles['Low'].min()
                info['daily_volume'] = candles['Volume'].sum() if 'Volume' in candles.columns else 0
                if len(candles) > 1:
                    info['daily_change'] = ((candles['Close'].iloc[-1] / candles['Close'].iloc[0]) - 1) * 100

            info['last_updated'] = datetime.now().isoformat()

            return info

        except Exception as e:
            logger.error(f"Ошибка получения информации по {ticker}: {e}")
            return None

    def get_market_status(self) -> Dict:
        """Получение статуса рынка"""
        try:
            # Проверяем доступность API через простой запрос
            url = f"{self.base_url}/engines.json"
            params = {'iss.meta': 'off', 'limit': 1}

            data = self._make_request(url, params, timeout=5)

            status = {
                'is_available': data is not None,
                'last_check': datetime.now().isoformat(),
                'api_version': 'ISS API v1',
                'cache_stats': self.get_cache_stats()
            }

            if data is None:
                logger.warning("Рынок недоступен или закрыт")
            else:
                logger.debug("Рынок доступен")

            return status

        except Exception as e:
            logger.error(f"Ошибка проверки статуса рынка: {e}")
            return {'is_available': False, 'error': str(e)}

    def get_cache_stats(self) -> Dict:
        """Получение статистики кэша"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            'size': len(self.cache),
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'ttl': self.cache_ttl
        }

    def clear_cache(self):
        """Очистка кэша"""
        old_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Кэш MOEX очищен (было {old_size} записей)")