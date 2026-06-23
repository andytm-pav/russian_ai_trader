"""
Получение данных с Московской биржи (MOEX) - оптимизированная версия
"""

import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import xml.etree.ElementTree as ET
import yfinance as yf

from utils.logger import get_logger

logger = get_logger("MOEX_FETCHER")


class MoexFetcher:
    """Класс для получения данных с MOEX"""

    def __init__(self, use_cache: bool = None, cache_ttl: int = None):
        self.base_url = "https://iss.moex.com/iss"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Загрузка настроек
        self.settings = self._load_settings()
        moex_config = self.settings.get('moex_fetcher', {})

        self.brent_normalization = self.settings.get('market_data', {}).get('brent_normalization', 100.0)


        self.max_contracts = moex_config.get('max_contracts', 500)

        # Кэширование из конфига
        self.use_cache = use_cache if use_cache is not None else moex_config.get('use_cache', True)
        self.cache_ttl = cache_ttl if cache_ttl is not None else moex_config.get('cache_ttl', 300)
        self.max_requests_per_minute = moex_config.get('max_requests_per_minute', 50)
        self.request_timestamps = []
        self.cache = {}

        # Статистика кэша
        self.cache_hits = 0
        self.cache_misses = 0

        # История для адаптивной нормализации ликвидности
        self.liquidity_history = []
        self.liquidity_history_size = self.settings.get('market_data', {}).get('liquidity_history_size', 20)

        self.default_rvi = self.settings.get('market_data', {}).get('default_rvi', 20.0)
        self.rvi_ticker = self.settings.get('market_data', {}).get('rvi_ticker', 'RVI')
        self.rvi_board = self.settings.get('market_data', {}).get('rvi_board', 'SNDX')  # Исправлено: SNDX для RVI
        self.liquidity_threshold = self.settings.get('market_data', {}).get('liquidity_threshold', 1000000000)
        self.market_mood_threshold = self.settings.get('market_data', {}).get('market_mood_change_threshold', 5.0)
        self.index_tickers = self.settings.get('market_data', {}).get('index_tickers',
                                                                      ['IMOEX', 'RTSI', 'MOEXBMI', 'MOEXFN', 'MOEXOG',
                                                                       'MOEXTL'])

        logger.info(
            f"Инициализирован MOEX Fetcher (кэш: {self.cache_ttl}с, лимит: {self.max_requests_per_minute} запр/мин)")

    def _safe_float(self, value, default=0.0):
        """Безопасное преобразование в float"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _get_index_change(self, current_value, change, change_pct):
        """
        Получение изменения индекса.
        Вычисляется из процентного изменения (change_pct) для точности.
        """
        if change_pct is not None and current_value is not None:
            try:
                change_pct_float = self._safe_float(change_pct)
                if change_pct_float == 0:
                    return 0.0
                previous_value = current_value / (1 + change_pct_float / 100)
                return current_value - previous_value
            except ZeroDivisionError:
                return 0.0

        # Запасной вариант: абсолютное изменение
        if change is not None:
            return self._safe_float(change)

        return 0.0


    def _load_settings(self) -> Dict:
        """Загрузка настроек из settings.json"""
        try:
            with open("config/settings.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Не удалось загрузить settings.json: {e}")
            return {}


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

    def _make_request(self, url: str, params: Dict = None, timeout: int = 15) -> Optional[Dict]:
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
        """Получение списка всех ликвидных бумаг"""
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
                'securities.columns': 'SECID,SHORTNAME,SECNAME,LOTSIZE,MINSTEP,PREVPRICE',
                'limit': 100
            }

            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if 'securities' not in data:
                logger.error("Нет ключа 'securities' в ответе")
                return {}

            # Обрабатываем список бумаг
            columns = data['securities']['columns']
            rows = data['securities']['data']

            tickers = []
            sec_dict = {}

            for row in rows:
                try:
                    ticker = row[columns.index('SECID')]
                    name = row[columns.index('SHORTNAME')]

                    lot_size = row[columns.index('LOTSIZE')] if 'LOTSIZE' in columns else 1
                    min_step = row[columns.index('MINSTEP')] if 'MINSTEP' in columns else 0.01

                    try:
                        lot_size = int(lot_size)
                    except:
                        lot_size = 1

                    try:
                        min_step = float(min_step)
                    except:
                        min_step = 0.01

                    prev_price = 0
                    if 'PREVPRICE' in columns:
                        idx = columns.index('PREVPRICE')
                        if idx < len(row) and row[idx] is not None:
                            try:
                                prev_price = float(row[idx])
                            except (ValueError, TypeError):
                                prev_price = 0

                    tickers.append(ticker)

                    sec_dict[ticker] = {
                        'name': name,
                        'full_name': row[columns.index('SECNAME')] if 'SECNAME' in columns else name,
                        'lot_size': lot_size,
                        'min_step': min_step,
                        'prev_price': prev_price
                    }

                except (IndexError, ValueError) as e:
                    logger.debug(f"Ошибка обработки строки: {e}")
                    continue

            # 2. Получаем текущие цены и метрики batch-запросом
            logger.debug(f"Запрашиваем данные для {len(tickers)} тикеров...")

            all_data = {}
            batch_size = 50

            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]

                url = f"{self.base_url}/engines/stock/markets/shares/boards/TQBR/securities.json"
                params = {
                    'securities': ','.join(batch),
                    'iss.meta': 'off',
                    'iss.only': 'marketdata',
                    'marketdata.columns': 'SECID,LAST,VALTODAY,VOLTODAY,SPREAD,ISSUECAPITALIZATION,NUMTRADES',
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

                                # Получаем все необходимые данные
                                data_row = {}

                                # LAST цена
                                if 'LAST' in md_columns:
                                    idx = md_columns.index('LAST')
                                    if idx < len(md_row) and md_row[idx] is not None:
                                        try:
                                            data_row['price'] = float(md_row[idx])
                                        except (ValueError, TypeError):
                                            data_row['price'] = 0.0

                                # VALTODAY (объем в рублях)
                                if 'VALTODAY' in md_columns:
                                    idx = md_columns.index('VALTODAY')
                                    if idx < len(md_row) and md_row[idx] is not None:
                                        try:
                                            data_row['volume'] = float(md_row[idx])
                                        except (ValueError, TypeError):
                                            data_row['volume'] = 0.0

                                # VOLTODAY (объем в штуках)
                                if 'VOLTODAY' in md_columns:
                                    idx = md_columns.index('VOLTODAY')
                                    if idx < len(md_row) and md_row[idx] is not None:
                                        try:
                                            data_row['volume_lots'] = float(md_row[idx])
                                        except (ValueError, TypeError):
                                            data_row['volume_lots'] = 0.0

                                # SPREAD
                                if 'SPREAD' in md_columns:
                                    idx = md_columns.index('SPREAD')
                                    if idx < len(md_row) and md_row[idx] is not None:
                                        try:
                                            data_row['spread'] = float(md_row[idx])
                                        except (ValueError, TypeError):
                                            data_row['spread'] = 0.0

                                # ISSUECAPITALIZATION
                                if 'ISSUECAPITALIZATION' in md_columns:
                                    idx = md_columns.index('ISSUECAPITALIZATION')
                                    if idx < len(md_row) and md_row[idx] is not None:
                                        try:
                                            data_row['market_cap'] = float(md_row[idx])
                                        except (ValueError, TypeError):
                                            data_row['market_cap'] = 0.0

                                if 'NUMTRADES' in md_columns:
                                    idx = md_columns.index('NUMTRADES')
                                    if idx < len(md_row) and md_row[idx] is not None:
                                        try:
                                            data_row['num_trades'] = int(md_row[idx])
                                        except (ValueError, TypeError):
                                            data_row['num_trades'] = 0

                                all_data[ticker] = data_row

                            except (IndexError, ValueError) as e:
                                logger.debug(f"Ошибка обработки marketdata: {e}")
                                continue
                except Exception as e:
                    logger.warning(f"Ошибка batch-запроса для {len(batch)} тикеров: {e}")
                    # Fallback
                    for ticker in batch:
                        price = self.get_price(ticker)
                        all_data[ticker] = {'price': price if price else 0.0}

            # 3. Формируем итоговый результат
            securities = {}
            for ticker in tickers:
                if ticker in sec_dict:
                    base_data = sec_dict[ticker]
                    market_data = all_data.get(ticker, {})

                    current_price = market_data.get('price', 0.0)
                    prev_price = base_data['prev_price']

                    # Расчет момента
                    momentum = 0.0
                    if prev_price and prev_price > 0 and current_price > 0:
                        momentum = ((current_price / prev_price) - 1) * 100

                    # Расчет ликвидности (на основе объема)
                    volume = market_data.get('volume', 0.0)
                    liquidity = 0.0
                    if volume > 0:
                        # Нормализуем ликвидность от 0 до 1 (логарифмическая шкала)
                        # Предполагаем, что объем > 1 млрд руб = высокая ликвидность
                        liquidity = min(1.0, volume / self.liquidity_threshold)

                    securities[ticker] = {
                        'name': base_data['name'],
                        'full_name': base_data['full_name'],
                        'lot_size': base_data['lot_size'],
                        'min_step': base_data['min_step'],
                        'price': current_price,
                        'prev_price': prev_price,
                        'volume': volume,
                        'change': current_price - prev_price if prev_price else 0,
                        'momentum': momentum,
                        'liquidity': liquidity,
                        'spread': market_data.get('spread', 0.0),
                        'market_cap': market_data.get('market_cap', 0.0),
                        'num_trades': market_data.get('num_trades', 0),
                        'spread_pct': (market_data.get('spread', 0.0) / current_price * 100) if current_price > 0 else 0.0,
                        'update_time': datetime.now().isoformat()
                    }

            logger.info(f"Загружено {len(securities)} бумаг с реальными метриками")

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

            response = self.session.get(url, params=params, timeout=10)
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

    def get_rtsi(self) -> Optional[Dict[str, float]]:
        """
        Получение индекса РТС (RTSI)
        """
        cache_key = "rtsi"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            url = f"{self.base_url}/engines/stock/markets/index/securities/RTSI.json"
            params = {
                'iss.meta': 'off',
                'iss.only': 'marketdata',
                'marketdata.columns': 'SECID,CURRENTVALUE,LASTCHANGE,LASTCHANGEPRC'
            }

            data = self._make_request(url, params, timeout=10)
            if data and 'marketdata' in data and data['marketdata']['data']:
                row = data['marketdata']['data'][0]
                cols = data['marketdata']['columns']

                result = {
                    'value': float(row[cols.index('CURRENTVALUE')]),
                    'change': float(row[cols.index('LASTCHANGE')]) if 'LASTCHANGE' in cols else 0,
                    'change_pct': float(row[cols.index('LASTCHANGEPRC')]) if 'LASTCHANGEPRC' in cols else 0,
                    'timestamp': datetime.now().isoformat()
                }

                self._save_to_cache(cache_key, result)
                logger.debug(f"📊 RTSI получен: {result['value']} (изм: {result['change_pct']}%)")
                return result

            return None

        except Exception as e:
            logger.error(f"Ошибка получения RTSI: {e}")
            return None

    def get_rvi(self) -> Optional[Dict[str, float]]:
        """
        Получение индекса волатильности RVI
        """
        cache_key = "rvi"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            # Исправленный эндпоинт для RVI
            url = f"{self.base_url}/engines/stock/markets/index/securities/RVI.json"
            params = {
                'iss.meta': 'off',
                'iss.only': 'marketdata',
                'marketdata.columns': 'SECID,CURRENTVALUE,LASTCHANGE,LASTCHANGEPRC'
            }

            data = self._make_request(url, params, timeout=10)
            if data and 'marketdata' in data and data['marketdata']['data']:
                row = data['marketdata']['data'][0]
                cols = data['marketdata']['columns']

                result = {
                    'value': float(row[cols.index('CURRENTVALUE')]),
                    'change': float(row[cols.index('LASTCHANGE')]) if 'LASTCHANGE' in cols else 0,
                    'change_pct': float(row[cols.index('LASTCHANGEPRC')]) if 'LASTCHANGEPRC' in cols else 0,
                    'timestamp': datetime.now().isoformat()
                }

                self._save_to_cache(cache_key, result)
                logger.debug(f"📊 RVI получен: {result['value']} (изм: {result['change_pct']}%)")
                return result

            # Если данных нет, возвращаем дефолт из конфига
            return {'value': self.default_rvi, 'change': 0, 'change_pct': 0, 'timestamp': datetime.now().isoformat()}

        except Exception as e:
            logger.error(f"Ошибка получения RVI: {e}")
            return {'value': self.default_rvi, 'change': 0, 'change_pct': 0, 'timestamp': datetime.now().isoformat()}

    def get_market_indices(self) -> Dict[str, float]:
        """Получение индексов с изменениями"""
        cache_key = "market_indices"
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            logger.debug(f"📊 МАКРО ИЗ КЭША: IMOEX={cached.get('IMOEX')}")
            return cached

        result = {}

        url = f"{self.base_url}/engines/stock/markets/index/boards/SNDX/securities.json"
        params = {
            'iss.meta': 'off',
            'iss.only': 'marketdata',
            'marketdata.columns': 'SECID,CURRENTVALUE,LASTCHANGE,LASTCHANGEPRC'
        }

        data = self._make_request(url, params, timeout=10)
        if data and 'marketdata' in data:
            for row in data['marketdata']['data']:
                if len(row) < 4:
                    continue

                ticker = row[0]
                if ticker in self.index_tickers:
                    current = self._safe_float(row[1] if len(row) > 1 else None)
                    change_raw = row[2] if len(row) > 2 else None
                    change_pct_raw = row[3] if len(row) > 3 else None

                    result[ticker] = current
                    result[f"{ticker}_change"] = self._get_index_change(current, change_raw, change_pct_raw)
                    result[f"{ticker}_change_pct"] = self._safe_float(change_pct_raw)

        rtsi = self.get_rtsi()
        if rtsi:
            result['RTSI'] = rtsi['value']
            result['RTSI_change'] = rtsi['change']
            result['RTSI_change_pct'] = rtsi['change_pct']

        rvi = self.get_rvi()
        if rvi:
            result['RVI'] = rvi['value']
            result['RVI_change'] = rvi['change']
            result['RVI_change_pct'] = rvi['change_pct']

        result['update_time'] = datetime.now().isoformat()
        result['market_mood'] = self._calculate_market_mood(result)

        self._save_to_cache(cache_key, result)
        logger.info(f"📊 МАКРО: IMOEX={result.get('IMOEX')}, RTSI={result.get('RTSI')}, RVI={result.get('RVI')}, "
                    f"USD/RUB={self.get_usd_rub() or 0:.2f}, Brent={self.get_brent_price() or 0:.2f}, "
                    f"VIX={self.get_vix() or 0:.2f}, ЦБ={self._get_cbr_key_rate() or 0:.2f}%, "
                    f"Liq={self.get_market_liquidity_ratio():.4f}")
        return result

    def get_macro_data(self) -> Dict[str, float]:
        """Получение всех макро-данных"""
        indices = self.get_market_indices()

        macro_data = {
            'imoex': indices.get('IMOEX', 0.0),
            'imoex_change': indices.get('IMOEX_change', 0.0),
            'market_mood': indices.get('market_mood', 0.0),
            'rtsi': indices.get('RTSI', 0.0),
            'rtsi_change': indices.get('RTSI_change', 0.0),
            'rvi': indices.get('RVI', self.default_rvi),
            'rvi_change': indices.get('RVI_change', 0.0),
            'moexbmi': indices.get('MOEXBMI', 0.0),
            'moexfn': indices.get('MOEXFN', 0.0),
            'moexog': indices.get('MOEXOG', 0.0),
            'moextl': indices.get('MOEXTL', 0.0),
            'brent': self.get_brent_price() or 0.0,
            'brent_change': self.get_brent_change(),
            'usd_rub': self.get_usd_rub() or 0.0,
            'usd_rub_change': self.get_usd_rub_change(),
            'cbr_rate': self._get_cbr_key_rate() or 0.0,
            'vix': self.get_vix() or 0.0,
            'shares_turnover': self.get_shares_turnover(),
            'market_cap': self.get_market_capitalization(),
            'market_liquidity_ratio': self.get_market_liquidity_ratio(),
            'market_activity_score': self.get_market_activity_score(),
        }

        # logger.debug(f"📊 Макро-данные: IMOEX={macro_data['imoex']:.2f}, "
        #             f"USD/RUB={macro_data['usd_rub']:.2f}, "
        #             f"Brent={macro_data['brent']:.2f}, "
        #             f"VIX={macro_data['vix']:.2f}, "
        #             f"ЦБ ставка={macro_data['cbr_rate']:.2f}%, "
        #             f"liquidity={macro_data['market_liquidity_ratio']:.4f}")

        return macro_data

    def get_active_brent_contract(self) -> Optional[str]:
        """Определение активного фьючерсного контракта Brent"""
        cache_key = "active_brent_contract"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            limit = self.settings.get('moex_fetcher', {}).get('max_contracts', 500)
            url = f"{self.base_url}/engines/futures/markets/forts/securities.json"
            params = {
                'iss.meta': 'off',
                'iss.only': 'securities',
                'securities.columns': 'SECID,LASTTRADEDATE',
                'limit': limit
            }

            data = self._make_request(url, params, timeout=15)
            if data and 'securities' in data:
                cols = data['securities']['columns']
                rows = data['securities']['data']

                brent_contracts = []
                for row in rows:
                    secid = row[cols.index('SECID')]
                    last_trade = row[cols.index('LASTTRADEDATE')] if 'LASTTRADEDATE' in cols else None

                    if secid.startswith('BR-') and last_trade:
                        try:
                            last_trade_date = datetime.strptime(last_trade, '%Y-%m-%d')
                            brent_contracts.append({
                                'secid': secid,
                                'last_trade': last_trade_date
                            })
                        except:
                            continue

                if brent_contracts:
                    brent_contracts.sort(key=lambda x: x['last_trade'])
                    active = brent_contracts[0]['secid']
                    self._save_to_cache(cache_key, active)
                    logger.debug(f"Активный Brent контракт: {active}")
                    return active

            return None

        except Exception as e:
            logger.error(f"Ошибка определения активного Brent: {e}")
            return None

    def get_brent_price(self) -> Optional[float]:
        """Получение цены Brent (приоритет: MOEX → Investing.com → кэш)"""
        cache_key = "brent_price"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # 1. MOEX фьючерсы
        try:
            contract = self.get_active_brent_contract()
            if contract:
                url = f"{self.base_url}/engines/futures/markets/forts/securities/{contract}.json"
                params = {'iss.meta': 'off', 'iss.only': 'marketdata', 'marketdata.columns': 'SECID,LAST'}
                data = self._make_request(url, params, timeout=10)
                if data and 'marketdata' in data and data['marketdata']['data']:
                    row = data['marketdata']['data'][0]
                    cols = data['marketdata']['columns']
                    if 'LAST' in cols:
                        idx = cols.index('LAST')
                        if idx < len(row) and row[idx] is not None:
                            price = self._safe_float(row[idx])
                            if price > 0:
                                self._save_to_cache(cache_key, price)
                                return price
        except Exception as e:
            logger.debug(f"Brent MOEX недоступен: {e}")

        # 2. Investing.com
        try:
            import re
            resp = self.session.get(
                "https://ru.investing.com/commodities/brent-oil",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=10
            )
            if resp.status_code == 200:
                match = re.search(r'data-test="instrument-price-last">\s*([\d.,]+)', resp.text)
                if not match:
                    match = re.search(r'"last":([\d.]+)', resp.text)
                if match:
                    raw = match.group(1).replace('.', '').replace(',', '.')
                    price = float(raw)
                    if price > 0:
                        self._save_to_cache(cache_key, price)
                        logger.debug(f"Brent (Investing.com): {price:.2f}")
                        return price
        except Exception as e:
            logger.debug(f"Brent Investing.com недоступен: {e}")

        # 3. Устаревший кэш
        if cache_key in self.cache:
            cached_data, _ = self.cache[cache_key]
            logger.debug(f"Brent из устаревшего кэша: {cached_data:.2f}")
            return cached_data

        return None

    def get_brent_change(self) -> float:
        """Получение изменения цены Brent через Investing.com"""
        cache_key = "brent_change"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            import re
            resp = self.session.get(
                "https://ru.investing.com/commodities/brent-oil",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=10
            )
            if resp.status_code == 200:
                # Ищем процент изменения
                match = re.search(r'data-test="instrument-price-change-percent">\s*([+\-\d.,]+)%', resp.text)
                if not match:
                    match = re.search(r'"changePercent":([\d.\-]+)', resp.text)
                if match:
                    raw = match.group(1).replace(',', '.')
                    change = float(raw)
                    if change != 0:
                        self._save_to_cache(cache_key, change)
                        logger.debug(f"Brent change (Investing.com): {change:+.2f}%")
                        return change
        except Exception as e:
            logger.debug(f"Brent change Investing.com недоступен: {e}")

        # Устаревший кэш
        if cache_key in self.cache:
            cached_data, _ = self.cache[cache_key]
            return cached_data

        return 0.0

    def get_vix(self) -> Optional[float]:
        """Получение индекса волатильности (Investing.com → MOEX RVI)"""
        cache_key = "vix"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # 1. Investing.com
        try:
            import re
            resp = self.session.get(
                "https://ru.investing.com/indices/volatility-s-p-500",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=10
            )
            if resp.status_code == 200:
                match = re.search(r'data-test="instrument-price-last">\s*([\d.,]+)', resp.text)
                if not match:
                    match = re.search(r'"last":([\d.]+)', resp.text)
                if match:
                    raw = match.group(1).replace('.', '').replace(',', '.')
                    value = float(raw)
                    if value > 0:
                        self._save_to_cache(cache_key, value)
                        logger.debug(f"VIX (Investing.com): {value:.2f}")
                        return value
        except Exception as e:
            logger.debug(f"VIX Investing.com недоступен: {e}")

        # 2. MOEX RVI как запасной вариант
        rvi_data = self.get_rvi()
        if rvi_data:
            rvi_value = rvi_data.get('value', self.default_rvi)
            self._save_to_cache(cache_key, rvi_value)
            logger.debug(f"VIX через RVI MOEX: {rvi_value:.2f}")
            return rvi_value

        return self.default_rvi

    def get_usd_rub(self) -> Optional[float]:
        """Получение курса USD/RUB (приоритет: MOEX → ЦБ РФ)"""
        cache_key = "usd_rub"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # 1. Пробуем MOEX SELT
        try:
            url = f"{self.base_url}/engines/currency/markets/selt/securities/USD000UTSTOM.json"
            params = {
                'iss.meta': 'off',
                'iss.only': 'marketdata',
                'marketdata.columns': 'SECID,LAST'
            }

            data = self._make_request(url, params, timeout=10)
            if data and 'marketdata' in data and data['marketdata']['data']:
                row = data['marketdata']['data'][0]
                cols = data['marketdata']['columns']

                if 'LAST' in cols:
                    idx = cols.index('LAST')
                    if idx < len(row) and row[idx] is not None:
                        value = self._safe_float(row[idx])
                        if value > 0:
                            self._save_to_cache(cache_key, value)
                            logger.debug(f"USD/RUB (MOEX): {value:.2f}")
                            return value

        except Exception as e:
            logger.debug(f"USD/RUB MOEX недоступен: {e}")

        # 2. Fallback: ЦБ РФ
        cbr_value = self._get_cbr_usd_rub()
        if cbr_value is not None and cbr_value > 0:
            self._save_to_cache(cache_key, cbr_value)
            logger.debug(f"USD/RUB (ЦБ РФ): {cbr_value:.2f}")
            return cbr_value

        logger.debug("USD/RUB не доступен (ни MOEX, ни ЦБ)")
        return None

    def get_shares_turnover(self) -> float:
        """Получение оборота рынка акций"""
        cache_key = "shares_turnover"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            url = f"{self.base_url}/engines/stock/turnovers.json"
            params = {'iss.meta': 'off', 'iss.only': 'turnovers'}

            data = self._make_request(url, params, timeout=10)
            if data and 'turnovers' in data:
                cols = data['turnovers']['columns']
                for row in data['turnovers']['data']:
                    if row[cols.index('NAME')] == 'shares':
                        turnover = self._safe_float(row[cols.index('VALTODAY')])
                        self._save_to_cache(cache_key, turnover)
                        return turnover
            return 0.0
        except Exception as e:
            logger.error(f"Ошибка получения оборота: {e}")
            return 0.0

    def get_market_capitalization(self) -> float:
        """Получение рыночной капитализации"""
        cache_key = "market_cap"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            url = f"{self.base_url}/statistics/engines/stock/capitalization.json"
            params = {'iss.meta': 'off', 'iss.only': 'capitalization'}

            data = self._make_request(url, params, timeout=10)
            if data and 'capitalization' in data and data['capitalization']['data']:
                row = data['capitalization']['data'][0]
                cols = data['capitalization']['columns']
                cap = self._safe_float(row[cols.index('CAPITALIZATION')])
                self._save_to_cache(cache_key, cap)
                return cap
            return 0.0
        except Exception as e:
            logger.error(f"Ошибка получения капитализации: {e}")
            return 0.0

    def get_market_liquidity_ratio(self) -> float:
        """Коэффициент ликвидности рынка (адаптивная нормализация)"""
        cache_key = "liquidity_ratio"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # API MOEX возвращает оборот в миллионах рублей — переводим в рубли
        turnover_millions = self.get_shares_turnover()
        turnover_rub = turnover_millions * 1_000_000.0

        market_cap = self.get_market_capitalization()

        if market_cap <= 0:
            return 0.0

        # Сырое отношение оборота к капитализации
        raw_ratio = turnover_rub / market_cap

        # Накапливаем историю
        history_size = self.settings.get('market_data', {}).get('liquidity_history_size', 20)
        self.liquidity_history.append(raw_ratio)
        if len(self.liquidity_history) > history_size:
            self.liquidity_history = self.liquidity_history[-history_size:]

        # Адаптивная нормализация: процентиль текущего значения относительно истории
        if len(self.liquidity_history) >= 3:
            min_hist = min(self.liquidity_history)
            max_hist = max(self.liquidity_history)
            if max_hist > min_hist:
                ratio = (raw_ratio - min_hist) / (max_hist - min_hist)
                ratio = max(0.0, min(1.0, ratio))
            else:
                ratio = 0.5
        else:
            # Мало истории — используем фиксированный масштаб
            ratio = min(raw_ratio * 1000.0, 1.0)

        self._save_to_cache(cache_key, ratio)
        return ratio

    def get_market_activity_score(self) -> float:
        """Активность рынка"""
        cache_key = "market_activity"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        turnover = self.get_shares_turnover()
        if turnover == 0:
            return 0.0

        avg_key = "avg_turnover_last_hour"
        avg_cached = self._get_from_cache(avg_key)

        if avg_cached is None:
            self._save_to_cache(cache_key, 1.0)
            return 1.0

        score = turnover / max(avg_cached, 1.0)
        score = min(score, 2.0)
        self._save_to_cache(cache_key, score)
        return score



    def get_usd_rub_change(self) -> float:
        """Получение изменения USD/RUB"""
        cache_key = "usd_rub_change"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            url = f"{self.base_url}/engines/currency/markets/selt/securities/USD000UTSTOM.json"
            params = {
                'iss.meta': 'off',
                'iss.only': 'marketdata',
                'marketdata.columns': 'SECID,CHANGE,CHANGEPRC'
            }

            data = self._make_request(url, params, timeout=10)
            if data and 'marketdata' in data and data['marketdata']['data']:
                row = data['marketdata']['data'][0]
                cols = data['marketdata']['columns']

                if 'CHANGE' in cols:
                    idx = cols.index('CHANGE')
                    if idx < len(row) and row[idx] is not None:
                        change = self._safe_float(row[idx])
                        self._save_to_cache(cache_key, change)
                        return change

            return 0.0

        except Exception as e:
            logger.error(f"Ошибка получения изменения USD/RUB: {e}")
            return 0.0

    def _get_cbr_usd_rub(self) -> Optional[float]:
        """Получение курса USD/RUB с сайта ЦБ РФ"""
        cache_key = "cbr_usd_rub"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            url = "http://www.cbr.ru/scripts/XML_daily.asp"
            response = self.session.get(url, timeout=10)
            response.encoding = 'windows-1251'
            root = ET.fromstring(response.text)

            for valute in root.findall('Valute'):
                char_code = valute.find('CharCode').text
                if char_code == 'USD':
                    value = valute.find('Value').text
                    nominal = valute.find('Nominal').text
                    usd_rub = float(value.replace(',', '.')) / float(nominal)
                    self._save_to_cache(cache_key, usd_rub)
                    return usd_rub

            logger.debug("USD не найден в ответе ЦБ РФ")
            return None

        except Exception as e:
            logger.error(f"Ошибка получения USD/RUB с ЦБ: {e}")
            return None

    def _get_cbr_key_rate(self) -> Optional[float]:
        """Получение ключевой ставки ЦБ РФ"""
        cache_key = "cbr_key_rate"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            date_from = (datetime.now() - timedelta(days=5)).strftime('%d/%m/%Y')
            date_to = datetime.now().strftime('%d/%m/%Y')

            url = "http://www.cbr.ru/scripts/XML_depo.asp"
            params = {'date_req1': date_from, 'date_req2': date_to}

            response = self.session.get(url, params=params, timeout=10)
            response.encoding = 'windows-1251'
            root = ET.fromstring(response.text)

            latest = root.find('Record')
            if latest is not None:
                overnight = latest.find('Overnight')
                if overnight is not None:
                    rate = float(overnight.text.replace(',', '.'))
                    self._save_to_cache(cache_key, rate)
                    return rate

            logger.debug("Ключевая ставка не найдена в ответе ЦБ РФ")
            return None

        except Exception as e:
            logger.error(f"Ошибка получения ключевой ставки ЦБ: {e}")
            return None


    def _calculate_market_mood(self, indices: Dict[str, float]) -> float:
        """Расчет общего настроения рынка на основе индексов"""
        try:
            imoex = indices.get('IMOEX', 0)
            if imoex <= 0:
                return 0.0

            cache_key = "prev_imoex"
            prev_imoex = self._get_from_cache(cache_key)

            if prev_imoex and prev_imoex > 0:
                change = ((imoex / prev_imoex) - 1) * 100
                mood_change_threshold = self.settings.get('market_data', {}).get('market_mood_change_threshold', 5.0)
                mood = max(-1.0, min(1.0, change / mood_change_threshold))
            else:
                mood = 0.0

            self._save_to_cache(cache_key, imoex)
            return mood

        except Exception as e:
            logger.error(f"Ошибка расчета market mood: {e}")
            return 0.0

    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Получение полной информации по тикеру"""
        try:
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

            price = self.get_price(ticker)
            if price:
                info['current_price'] = price

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
            url = f"{self.base_url}/engines.json"
            params = {'iss.meta': 'off', 'limit': 1}

            data = self._make_request(url, params, timeout=15)

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