"""
Получение данных с Московской биржи (MOEX) - оптимизированная версия
"""

import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

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

        # Кэширование из конфига
        self.use_cache = use_cache if use_cache is not None else moex_config.get('use_cache', True)
        self.cache_ttl = cache_ttl if cache_ttl is not None else moex_config.get('cache_ttl', 300)
        self.max_requests_per_minute = moex_config.get('max_requests_per_minute', 50)
        self.request_timestamps = []
        self.cache = {}

        # Статистика кэша
        self.cache_hits = 0
        self.cache_misses = 0

        # Для макро-данных
        self.last_indices = {}
        self.rvi_cache = {}

        self.default_rvi = self.settings.get('market_data', {}).get('default_rvi', 20.0)
        self.rvi_ticker = self.settings.get('market_data', {}).get('rvi_ticker', 'RVI')
        self.rvi_board = self.settings.get('market_data', {}).get('rvi_board', 'OPTN')
        self.liquidity_threshold = self.settings.get('market_data', {}).get('liquidity_threshold', 1000000000)
        self.market_mood_threshold = self.settings.get('market_data', {}).get('market_mood_change_threshold', 5.0)
        self.index_tickers = self.settings.get('market_data', {}).get('index_tickers',
                                                                      ['IMOEX', 'RTSI', 'MOEXBMI', 'MOEXFN', 'MOEXOG',
                                                                       'MOEXTL'])

        logger.info(
            f"Инициализирован MOEX Fetcher (кэш: {self.cache_ttl}с, лимит: {self.max_requests_per_minute} запр/мин)")

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
                    'marketdata.columns': 'SECID,LAST,VALTODAY,VOLTODAY,SPREAD,ISSUECAPITALIZATION',
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
                        'volume': volume,  # ✅ РЕАЛЬНЫЕ ДАННЫЕ
                        'change': current_price - prev_price if prev_price else 0,
                        'momentum': momentum,
                        'liquidity': liquidity,  # ✅ РАССЧИТАНО ИЗ РЕАЛЬНЫХ ДАННЫХ
                        'spread': market_data.get('spread', 0.0),  # ✅ РЕАЛЬНЫЕ ДАННЫЕ
                        'market_cap': market_data.get('market_cap', 0.0),  # ✅ РЕАЛЬНЫЕ ДАННЫЕ
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

    def get_market_indices(self) -> Dict[str, float]:
        """Получение индексов с изменениями"""
        cache_key = "market_indices"
        cached = self._get_from_cache(cache_key)

        if cached is not None:
            logger.debug(f"📊 МАКРО ИЗ КЭША: IMOEX={cached.get('IMOEX')}, RVI={cached.get('RVI')}")
            return cached

        # Используем список тикеров из конфига
        current_prices = {}
        for ticker in self.index_tickers:
            price = self.get_price(ticker)
            if price:
                current_prices[ticker] = price

        # Рассчитываем изменения относительно предыдущих значений
        result = {}
        for ticker, price in current_prices.items():
            result[ticker] = price

            # Изменение за день
            if ticker in self.last_indices:
                prev = self.last_indices[ticker].get('price', price)
                change = ((price / prev) - 1) * 100
                result[f"{ticker}_change"] = change
            else:
                result[f"{ticker}_change"] = 0.0

        # Добавляем RVI (индекс волатильности)
        rvi = self.get_rvi()
        if rvi:
            result['RVI'] = rvi['value']
            result['RVI_change'] = rvi['change']

        # Сохраняем текущие для следующего раза
        self.last_indices = {
            ticker: {'price': price, 'timestamp': datetime.now()}
            for ticker, price in current_prices.items()
        }

        result['update_time'] = datetime.now().isoformat()
        result['market_mood'] = self._calculate_market_mood(current_prices)

        self._save_to_cache(cache_key, result)

        logger.info(f"📊 МАКРО СВЕЖИЕ: IMOEX={result.get('IMOEX')}, RVI={result.get('RVI')}")
        self._save_to_cache(cache_key, result)

        return result

    def get_rvi(self) -> Optional[Dict[str, float]]:
        """
        Получение индекса волатильности RVI
        """
        cache_key = "rvi"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        try:
            # Используем ticker и board из конфига
            url = f"{self.base_url}/engines/stock/markets/options/boards/{self.rvi_board}/securities/{self.rvi_ticker}.json"
            params = {'iss.meta': 'off'}

            data = self._make_request(url, params, timeout=10)
            if data and 'marketdata' in data:
                columns = data['marketdata']['columns']
                rows = data['marketdata']['data']

                if rows:
                    row = rows[0]
                    price_idx = columns.index('LAST') if 'LAST' in columns else -1
                    change_idx = columns.index('CHANGE') if 'CHANGE' in columns else -1

                    result = {
                        'value': float(row[price_idx]) if price_idx >= 0 else self.default_rvi,
                        'change': float(row[change_idx]) if change_idx >= 0 else 0,
                        'timestamp': datetime.now().isoformat()
                    }

                    self._save_to_cache(cache_key, result)
                    return result

            # Если данных нет, возвращаем дефолт из конфига
            return {'value': self.default_rvi, 'change': 0, 'timestamp': datetime.now().isoformat()}

        except Exception as e:
            logger.error(f"Ошибка получения RVI: {e}")
            return {'value': self.default_rvi, 'change': 0, 'timestamp': datetime.now().isoformat()}

    def get_macro_data(self) -> Dict[str, float]:
        """
        Получение всех макро-данных для заполнения резерва
        Возвращает словарь с ключами:
        - imoex, imoex_change
        - rtsi, rtsi_change
        - rvi, rvi_change
        """
        macro_data = {}

        # Получаем индексы
        indices = self.get_market_indices()

        # IMOEX
        macro_data['imoex'] = indices.get('IMOEX', 0.0)
        macro_data['imoex_change'] = indices.get('IMOEX_change', 0.0)

        # RTSI
        macro_data['rtsi'] = indices.get('RTSI', 0.0)
        macro_data['rtsi_change'] = indices.get('RTSI_change', 0.0)

        # RVI (волатильность)
        macro_data['rvi'] = indices.get('RVI', 20.0)  # дефолт 20 если нет
        macro_data['rvi_change'] = indices.get('RVI_change', 0.0)

        # ✅ ДОБАВИТЬ ЛОГ
        logger.debug(f"📊 Макро-данные: IMOEX={macro_data['imoex']:.2f} "
                     f"(изм:{macro_data['imoex_change']:+.2f}%), "
                     f"RVI={macro_data['rvi']:.2f}")

        return macro_data

    def _calculate_market_mood(self, indices: Dict[str, float]) -> float:
        """Расчет общего настроения рынка на основе индексов"""
        try:
            if not indices:
                return 0.0

            # Получаем IMOEX (основной индекс)
            imoex = indices.get('IMOEX', 0)
            if imoex <= 0:
                return 0.0

            # Сравниваем с предыдущим значением (можно получать из кэша)
            cache_key = "prev_imoex"
            prev_imoex = self._get_from_cache(cache_key)

            if prev_imoex and prev_imoex > 0:
                change = ((imoex / prev_imoex) - 1) * 100
                # Mood от -1 до 1 на основе изменения
                # ПОЛУЧАЕМ ПОРОГ ИЗ КОНФИГА
                mood_change_threshold = self.settings.get('market_data', {}).get('market_mood_change_threshold', 5.0)
                mood = max(-1.0, min(1.0, change / mood_change_threshold))
            else:
                mood = 0.0

            # Сохраняем текущее значение
            self._save_to_cache(cache_key, imoex)

            return mood

        except Exception as e:
            logger.error(f"Ошибка расчета market mood: {e}")
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