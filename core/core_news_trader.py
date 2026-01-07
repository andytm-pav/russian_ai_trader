"""
Ядро новостной торговли - анализ RSS и генерация торговых сигналов
"""

import json
import time
import feedparser
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re

from core.core_technical_trader import TechnicalTraderCore
from utils.logger import setup_logger
from models.trader_model import trader_model_instance

logger = setup_logger("NEWS_CORE")


class NewsTraderCore:
    """Ядро для торговли на основе новостей из RSS"""

    def __init__(self, config_path: str = "config/rss_sources.json"):
        self.config = self._load_config(config_path)
        self.model = trader_model_instance
        self.technical_core = TechnicalTraderCore()
        self.last_fetch_time = 0
        self.news_cache = defaultdict(list)
        self.running = False
        self.fetch_thread = None

        logger.info(f"Инициализировано ядро новостной торговли с {len(self.config['sources'])} источниками")

    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации RSS"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации RSS: {e}")
            return {
                "sources": [],
                "update_interval_minutes": 5,
                "max_news_per_source": 20
            }

    def start_continuous_fetching(self):
        """Запуск непрерывного сбора новостей в отдельном потоке"""
        if self.running:
            return

        self.running = True
        self.fetch_thread = threading.Thread(
            target=self._fetch_loop,
            daemon=True,
            name = "NewsFetcherThread"
        )
        self.fetch_thread.start()
        logger.info("Запущен непрерывный сбор новостей")
        # +++ ДЕТАЛИ О ИНТЕРВАЛЕ +++
        interval = self.config.get('update_interval_minutes', 5)
        logger.debug(f"Интервал обновления: {interval} минут ({interval * 60} секунд)")

    def stop_continuous_fetching(self):
        """Остановка сбора новостей"""
        self.running = False
        if self.fetch_thread:
            self.fetch_thread.join(timeout=5)
        logger.info("Сбор новостей остановлен")

    def _fetch_loop(self):
        """Цикл непрерывного сбора новостей с реализацией кэширования"""
        # Получаем интервал обновления из конфига (в секундах)
        # По умолчанию 5 минут = 300 секунд между циклами
        update_interval = self.config.get('update_interval_minutes', 5) * 60

        # Переменная для отслеживания количества выполненных циклов
        # Помогает при отладке и логировании периодических событий
        cycle_count = 0

        # Основной рабочий цикл - выполняется пока self.running = True
        # Флаг running изменяется методами start/stop_continuous_fetching()
        while self.running:
            try:
                cycle_count += 1
                logger.debug(f"Начало цикла сбора новостей #{cycle_count}")

                # 1. СБОР НОВОСТЕЙ СО ВСЕХ АКТИВНЫХ ИСТОЧНИКОВ
                # Метод fetch_all_news() возвращает словарь вида:
                # {'Название источника': [список_новостей], ...}
                all_news = self.fetch_all_news()

                # 2. ОБНОВЛЕНИЕ КЭША НОВОСТЕЙ (КЛЮЧЕВОЕ ИЗМЕНЕНИЕ)
                # В оригинале news_cache объявлен, но никогда не заполняется
                # Здесь мы наполняем его реальными данными
                total_news_collected = 0

                # Проходим по всем собранным новостям
                for source_name, news_list in all_news.items():
                    if news_list:  # Если для источника есть новости
                        # Добавляем новости в кэш для этого источника
                        # extend() добавляет элементы списка в конец существующего списка
                        self.news_cache[source_name].extend(news_list)

                        # 3. ОГРАНИЧЕНИЕ РАЗМЕРА КЭША ДЛЯ КАЖДОГО ИСТОЧНИКА
                        # Предотвращает неограниченный рост потребления памяти
                        max_cache_size = self.config.get('max_news_per_source', 20)

                        # Оставляем только последние N новостей (самые свежие)
                        # Срезы в Python безопасны при превышении индекса
                        self.news_cache[source_name] = self.news_cache[source_name][-max_cache_size:]

                        total_news_collected += len(news_list)

                # Обновляем время последнего успешного сбора
                # Используется в get_news_summary() для отображения статуса
                self.last_fetch_time = time.time()

                # 4. ЛОГИРОВАНИЕ РЕЗУЛЬТАТОВ ЦИКЛА
                # Вычисляем общее количество новостей в кэше после обновления
                total_in_cache = sum(len(items) for items in self.news_cache.values())

                logger.debug(
                    f"Цикл #{cycle_count}: собрано {total_news_collected} новостей, "
                    f"в кэше {total_in_cache} новостей из {len(self.news_cache)} источников"
                )

                # 5. ПЕРИОДИЧЕСКОЕ ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ (каждые 10 циклов)
                # Помогает мониторить состояние системы без избыточных логов
                if cycle_count % 10 == 0:
                    # Используем существующий метод get_news_summary()
                    summary = self.get_news_summary()
                    logger.info(
                        f"Статус кэша новостей (цикл #{cycle_count}): "
                        f"{summary['cache_size']} новостей из {summary['sources_count']} источников"
                    )

                # 6. ОЖИДАНИЕ СЛЕДУЮЩЕГО ЦИКЛА СБОРА
                # time.sleep() приостанавливает выполнение потока на указанное время
                # Это позволяет соблюдать интервал между запросами к RSS-источникам
                time.sleep(update_interval)

            except Exception as e:
                # ОБРАБОТКА ОШИБОК В РАБОЧЕМ ЦИКЛЕ
                # Важно перехватывать все исключения, чтобы поток не завершался аварийно
                logger.error(f"Ошибка в цикле сбора новостей: {e}")

                # При ошибке ждем 60 секунд перед следующей попыткой
                # Это дает время на восстановление сети или сервисов RSS
                time.sleep(60)

    def fetch_all_news(self) -> Dict[str, List[Dict]]:
        """Сбор новостей со всех источников"""
        all_news = {}

        for source in self.config['sources']:
            if not source.get('enabled', True):
                continue

            try:
                news = self._fetch_source_news(source)
                if news:
                    all_news[source['name']] = news
                    logger.debug(f"Источник {source['name']}: собрано {len(news)} новостей")
            except Exception as e:
                logger.error(f"Ошибка сбора новостей из {source['name']}: {e}")

        self.last_fetch_time = time.time()
        return all_news

    def _fetch_source_news(self, source: Dict) -> List[Dict]:
        """Сбор новостей из одного источника"""
        try:
            feed = feedparser.parse(source['url'])
            news_items = []

            max_news = self.config.get('max_news_per_source', 20)

            for entry in feed.entries[:max_news]:
                # Пропускаем старые новости
                published = self._parse_date(entry.get('published', ''))
                if published and (time.time() - published.timestamp()) > 86400:  # Старше суток
                    continue

                news_item = {
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': published.isoformat() if published else '',
                    'source': source['name'],
                    'category': source.get('category', 'general'),
                    'priority': source.get('priority', 5)
                }

                # Фильтрация по ключевым словам
                if self._filter_news(news_item):
                    news_items.append(news_item)

            return news_items

        except Exception as e:
            logger.error(f"Ошибка парсинга RSS {source['url']}: {e}")
            return []

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Парсинг даты из различных форматов"""
        try:
            # Пробуем разные форматы
            formats = [
                '%a, %d %b %Y %H:%M:%S %z',
                '%a, %d %b %Y %H:%M:%S %Z',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S'
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            return datetime.now()
        except:
            return datetime.now()

    def _filter_news(self, news_item: Dict) -> bool:
        """Фильтрация новостей по ключевым словам"""
        keywords = self.config.get('keywords_filter', {})
        include = keywords.get('include', [])
        exclude = keywords.get('exclude', [])

        text = f"{news_item['title']} {news_item['summary']}".lower()

        # Проверяем исключающие слова
        for word in exclude:
            if word.lower() in text:
                return False

        # Если есть включающие слова, проверяем их
        if include:
            for word in include:
                if word.lower() in text:
                    return True
            return False

        return True

    def analyze_news_sentiment(self, news_items: List[Dict]) -> Dict[str, float]:
        """Анализ тональности новостей"""
        sentiment_dict = self.config.get('sentiment_dictionary', {})
        positive_words = set(word.lower() for word in sentiment_dict.get('positive', []))
        negative_words = set(word.lower() for word in sentiment_dict.get('negative', []))

        ticker_sentiments = defaultdict(list)

        # Загружаем список тикеров для поиска
        try:
            with open('config/tickers.json', 'r', encoding='utf-8') as f:
                tickers_config = json.load(f)
                tickers = [item['ticker'] for item in tickers_config.get('watchlist', [])]
        except:
            tickers = []

        for news in news_items:
            text = f"{news['title']} {news['summary']}".lower()

            # Находим упоминания тикеров
            mentioned_tickers = []
            for ticker in tickers:
                if re.search(rf'\b{ticker}\b', text, re.IGNORECASE):
                    mentioned_tickers.append(ticker)

            # Анализ тональности
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                sentiment = 0.0

            # Учитываем приоритет источника
            sentiment *= (news.get('priority', 5) / 10.0)

            # Распределяем сентимент по тикерам
            if mentioned_tickers:
                for ticker in mentioned_tickers:
                    ticker_sentiments[ticker].append(sentiment)
            else:
                # Если тикеры не упомянуты, добавляем в общий рыночный
                ticker_sentiments['MARKET'].append(sentiment)

        # Усредняем сентименты
        result = {}
        for ticker, sentiments in ticker_sentiments.items():
            if sentiments:
                result[ticker] = sum(sentiments) / len(sentiments)

        return result

    def generate_trading_signals(self, prices: Dict[str, float]) -> List[Dict]:
        """Генерация торговых сигналов на основе новостей"""
        try:
            # 1. ПОЛУЧАЕМ НОВОСТИ ИЗ КЭША вместо новых HTTP-запросов
            # Проверяем, есть ли данные в кэше
            news_items = []
            cache_used = True  # Флаг для отслеживания источника данных

            # Проходим по всем источникам в кэше
            for source_name, cached_news in self.news_cache.items():
                if cached_news:  # Если для источника есть кэшированные новости
                    news_items.extend(cached_news)

            # 2. ЕСЛИ КЭШ ПУСТ - делаем разовый сбор новостей
            # Это fallback-механизм на случай, если цикл сбора еще не успел наполнить кэш
            if not news_items:
                logger.info("Кэш новостей пуст, выполняется разовый сбор...")
                cache_used = False

                # Используем существующий метод для сбора
                all_news = self.fetch_all_news()

                # Сохраняем собранные новости в кэш для будущих циклов
                for source_name, news_list in all_news.items():
                    if news_list:
                        self.news_cache[source_name].extend(news_list)
                        # Ограничиваем размер кэша (как в _fetch_loop)
                        max_cache_size = self.config.get('max_news_per_source', 20)
                        self.news_cache[source_name] = self.news_cache[source_name][-max_cache_size:]
                        news_items.extend(news_list)

                # Обновляем время последнего сбора
                self.last_fetch_time = time.time()

            # 3. ЛОГИРОВАНИЕ ИСТОЧНИКА ДАННЫХ
            # Помогает в отладке и мониторинге производительности
            logger.debug(f"Генерация сигналов: использован {'КЭШ' if cache_used else 'разовый сбор'}, "
                         f"{len(news_items)} новостей")

            # 4. ДАЛЬНЕЙШАЯ ОБРАБОТКА БЕЗ ИЗМЕНЕНИЙ
            # Существующая логика работает с news_items как и раньше
            if not news_items:
                logger.debug("Нет новостей для анализа")
                return []

            # Анализ тональности (существующий метод)
            sentiments = self.analyze_news_sentiment(news_items)

            # Генерация сигналов (существующая логика)
            signals = []
            market_sentiment = sentiments.get('MARKET', 0.0)

            for ticker, price in prices.items():
                if ticker == 'MARKET':
                    continue


                ticker_sentiment = sentiments.get(ticker, market_sentiment)

                # Ищем новости, связанные с текущим тикером
                news_texts = []
                for news in news_items:
                    news_text = f"{news.get('title', '')} {news.get('summary', '')}"
                    # Простой поиск тикера в тексте (можно улучшить)
                    if ticker.lower() in news_text.lower():
                        news_texts.append(news_text)

                if news_texts:
                    # Используем модель для анализа (существующий код)
                    news_features = self.model.encode_news(news_texts[:3])

                    # 1. Получаем технические индикаторы для тикера
                    indicators = self.technical_core.calculate_indicators(ticker)

                    # 2. Рассчитываем реальный моментум
                    momentum = indicators.get('momentum', 0.0)  # ✅ Берем из индикаторов

                    # 3. Подготавливаем рыночные данные
                    market_data = {
                        'volume': indicators.get('volume_ratio', 1.0),
                        'spread': 0.01,
                        'liquidity': 0.5,
                        'rsi': indicators.get('rsi', 50),
                        'volatility': indicators.get('atr', 0) / price if price > 0 else 0.1,
                        'sma_10_ratio': indicators.get('sma_10', price) / price if price > 0 else 1.0,
                        'sma_20_ratio': indicators.get('sma_20', price) / price if price > 0 else 1.0,
                        'bb_position': self._calculate_bb_position(price, indicators),
                        'market_cap': 0,
                        'pe_ratio': 15
                    }

                    # 4. ИСПРАВЛЕННЫЙ вызов build_state_vector
                    state = self.model.build_state_vector(
                        ticker=ticker,
                        price=price,
                        momentum=momentum,  # ✅ Теперь не 0.0
                        sentiment=ticker_sentiment,
                        news_features=news_features,
                        market_data=market_data  # ✅ Теперь не {'volume': 0, 'spread': 0.01}
                    )

                    # Выбираем действие через модель
                    action, confidence, _ = self.model.choose_action(state, ticker, price)

                    # Генерация сигнала при высокой уверенности
                    if confidence > 0.7:
                        signal = {
                            'ticker': ticker,
                            'action': ['BUY', 'HOLD', 'SELL'][action],
                            'confidence': float(confidence),
                            'sentiment': float(ticker_sentiment),
                            'price': float(price),
                            'reason': 'news_analysis',
                            'timestamp': datetime.now().isoformat(),
                            'news_count': len(news_texts),
                            'data_source': 'cache' if cache_used else 'direct_fetch',
                            'momentum': float(momentum)  # ✅ ДОБАВЛЯЕМ momentum в сигнал
                        }

                        signals.append(signal)
                        logger.info(f"Сгенерирован сигнал: {ticker} {signal['action']} "
                                    f"(conf={confidence:.2f}, sent={ticker_sentiment:.2f}, mom={momentum:.2f})")

            return signals

        except Exception as e:
            logger.error(f"Ошибка генерации сигналов: {e}")
            return []

    def get_news_summary(self) -> Dict:
        """Получение сводки новостей"""
        # ВЫЧИСЛЯЕМ РЕАЛЬНЫЙ РАЗМЕР КЭША
        # sum() проходит по всем источникам и суммирует количество новостей
        total_cached_news = sum(len(v) for v in self.news_cache.values())

        # СЧИТАЕМ АКТИВНЫЕ ИСТОЧНИКИ
        # Фильтруем источники с enabled=true или без этого флага (по умолчанию включены)
        active_sources = len([
            s for s in self.config['sources']
            if s.get('enabled', True)
        ])

        # СЧИТАЕМ ИСТОЧНИКИ С ДАННЫМИ В КЭШЕ
        sources_with_data = len([
            name for name, news_list in self.news_cache.items()
            if news_list  # Источник считается активным в кэше, если есть хотя бы одна новость
        ])

        return {
            # Время последнего обновления (форматированное из timestamp)
            'last_fetch': datetime.fromtimestamp(self.last_fetch_time).isoformat()
            if self.last_fetch_time else None,

            # Количество активных источников из конфига
            'sources_count': active_sources,

            # РЕАЛЬНЫЙ РАЗМЕР КЭША вместо фиктивного значения
            'cache_size': total_cached_news,

            # ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ для мониторинга
            'sources_with_data': sources_with_data,  # Источники, которые реально дали данные
            'avg_news_per_source': total_cached_news / max(sources_with_data, 1),  # Среднее количество новостей
            'cache_status': 'active' if self.running else 'stopped',  # Статус фонового сбора

            # Сохраняем обратную совместимость - если где-то ожидают старые ключи
            'last_update': datetime.now().isoformat()  # Дополнительное поле для временных меток
        }

    def _calculate_bb_position(self, price: float, indicators: Dict) -> float:
        """Расчет позиции в Bollinger Bands"""
        bb_upper = indicators.get('bb_upper', price * 1.1)  # Дефолт: +10%
        bb_lower = indicators.get('bb_lower', price * 0.9)  # Дефолт: -10%

        if bb_upper > bb_lower:
            return (price - bb_lower) / (bb_upper - bb_lower)
        return 0.5  # По умолчанию середина