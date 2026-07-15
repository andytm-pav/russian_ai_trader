import json
import time
import hashlib
import re
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path
from utils.logger import get_logger

try:
    import ahocorasick

    AHOCORASICK_AVAILABLE = True
except ImportError:
    AHOCORASICK_AVAILABLE = False

try:
    from transformers import pipeline
    import torch

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================
# 🆕 Context-aware sentiment overrides для российского рынка
# ============================================================
# Сильные позитивные паттерны: даже если ML-модель сказала NEGATIVE,
# эти фразы должны «перевернуть» сентимент в POSITIVE (или хотя бы в NEUTRAL).
STRONG_POSITIVE_PATTERNS = [
    # Прямые оценки привлекательности
    r'сохраня[юет]+\s+привлекательность',
    r'сохраня[юет]+\s+инвестиционн',
    r'выглядят\s+привлекательн',
    r'выглядит\s+привлекательн',
    r'остают[а-я]+\s+привлекательн',
    r'привлекательн[а-я]+\s+для\s+покупк',
    r'привлекательн[а-я]+\s+для\s+инвестици',
    r'привлекательн[а-я]+\s+на\s+горизонт',
    r'высока[яе]+\s+привлекательность',
    # Рост / потенциал
    r'потенциал\s+роста',
    r'потенциал\s+прибавить',
    r'целевая\s+цена\s+(?:выше|раст)',
    r'целевой\s+уровень\s+выше',
    r'апсайд',
    r'бычий\s+(?:прогноз|сценарий|настро)',
    r'позитивн[а-я]+\s+динамик',
    r'позитивн[а-я]+\s+перспектив',
    r'устойчив[а-я]+\s+рост',
    r'продолж[а-я]+\s+рости',
    r'продолж[а-я]+\s+расти',
    r'относител[а-я]+\s+устойчив',
    # Дивиденды / прибыль
    r'рекордн[а-я]+\s+(?:прибыль|дивиденд|выплат)',
    r'высок[а-я]+\s+дивидендн[а-я]+\s+доходн',
    r'рекоменду[емт]\s+(?:к\s+)?покупк',
    r'превзошл[а-я]+\s+ожидания',
    r'лучше\s+ожиданий',
    r'превысил[а-я]+\s+ожидания',
    r'выросл[а-я]+\s+дивидендн',
    r'запустил[а-я]+\s+программу\s+выкупа',
    r'байбэк',
    # Снижение «плохих» метрик = позитив
    r'снижени[аея]\s+(?:убыт|долг|задолженн|кредитн|дефицит|инфляц|безработ)',
    r'падени[аея]\s+(?:убыт|долг|задолженн|кредитн|дефицит|инфляц|безработ)',
    r'сокращени[аея]\s+(?:убыт|долг|задолженн|кредитн|дефицит)',
    # Санкции — уже учтено / не сюрприз
    r'(?:не\s+)?(?:станет|станут)\s+сюрпризом',
    r'уже\s+учтен[оаы]\s+в\s+(?:цен|котиров)',
    r'дисконтированн[а-я]+\s+(?:цен|рынок)',
    r'отыгран[оаы]\s+рынком',
    r'статус-кво',
    # Апгрейды аналитиков
    r'апгрейд',
    r'повысил[а-я]+\s+рейтинг',
    r'повышени[аея]\s+рейтинг',
    r'улучшени[ея]\s+показател',
    r'улучшени[ея]\s+прогноз',
    # Импортозамещение / господдержка
    r'господдержк[ауе]',
    r'импортозамещ',
    r'госзаказ',
    r'субсиди[а-я]+',
]

# Сильные негативные паттерны: даже если ML-модель сказала POSITIVE,
# эти фразы должны «перевернуть» сентимент в NEGATIVE.
STRONG_NEGATIVE_PATTERNS = [
    # Корпоративные катастрофы
    r'дефолт',
    r'банкротств',
    r'мошенничеств',
    r'отзыв\s+лицензи',
    r'санкции\s+(?:против|в\s+отношени)\s+(?:компани|руководств|акционер|бизнес)',
    r'включени[ея]\s+в\s+sdn',
    r'блокировк[ауе]\s+активов',
    r'заморозк[ауе]\s+активов',
    r'арест\s+счет',
    # Уголовные/регуляторные
    r'уголовн[а-я]+\s+дел',
    r'следственн[а-я]+\s+комитет',
    r'проверк[ауе]\s+(?:фас|центробанк|цб|следственн|прокуратур|налогов)',
    r'штраф\s+от\s+(?:фас|центробанк|цб|налогов)',
    # Падения и обвалы
    r'обвал\s+котиров',
    r'обвал\s+акций',
    r'резкое\s+падение',
    r'резкое\s+снижение',
    r'крупнейшее\s+падение',
    r'просадк[ауе]\s+(?:более|сильн|значительн|крупн)',
    # Остановка деятельности
    r'остановк[ауе]\s+производств',
    r'остановк[ауе]\s+завод',
    r'остановк[ауе]\s+ добыч',
    r'авари[а-я]+\s+на\s+завод',
    r'забастовк',
    # Убытки
    r'чистый\s+убыт',
    r'крупнейший\s+убыт',
    r'рекордный\s+убыт',
    r'пропис[а-я]+\s+убыт',
    r'ухудшени[ея]\s+финансов',
    r'ухудшени[ея]\s+показател',
    # Даунгрейды
    r'даунгрейд',
    r'понизил[а-я]+\s+рейтинг',
    r'понижени[аея]\s+рейтинг',
    r'негативн[а-я]+\s+прогноз',
    r'не\s+рекоменду[емт]\s+к\s+покупк',
    # Делистинг
    r'делистинг',
    r'исключени[ея]\s+из\s+индекса',
    r'принудительн[а-я]+\s+делистинг',
]

# Кэшируем скомпилированные паттерны на уровне модуля (компилируем 1 раз)
_POSITIVE_RE = [re.compile(p, re.IGNORECASE) for p in STRONG_POSITIVE_PATTERNS]
_NEGATIVE_RE = [re.compile(p, re.IGNORECASE) for p in STRONG_NEGATIVE_PATTERNS]


def apply_russian_market_context_override(title: str,
                                          content: str,
                                          ml_sentiment: float) -> Tuple[float, str, str]:
    """
    Контекстная корректировка сентимента для российского фондового рынка.

    Возвращает:
        (corrected_sentiment, override_reason, override_type)
        override_type: 'positive' | 'negative' | None
    """
    text = f"{title} {content[:1000]}".lower()

    # 🆕 Сначала проверяем контекст «снижение плохих метрик» (сильный позитив):
    # если в тексте есть «снизил/сократил/уменьшил {убыток/долг/задолженность/...}»,
    # это позитивный сигнал — вырезаем эти сегменты из текста, чтобы
    # негативный паттерн «чистый убыток» не сработал ложно.
    reduction_pattern = re.compile(
        r'(?:снижени[аея]\s+|падени[аея]\s+|сокращени[аея]\s+|уменьшени[аея]\s+|'
        r'снизил[а-я]*\s+|сократил[а-я]*\s+|уменьшил[а-я]*\s+)'
        r'(?:чистый\s+|крупн[a-я]+\s+|значительн[a-я]+\s+|сильн[a-я]+\s+|рекордн[a-я]+\s+)?'
        r'(?:убыт[а-я]*|долг[а-я]*|задолженн[a-я]*|кредитн[a-я]*|дефицит[а-я]*|'
        r'инфляц[a-я]*|безработиц[a-я]*|расход[a-я]*)',
        re.IGNORECASE
    )
    reduction_matches = reduction_pattern.findall(text)
    # Удаляем эти сегменты из текста для последующего поиска негативных паттернов
    text_for_negative_check = reduction_pattern.sub(' ', text)

    pos_match = None
    for pat in _POSITIVE_RE:
        m = pat.search(text)
        if m:
            pos_match = m.group(0)
            break

    # Если нашли контекст "снижение убытков", считаем это позитивным сигналом
    if reduction_matches and not pos_match:
        pos_match = f"снижение_плохих_метрик:{reduction_matches[0].strip()[:30]}"

    neg_match = None
    for pat in _NEGATIVE_RE:
        m = pat.search(text_for_negative_check)
        if m:
            neg_match = m.group(0)
            break

    # Если есть обе категории — оставляем ML-решение (взаимная компенсация)
    if pos_match and neg_match:
        return ml_sentiment, f"mixed:{pos_match}/{neg_match}", None

    # Сильный позитив перевешивает слабый негатив ML
    if pos_match:
        if ml_sentiment < 0:
            # Переворачиваем негатив в позитив с амплитудой 0.5-0.7
            corrected = max(abs(ml_sentiment) * 0.7, 0.45)
            return corrected, f"pos_override:{pos_match}", 'positive'
        elif ml_sentiment < 0.3:
            # Усиливаем слабый позитив
            return max(ml_sentiment, 0.45), f"pos_boost:{pos_match}", 'positive'
        return ml_sentiment, f"pos_keep:{pos_match}", 'positive'

    # Сильный негатив перевешивает слабый позитив ML
    if neg_match:
        if ml_sentiment > 0:
            corrected = -max(abs(ml_sentiment) * 0.7, 0.45)
            return corrected, f"neg_override:{neg_match}", 'negative'
        elif ml_sentiment > -0.3:
            return min(ml_sentiment, -0.45), f"neg_boost:{neg_match}", 'negative'
        return ml_sentiment, f"neg_keep:{neg_match}", 'negative'

    return ml_sentiment, "", None


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class NewsAnalyzer:

    def __init__(self, config: Dict, tickers_config: Dict, models_data_dir: Path, log_enabled: bool = True, log_level: str = "INFO"):
        self.logger = get_logger("NEWS_ANALYZER", log_level if log_enabled else "CRITICAL")
        self.config = config
        self.tickers_config = tickers_config
        self.models_data_dir = models_data_dir

        self.sentiment_cache = {}
        self.filter_cache = {}
        self.signal_history = {}

        self._use_automaton = False
        self._use_regex = False
        self.include_patterns = []
        self.exclude_patterns = []
        self.filter_automaton = None

        self._init_from_config()
        self._init_filters()
        self._init_sentiment_model()
        self._init_tickers()
        self._init_broker()

        self.stats = {
            'total_processed': 0,
            'filtered_out': 0,
            'filtered_in': 0,
            'sentiment_analyzed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'signals_generated': 0
        }
        self.logger.info(f"NewsAnalyzer инициализирован: ML={'Да' if self.use_ml_sentiment else 'Нет'}, Aho-Corasick={'Да' if self.use_ahocorasick_filter else 'Нет'}")

    def _init_from_config(self):
        cache_config = self.config.get('cache', {})
        self.cache_ttl = cache_config.get('ttl_seconds', 3600)

        processing_config = self.config.get('processing', {})
        self.max_text_length = processing_config.get('max_text_length_chars', 512)
        self.batch_size = processing_config.get('batch_size', 32)
        self.deduplication_method = processing_config.get('deduplication_method', 'title_hash')

        filter_config = self.config.get('filter', {})
        self.use_ahocorasick_filter = filter_config.get('use_ahocorasick', False)
        self.include_words = filter_config.get('include', [])
        self.exclude_words = filter_config.get('exclude', [])
        self.use_word_forms = filter_config.get('use_word_forms', False)
        self.min_word_length = filter_config.get('min_word_length_for_forms', 3)
        self.word_endings = filter_config.get('word_endings', [])
        self.word_exceptions = filter_config.get('word_exceptions', [])

        sentiment_config = self.config.get('sentiment', {})
        self.use_ml_sentiment = sentiment_config.get('use_ml', True)
        self.use_lexicon_fallback = sentiment_config.get('use_lexicon_fallback', False)
        self.ml_device = sentiment_config.get('ml_device', -1)
        self.sentiment_model_name = sentiment_config.get('model_name', '')
        self.positive_threshold = sentiment_config.get('positive_threshold', 0.5)
        self.negative_threshold = sentiment_config.get('negative_threshold', -0.5)
        self.priority_weight = sentiment_config.get('priority_weight', 0.1)

        tickers_config_local = self.config.get('tickers', {})
        self.tickers_file = tickers_config_local.get('file_path', '')
        self.extract_from_title = tickers_config_local.get('extract_from_title', True)
        self.extract_from_content = tickers_config_local.get('extract_from_content', False)

    def _init_filters(self):
        use_ahocorasick = self.use_ahocorasick_filter

        if use_ahocorasick and AHOCORASICK_AVAILABLE and (self.include_words or self.exclude_words):
            self.filter_automaton = ahocorasick.Automaton()
            for word in self.include_words:
                self.filter_automaton.add_word(word.lower(), ('include', word))
            for word in self.exclude_words:
                self.filter_automaton.add_word(word.lower(), ('exclude', word))
            self.filter_automaton.make_automaton()
            self._use_automaton = True
            self._use_regex = False
            self.logger.info(f"Aho-Corasick фильтр активирован: include={len(self.include_words)}, exclude={len(self.exclude_words)}")

        elif self.use_word_forms and self.word_endings:
            self.include_patterns = []
            self.exclude_patterns = []
            for word in self.include_words:
                if len(word) >= self.min_word_length:
                    base = word
                    for exc in self.word_exceptions:
                        if word.endswith(exc):
                            base = word[:-len(exc)]
                            break
                    endings = '|'.join(re.escape(e) for e in self.word_endings)
                    pattern = rf'\b{re.escape(base)}(?:{endings})?\b'
                else:
                    pattern = rf'\b{re.escape(word)}\b'
                self.include_patterns.append(re.compile(pattern, re.IGNORECASE))

            for word in self.exclude_words:
                pattern = rf'\b{re.escape(word)}\b'
                self.exclude_patterns.append(re.compile(pattern, re.IGNORECASE))

            self._use_automaton = False
            self._use_regex = True
            self.logger.info(f"Regex фильтр с словоформами: include={len(self.include_patterns)}, exclude={len(self.exclude_patterns)}")

        else:
            self.include_patterns = []
            self.exclude_patterns = []
            for word in self.include_words:
                pattern = rf'\b{re.escape(word)}\b'
                self.include_patterns.append(re.compile(pattern, re.IGNORECASE))
            for word in self.exclude_words:
                pattern = rf'\b{re.escape(word)}\b'
                self.exclude_patterns.append(re.compile(pattern, re.IGNORECASE))
            self._use_automaton = False
            self._use_regex = True
            self.logger.info(f"Regex фильтр без словоформ: include={len(self.include_patterns)}, exclude={len(self.exclude_patterns)}")

    def _init_sentiment_model(self):
        self.sentiment_model = None

        if self.use_ml_sentiment and TRANSFORMERS_AVAILABLE:
            try:
                model_path = self.models_data_dir / self.sentiment_model_name.replace('/', '_')
                if model_path.exists() and any(model_path.iterdir()):
                    self.sentiment_model = pipeline(
                        "text-classification",
                        model=str(model_path),
                        device=self.ml_device
                    )
                else:
                    self.sentiment_model = pipeline(
                        "text-classification",
                        model=self.sentiment_model_name,
                        device=self.ml_device
                    )
                    model_path.mkdir(parents=True, exist_ok=True)
                    self.sentiment_model.save_pretrained(str(model_path))
                self.logger.info(f"ML модель загружена: {self.sentiment_model_name} (device={self.ml_device})")
            except Exception as e:
                if self.use_lexicon_fallback:
                    self.logger.warning(f"Ошибка загрузки ML модели: {e}, переключение на словарный метод")
                    self.use_ml_sentiment = False
                    self._init_lexicon()
                else:
                    raise
        else:
            self._init_lexicon()
            self.logger.info(f"Загружен словарь: +{len(self.positive_words)} / -{len(self.negative_words)}")

    def _init_lexicon(self):
        lexicon_config = self.config.get('sentiment', {}).get('lexicon', {})
        self.positive_words = set(lexicon_config.get('positive', []))
        self.negative_words = set(lexicon_config.get('negative', []))

    def _init_tickers(self):
        try:
            watchlist = self.tickers_config.get('watchlist', [])
            self.ticker_variants = {}
            self.tickers_list = []

            for item in watchlist:
                ticker = item['ticker']
                self.tickers_list.append(ticker)

                name = item.get('name', '').lower()
                if name:
                    self.ticker_variants[name] = ticker

                for variant in item.get('variants', []):
                    self.ticker_variants[variant.lower()] = ticker

            self.tickers_set = set(self.tickers_list)

            self.logger.info(f"Загружено тикеров: {len(self.tickers_set)}, вариантов: {len(self.ticker_variants)}")

        except Exception:
            self.logger.warning("Список тикеров пуст")
            self.tickers_set = set()
            self.ticker_variants = {}

    def _init_broker(self):
        broker_config = self.config.get('broker', {})
        self.broker_enabled = broker_config.get('enabled', True)
        self.min_news_for_signal = broker_config.get('min_news_count', 1)
        self.buy_threshold = broker_config.get('buy_threshold', 0.6)
        self.sell_threshold = broker_config.get('sell_threshold', -0.6)
        self.confidence_multiplier = broker_config.get('confidence_multiplier', 1.0)
        self.max_signals_per_run = broker_config.get('max_signals_per_run', 10)
        self.signal_cooldown_minutes = broker_config.get('signal_cooldown_minutes', 0)
        self.require_multiple_sources = broker_config.get('require_multiple_sources', False)
        self.signal_blacklist = set(broker_config.get('signal_blacklist', []))

    def filter_news(self, news_item: Dict) -> bool:
        text = f"{news_item.get('title', '')} {news_item.get('content', '')}".lower()
        text_hash = hashlib.md5(text.encode()).hexdigest()

        cached = self.filter_cache.get(text_hash)
        if cached and (time.time() - cached['timestamp'] < self.cache_ttl):
            self.stats['cache_hits'] += 1
            return cached['result']

        self.stats['cache_misses'] += 1
        result = self._apply_filter(text)

        self.filter_cache[text_hash] = {'result': result, 'timestamp': time.time()}

        if result:
            self.stats['filtered_in'] += 1
        else:
            self.stats['filtered_out'] += 1

        return result

    def _apply_filter(self, text: str) -> bool:
        text_lower = text.lower()

        if self.exclude_words:
            if self._use_automaton:
                for _, (word_type, _) in self.filter_automaton.iter(text_lower):
                    if word_type == 'exclude':
                        return False
            else:
                for pattern in self.exclude_patterns:
                    if pattern.search(text_lower):
                        return False

        if self.include_words:
            if self._use_automaton:
                for _, (word_type, _) in self.filter_automaton.iter(text_lower):
                    if word_type == 'include':
                        return True
                return False
            else:
                for pattern in self.include_patterns:
                    if pattern.search(text_lower):
                        return True
                return False

        return True

    def analyze_sentiment_batch(self, news_items: List[Dict]) -> List[Dict]:
        if not news_items:
            return []

        if self.use_ml_sentiment and self.sentiment_model:
            result = self._analyze_with_ml(news_items)
        else:
            result = self._analyze_with_lexicon(news_items)

        self.stats['sentiment_analyzed'] += len(news_items)
        return result

    def _analyze_with_ml(self, news_items: List[Dict]) -> List[Dict]:
        texts = []
        news_indices = []
        results = []

        for i, news in enumerate(news_items):
            text = f"{news.get('title', '')} {news.get('content', '')}"[:self.max_text_length]
            text_hash = hashlib.md5(text.encode()).hexdigest()

            cached = self.sentiment_cache.get(text_hash)
            if cached and (time.time() - cached['timestamp'] < self.cache_ttl):
                self.stats['cache_hits'] += 1
                news['sentiment'] = cached['sentiment']
                news['sentiment_score'] = cached['score']
                news['sentiment_label'] = self._get_label(cached['sentiment'])
                results.append(news)
            else:
                self.stats['cache_misses'] += 1
                texts.append(text)
                news_indices.append(i)

        if texts:
            for start_idx in range(0, len(texts), self.batch_size):
                batch = texts[start_idx:start_idx + self.batch_size]
                try:
                    model_results = self.sentiment_model(batch)
                    for j, mr in enumerate(model_results):
                        idx = news_indices[start_idx + j]
                        news_item = news_items[idx]

                        label = mr['label'].lower()
                        score = mr['score']
                        ml_sentiment = score if 'positive' in label else (-score if 'negative' in label else 0.0)

                        # 🆕 Контекстная корректировка для российского рынка
                        title = news_item.get('title', '')
                        content = news_item.get('content', '')
                        corrected, override_reason, override_type = apply_russian_market_context_override(
                            title, content, ml_sentiment
                        )

                        if override_reason:
                            # Логируем только если сентимент реально изменился
                            if abs(corrected - ml_sentiment) > 0.01:
                                self.logger.debug(
                                    f"Sentiment override: '{title[:60]}' "
                                    f"ML={ml_sentiment:+.2f} → {corrected:+.2f} ({override_reason})"
                                )

                        sentiment = corrected

                        news_item['sentiment'] = sentiment
                        news_item['sentiment_score'] = abs(sentiment)
                        news_item['sentiment_label'] = self._get_label(sentiment)
                        # Сохраняем причину override для веб-интерфейса
                        if override_reason:
                            news_item['sentiment_override'] = override_reason

                        text_hash = hashlib.md5(batch[j].encode()).hexdigest()
                        self.sentiment_cache[text_hash] = {
                            'sentiment': sentiment,
                            'score': abs(sentiment),
                            'timestamp': time.time()
                        }
                        results.append(news_item)
                except Exception as e:
                    self.logger.warning(f"ML batch sentiment failed: {e}")
                    for j in range(start_idx, min(start_idx + self.batch_size, len(texts))):
                        idx = news_indices[j]
                        news_item = news_items[idx]
                        news_item['sentiment'] = 0.0
                        news_item['sentiment_score'] = 0.0
                        news_item['sentiment_label'] = 'NEUTRAL'
                        results.append(news_item)

        return results

    def _analyze_with_lexicon(self, news_items: List[Dict]) -> List[Dict]:
        for news in news_items:
            text = f"{news.get('title', '')} {news.get('content', '')}".lower()

            pos_count = sum(1 for w in self.positive_words if w in text)
            neg_count = sum(1 for w in self.negative_words if w in text)

            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                sentiment = 0.0

            priority = news.get('priority', 5)
            sentiment = sentiment * (1 - self.priority_weight) + (priority / 10.0) * self.priority_weight

            news['sentiment'] = sentiment
            news['sentiment_score'] = abs(sentiment)
            news['sentiment_label'] = self._get_label(sentiment)

        return news_items

    def _get_label(self, sentiment: float) -> str:
        if sentiment >= self.positive_threshold:
            return 'POSITIVE'
        elif sentiment <= self.negative_threshold:
            return 'NEGATIVE'
        return 'NEUTRAL'

    def extract_tickers(self, news_items: List[Dict]) -> List[Dict]:
        if not self.tickers_set:
            for news in news_items:
                news['tickers'] = []
                news['tickers_count'] = 0
            return news_items

        for news in news_items:
            text = ""
            if self.extract_from_title:
                text += news.get('title', '').lower()
            if self.extract_from_content:
                text += " " + news.get('content', '').lower()

            found_tickers = set()

            for variant, ticker in self.ticker_variants.items():
                if variant in text:
                    found_tickers.add(ticker)

            text_upper = text.upper()
            for ticker in self.tickers_list:
                if ticker in text_upper:
                    found_tickers.add(ticker)

            news['tickers'] = list(found_tickers)
            news['tickers_count'] = len(found_tickers)

        return news_items

    def generate_signals(self, aggregated_by_ticker: Dict[str, Dict]) -> List[Dict]:
        if not self.broker_enabled:
            return []

        signals = []
        current_time = time.time()

        for ticker, data in aggregated_by_ticker.items():
            if ticker in self.signal_blacklist:
                continue

            if data['news_count'] < self.min_news_for_signal:
                continue

            if self.require_multiple_sources:
                sources = set(news.get('source', '') for news in data.get('latest_news', []))
                if len(sources) < 2:
                    continue

            avg_sentiment = data['avg_sentiment']

            if avg_sentiment >= self.buy_threshold:
                signal_type = SignalType.BUY
                base_confidence = avg_sentiment
            elif avg_sentiment <= self.sell_threshold:
                signal_type = SignalType.SELL
                base_confidence = abs(avg_sentiment)
            else:
                continue

            confidence = min(base_confidence * self.confidence_multiplier, 1.0)

            last_signal_time = self.signal_history.get(ticker, 0)
            cooldown_seconds = self.signal_cooldown_minutes * 60
            if current_time - last_signal_time < cooldown_seconds:
                continue

            signal = {
                'ticker': ticker,
                'signal': signal_type.value,
                'confidence': round(confidence, 3),
                'sentiment': round(avg_sentiment, 3),
                'news_count': data['news_count'],
                'timestamp': datetime.now().isoformat(),
                'reason': f"{signal_type.value} сигнал на основе {data['news_count']} новостей со средним сентиментом {avg_sentiment:.2f}"
            }

            signals.append(signal)
            self.signal_history[ticker] = current_time

        signals.sort(key=lambda x: x['confidence'], reverse=True)
        self.stats['signals_generated'] = len(signals)

        self.logger.info(f"Сгенерировано {len(signals)} сигналов: {[(s['ticker'], s['signal']) for s in signals[:5]]}")

        return signals[:self.max_signals_per_run]

    def aggregate_by_ticker(self, news_items: List[Dict]) -> Dict[str, Dict]:
        result = {}

        for news in news_items:
            for ticker in news.get('tickers', []):
                if ticker not in result:
                    result[ticker] = {
                        'ticker': ticker,
                        'news_count': 0,
                        'sentiments': [],
                        'sources': set(),
                        'latest_news': []
                    }

                result[ticker]['news_count'] += 1
                result[ticker]['sentiments'].append(news.get('sentiment', 0.0))
                result[ticker]['sources'].add(news.get('source_name', 'unknown'))
                result[ticker]['latest_news'].append({
                    'title': news.get('title', ''),
                    'sentiment': news.get('sentiment', 0.0),
                    'source': news.get('source_name', ''),
                    'published_at': news.get('published_at', '')
                })

        for ticker, data in result.items():
            data['avg_sentiment'] = sum(data['sentiments']) / len(data['sentiments'])
            data['sentiment_label'] = self._get_label(data['avg_sentiment'])
            data['sources'] = list(data['sources'])
            data['latest_news'] = sorted(data['latest_news'],
                                         key=lambda x: x.get('published_at', ''),
                                         reverse=True)[:5]

        return result

    def remove_duplicates(self, news_items: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []

        for item in news_items:
            if self.deduplication_method == 'title_hash':
                key = hashlib.md5(item.get('title', '').lower().encode()).hexdigest()
            elif self.deduplication_method == 'link':
                key = item.get('link', '')
            else:
                key = item.get('id', '')

            if key and key not in seen:
                seen.add(key)
                unique.append(item)

        return unique

    def process(self, raw_news: List[Dict]) -> Dict:
        if not raw_news:
            return {'status': 'no_news', 'processed': 0}

        self.stats['total_processed'] += len(raw_news)

        deduplicated = self.remove_duplicates(raw_news)
        filtered_news = [n for n in deduplicated if self.filter_news(n)]
        
        self.logger.info(f"Фильтрация: {len(deduplicated)} -> {len(filtered_news)} новостей")

        if not filtered_news:
            return {
                'status': 'all_filtered_out',
                'total_raw': len(raw_news),
                'total_deduplicated': len(deduplicated),
                'total_filtered': 0,
                'total_analyzed': 0,
                'tickers_found': 0,
                'news': [],
                'aggregated_by_ticker': {},
                'signals': [],
                'stats': self.stats
            }

        analyzed_news = self.analyze_sentiment_batch(filtered_news)
        news_with_tickers = self.extract_tickers(analyzed_news)
        ticker_aggregation = self.aggregate_by_ticker(news_with_tickers)
        signals = self.generate_signals(ticker_aggregation)

        return {
            'status': 'success',
            'total_raw': len(raw_news),
            'total_deduplicated': len(deduplicated),
            'total_filtered': len(filtered_news),
            'total_analyzed': len(analyzed_news),
            'tickers_found': len(ticker_aggregation),
            'news': news_with_tickers,
            'aggregated_by_ticker': ticker_aggregation,
            'signals': signals,
            'stats': self.stats
        }

    def get_stats(self) -> Dict:
        return self.stats

    def clear_cache(self):
        self.sentiment_cache.clear()
        self.filter_cache.clear()
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0