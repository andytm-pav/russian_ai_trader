"""
Диагностика новостного сегмента системы russian_ai_trader
Только чтение данных, без изменений
"""

import json
import time
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any
import os

# Попытка импорта классов для прямого доступа
try:
    from fetchers.news_fetcher import OptimizedNewsFetcher
    from fetchers.rss_fetcher import RSSFetcher
    from core.core_news_trader import NewsTraderCore
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False
    print("⚠️ Модули не найдены, работаем только с конфигами")


class NewsSegmentDiagnostic:
    """Диагностика новостного сегмента"""

    def __init__(self):
        self.config_path = "config/rss_sources.json"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {},
            "sources": {},
            "performance": {},
            "sentiment": {},
            "issues": []
        }

    def load_config(self) -> Dict:
        """Загрузка конфигурации RSS"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Конфигурация загружена: {self.config_path}")
            return config
        except Exception as e:
            self.results["issues"].append(f"Ошибка загрузки конфига: {e}")
            return {}

    def analyze_config(self, config: Dict):
        """Анализ структуры конфигурации"""
        print("\n" + "="*60)
        print("1. АНАЛИЗ КОНФИГУРАЦИИ RSS")
        print("="*60)

        # Основные параметры
        sources = config.get('sources', [])
        enabled_sources = [s for s in sources if s.get('enabled', True)]

        self.results["config"] = {
            "total_sources": len(sources),
            "enabled_sources": len(enabled_sources),
            "update_interval": config.get('update_interval_minutes', 5),
            "max_news_per_source": config.get('max_news_per_source', 20),
            "use_ml_model": config.get('use_ml_model', False)
        }

        print(f"📊 Всего источников: {len(sources)}")
        print(f"📊 Активных источников: {len(enabled_sources)}")
        print(f"⏱  Интервал обновления: {config.get('update_interval_minutes', 5)} мин")
        print(f"📰 Макс. новостей/источник: {config.get('max_news_per_source', 20)}")
        print(f"🤖 ML модель: {'Включена' if config.get('use_ml_model') else 'Отключена'}")

        # Анализ источников по категориям
        categories = Counter()
        priorities = []

        for source in enabled_sources:
            categories[source.get('category', 'unknown')] += 1
            priorities.append(source.get('priority', 5))

        self.results["sources"]["by_category"] = dict(categories)
        self.results["sources"]["avg_priority"] = sum(priorities)/len(priorities) if priorities else 0

        print(f"\n📂 Распределение по категориям:")
        for cat, count in categories.most_common():
            print(f"   {cat}: {count}")

        # Проверка фильтров
        keywords = config.get('keywords_filter', {})
        include_count = len(keywords.get('include', []))
        exclude_count = len(keywords.get('exclude', []))

        self.results["config"]["include_keywords"] = include_count
        self.results["config"]["exclude_keywords"] = exclude_count

        print(f"\n🔍 Фильтры:")
        print(f"   Включающие слова: {include_count}")
        print(f"   Исключающие слова: {exclude_count}")

        # Словарь сентиментов
        sentiment_dict = config.get('sentiment_dictionary', {})
        pos_words = len(sentiment_dict.get('positive', []))
        neg_words = len(sentiment_dict.get('negative', []))
        neutral_words = len(sentiment_dict.get('neutral', []))

        self.results["config"]["sentiment_words"] = {
            "positive": pos_words,
            "negative": neg_words,
            "neutral": neutral_words
        }

        print(f"\n😊 Словарь сентиментов:")
        print(f"   Позитивные: {pos_words}")
        print(f"   Негативные: {neg_words}")
        print(f"   Нейтральные: {neutral_words}")

    def check_source_availability(self, config: Dict):
        """Проверка доступности источников"""
        print("\n" + "="*60)
        print("2. ПРОВЕРКА ДОСТУПНОСТИ ИСТОЧНИКОВ")
        print("="*60)

        import requests

        sources = config.get('sources', [])
        enabled_sources = [s for s in sources if s.get('enabled', True)]

        working = 0
        slow = 0
        failed = 0

        for i, source in enumerate(enabled_sources[:10]):  # Проверяем первые 10
            url = source.get('url', '')
            name = source.get('name', 'Unknown')

            print(f"\n🔗 {i+1}. {name}")

            try:
                start = time.time()
                response = requests.get(url, timeout=5, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; NewsBot)'
                })
                elapsed = time.time() - start

                if response.status_code == 200:
                    status = "✅ Доступен"
                    working += 1
                    if elapsed > 2:
                        status += f" (медленно: {elapsed:.1f}с)"
                        slow += 1
                    else:
                        status += f" ({elapsed:.1f}с)"
                    print(f"   {status}")
                else:
                    status = f"❌ Ошибка {response.status_code}"
                    failed += 1
                    print(f"   {status}")
                    self.results["issues"].append(f"Источник {name} недоступен: {response.status_code}")

            except Exception as e:
                status = f"❌ Ошибка: {str(e)[:50]}"
                failed += 1
                print(f"   {status}")
                self.results["issues"].append(f"Источник {name} не отвечает: {str(e)[:50]}")

        self.results["performance"]["source_availability"] = {
            "working": working,
            "slow": slow,
            "failed": failed,
            "total_checked": min(10, len(enabled_sources))
        }

        print(f"\n📊 Итог: {working} доступно, {slow} медленных, {failed} ошибок")

    def analyze_fetchers(self):
        """Анализ работы фетчеров (если доступны)"""
        print("\n" + "="*60)
        print("3. АНАЛИЗ РАБОТЫ ФЕТЧЕРОВ")
        print("="*60)

        if not HAS_MODULES:
            print("⚠️ Модули не доступны, пропускаем")
            return

        try:
            # Проверяем OptimizedNewsFetcher
            print("\n📡 OptimizedNewsFetcher:")
            fetcher = OptimizedNewsFetcher(self.config_path)

            # Получаем статистику
            news = fetcher.get_last_news(limit=50)

            self.results["performance"]["fetcher"] = {
                "news_count": len(news),
                "cache_size": len(fetcher.news_cache),
                "stats": fetcher.stats
            }

            print(f"   Новостей в кэше: {len(news)}")
            print(f"   Cache hits: {fetcher.stats.get('cache_hits', 0)}")
            print(f"   Cache misses: {fetcher.stats.get('cache_misses', 0)}")
            print(f"   Среднее время сбора: {fetcher.stats.get('avg_fetch_time', 0):.2f}с")

            # Анализ сентимента
            if news:
                print("\n🎭 Анализ сентимента:")
                news_with_sentiment = fetcher.analyze_sentiment_batch(news[:10])

                sentiments = [n.get('sentiment', 0) for n in news_with_sentiment]
                avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

                self.results["sentiment"]["sample_avg"] = avg_sentiment
                print(f"   Средний сентимент (10 новостей): {avg_sentiment:.3f}")

                # Распределение
                pos = sum(1 for s in sentiments if s > 0.1)
                neg = sum(1 for s in sentiments if s < -0.1)
                neu = len(sentiments) - pos - neg

                print(f"   Позитивные: {pos}, Негативные: {neg}, Нейтральные: {neu}")

        except Exception as e:
            print(f"❌ Ошибка при анализе фетчеров: {e}")
            self.results["issues"].append(f"Ошибка fetcher: {e}")

    def analyze_core(self):
        """Анализ работы новостного ядра"""
        print("\n" + "="*60)
        print("4. АНАЛИЗ НОВОСТНОГО ЯДРА")
        print("="*60)

        if not HAS_MODULES:
            print("⚠️ Модули не доступны, пропускаем")
            return

        try:
            core = NewsTraderCore(self.config_path)

            # Получаем сентимент для тестовых тикеров
            test_tickers = ['SBER', 'GAZP', 'LKOH']

            print("\n📊 Сентимент по тикерам:")
            ticker_sentiments = {}

            for ticker in test_tickers:
                sentiment = core.get_current_sentiment(ticker)
                enhanced = core.get_enhanced_sentiment(ticker)

                ticker_sentiments[ticker] = {
                    "basic": sentiment,
                    "category": enhanced.get('sentiment_category'),
                    "impact": enhanced.get('impact_level'),
                    "news_count": enhanced.get('news_count', 0)
                }

                print(f"\n   {ticker}:")
                print(f"      Сентимент: {sentiment:.3f}")
                print(f"      Категория: {enhanced.get('sentiment_category')}")
                print(f"      Уровень: {enhanced.get('impact_level')}")
                print(f"      Новостей: {enhanced.get('news_count', 0)}")

            self.results["sentiment"]["tickers"] = ticker_sentiments

            # Рыночный сентимент
            market = core.get_market_sentiment()
            self.results["sentiment"]["market"] = market

            print(f"\n📈 Рыночный сентимент:")
            print(f"   Значение: {market.get('sentiment', 0):.3f}")
            print(f"   Тренд: {market.get('trend')}")
            print(f"   Новостей: {market.get('news_count', 0)}")

        except Exception as e:
            print(f"❌ Ошибка при анализе ядра: {e}")
            self.results["issues"].append(f"Ошибка core: {e}")

    def generate_report(self) -> Dict:
        """Генерация итогового отчета"""
        print("\n" + "="*60)
        print("5. ИТОГОВЫЙ ОТЧЕТ")
        print("="*60)

        # Оценка состояния
        score = 100
        issues = self.results.get("issues", [])

        # Штрафы за проблемы
        score -= len(issues) * 10

        # Бонусы за хорошие показатели
        if self.results["config"].get("enabled_sources", 0) >= 10:
            score += 10
        if self.results["config"].get("use_ml_model"):
            score += 15
        if self.results["performance"].get("fetcher", {}).get("news_count", 0) > 50:
            score += 10

        score = max(0, min(100, score))

        # Определение статуса
        if score >= 80:
            status = "✅ ОТЛИЧНО"
        elif score >= 60:
            status = "⚠️ УДОВЛЕТВОРИТЕЛЬНО"
        else:
            status = "❌ ТРЕБУЕТ ВНИМАНИЯ"

        print(f"\n📊 ОБЩАЯ ОЦЕНКА: {score}% - {status}")

        if issues:
            print(f"\n🔴 ПРОБЛЕМЫ ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("\n✅ Проблем не обнаружено")

        print(f"\n📁 Конфигурация:")
        print(f"   Источников: {self.results['config'].get('enabled_sources', 0)}/{self.results['config'].get('total_sources', 0)}")
        print(f"   ML модель: {'Да' if self.results['config'].get('use_ml_model') else 'Нет'}")

        if self.results.get("sentiment", {}).get("market"):
            print(f"\n📊 Текущий рыночный сентимент: {self.results['sentiment']['market'].get('sentiment', 0):.3f}")

        # Сохранение отчета
        report_file = f"data/news_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        os.makedirs("data", exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 Отчет сохранен: {report_file}")

        self.results["score"] = score
        self.results["status"] = status

        return self.results

    def run(self):
        """Запуск полной диагностики"""
        print("\n" + "🚀"*20)
        print("ДИАГНОСТИКА НОВОСТНОГО СЕГМЕНТА")
        print("🚀"*20 + "\n")

        config = self.load_config()
        if config:
            self.analyze_config(config)
            self.check_source_availability(config)

        self.analyze_fetchers()
        self.analyze_core()

        return self.generate_report()


if __name__ == "__main__":
    diagnostic = NewsSegmentDiagnostic()
    results = diagnostic.run()

    print("\n" + "="*60)
    print("Диагностика завершена")
    print("="*60)