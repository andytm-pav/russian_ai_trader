#!/usr/bin/env python3
"""
Отладка технических расчётов с реальными данными MOEX
Полностью совместим с проектом
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
import numpy as np

# Добавляем корень проекта в путь
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from utils.logger import setup_logger
    from core.core_technical_trader import TechnicalTraderCore
    from fetchers.moex_fetcher import MoexFetcher
    from utils.portfolio_manager import PortfolioManager

    logger = setup_logger("DEBUG_TECH")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Запускайте из корня проекта: python debug_real_data.py")
    sys.exit(1)


class TechnicalDebugger:
    """Отладчик технических расчётов"""

    def __init__(self):
        self.tech_core = TechnicalTraderCore()
        self.fetcher = MoexFetcher(use_cache=False)  # Отключаем кэш для свежих данных
        self.portfolio = PortfolioManager()
        self.tickers_to_analyze = []

    def load_tickers_from_portfolio(self):
        """Загрузка тикеров из портфеля"""
        try:
            # Получаем все тикеры из портфеля
            self.tickers_to_analyze = list(self.portfolio.positions.keys())

            if not self.tickers_to_analyze:
                print("📭 Портфель пуст, используем тестовые тикеры")
                # Если портфель пуст, используем популярные тикеры MOEX
                self.tickers_to_analyze = [
                    'SBER', 'GAZP', 'LKOH', 'YNDX', 'VTBR',  # Голубые фишки
                    'GMKN', 'ROSN', 'SBERP', 'TATN', 'ALRS'  # Популярные акции
                ]

            print(f"📋 Загружено {len(self.tickers_to_analyze)} тикеров из портфеля:")
            for i, ticker in enumerate(self.tickers_to_analyze[:10], 1):
                pos = self.portfolio.positions.get(ticker, {})
                qty = pos.get('qty', 0)
                avg_price = pos.get('avg_price', 0)
                if i <= 5:  # Показываем только первые 5
                    print(f"  {i:2}. {ticker:6} - {qty:4} шт. @ {avg_price:7.2f}₽")

            if len(self.tickers_to_analyze) > 10:
                print(f"  ... и ещё {len(self.tickers_to_analyze) - 10} тикеров")

        except Exception as e:
            print(f"❌ Ошибка загрузки тикеров из портфеля: {e}")
            # Fallback на тестовые тикеры
            self.tickers_to_analyze = ['SBER', 'GAZP', 'LKOH']

    def get_historical_data(self, ticker: str, days: int = 30):
        """Получение исторических данных для тикера"""
        try:
            print(f"📊 Получение данных для {ticker}...", end="")

            # Получаем свечи за последние дни
            candles = self.fetcher.get_candles(ticker, interval=24, count=days)

            if candles is None or candles.empty:
                print(f" ❌ Нет данных")
                return False

            print(f" ✅ {len(candles)} свечей")

            # Загружаем данные в техническое ядро
            loaded_count = 0
            for _, row in candles.iterrows():
                try:
                    # Используем Close цену и Volume
                    price = float(row['Close']) if 'Close' in row else 0

                    # Обрабатываем Volume
                    volume = 1000  # значение по умолчанию
                    if 'Volume' in row:
                        vol_val = row['Volume']
                        if not np.isnan(vol_val):
                            volume = int(vol_val)

                    if price > 0:
                        self.tech_core.update_price_data(ticker, price, volume)
                        loaded_count += 1
                except Exception as e:
                    continue

            print(f"  Загружено {loaded_count} цен в историю")
            return loaded_count > 0

        except Exception as e:
            print(f" ❌ Ошибка: {e}")
            return False

    def debug_indicators_calculation(self, ticker: str):
        """Детальная отладка расчёта индикаторов"""
        print(f"\n{'═' * 60}")
        print(f"🔍 ДЕТАЛЬНАЯ ОТЛАДКА: {ticker}")
        print(f"{'═' * 60}")

        # 1. Проверяем историю цен
        if ticker not in self.tech_core.price_history:
            print("❌ Нет исторических данных")
            return None

        data = self.tech_core.price_history[ticker]
        prices = np.array(data['prices'], dtype=float)
        print(f"📊 История цен: {len(prices)} баров")

        if len(prices) >= 5:
            print(f"   Последние 5 цен: {prices[-5]}")
            for i, price in enumerate(prices[-5:], 1):
                print(f"   -{i}: {price:.2f}")

        # 2. Расчёт индикаторов
        print(f"\n🧮 РАСЧЁТ ИНДИКАТОРОВ...")
        indicators = self.tech_core.calculate_indicators(ticker)

        if not indicators:
            print("❌ Не удалось рассчитать индикаторы")
            return None

        # 3. Выводим все рассчитанные индикаторы
        print(f"\n📈 РАССЧИТАННЫЕ ИНДИКАТОРЫ:")
        for key, value in sorted(indicators.items()):
            if isinstance(value, float):
                print(f"  {key:20} = {value:10.4f}")
            else:
                print(f"  {key:20} = {value}")

        # 4. Анализируем условия сигналов
        print(f"\n🔔 АНАЛИЗ УСЛОВИЙ СИГНАЛОВ:")

        signals_found = []

        # RSI
        rsi = indicators.get('rsi')
        if rsi:
            print(f"  📊 RSI = {rsi:.2f}")
            if rsi < 30:
                print(f"    ✅ RSI_OVERSOLD (RSI < 30): {rsi < 30} → conf=0.7")
                signals_found.append(('RSI_OVERSOLD', 'BUY', 0.7))
            elif rsi > 70:
                print(f"    ✅ RSI_OVERBOUGHT (RSI > 70): {rsi > 70} → conf=0.7")
                signals_found.append(('RSI_OVERBOUGHT', 'SELL', 0.7))
            else:
                print(f"    ➖ RSI в нормальном диапазоне (30-70)")
        else:
            print(f"  📊 RSI: ❌ Нет данных")

        # Тренд
        current = indicators.get('current_price', 0)
        sma_10 = indicators.get('sma_10')
        sma_20 = indicators.get('sma_20')

        if current and sma_10 and sma_20:
            print(f"\n  📉 ТРЕНД:")
            print(f"    Цена: {current:.2f}")
            print(f"    SMA(10): {sma_10:.2f}")
            print(f"    SMA(20): {sma_20:.2f}")

            if current > sma_10 > sma_20:
                print(f"    ✅ TREND_UP (цена > SMA10 > SMA20): {current > sma_10 > sma_20} → conf=0.5")
                signals_found.append(('TREND_UP', 'BUY', 0.5))
            elif current < sma_10 < sma_20:
                print(f"    ✅ TREND_DOWN (цена < SMA10 < SMA20): {current < sma_10 < sma_20} → conf=0.5")
                signals_found.append(('TREND_DOWN', 'SELL', 0.5))
            else:
                print(f"    ➖ Нет чёткого тренда")

        # Bollinger Bands
        bb_lower = indicators.get('bb_lower')
        bb_upper = indicators.get('bb_upper')
        bb_width = indicators.get('bb_width', 0)

        if current and bb_lower and bb_upper:
            print(f"\n  📊 BOLLINGER BANDS:")
            print(f"    Цена: {current:.2f}")
            print(f"    BB Lower: {bb_lower:.2f}")
            print(f"    BB Upper: {bb_upper:.2f}")
            print(f"    BB Width: {bb_width:.4f}")

            if current <= bb_lower and bb_width > 0.05:
                print(f"    ✅ BB_OVERSOLD (цена <= BB Lower и Width > 0.05) → conf=0.8")
                signals_found.append(('BB_OVERSOLD', 'BUY', 0.8))
            elif current >= bb_upper and bb_width > 0.05:
                print(f"    ✅ BB_OVERBOUGHT (цена >= BB Upper и Width > 0.05) → conf=0.8")
                signals_found.append(('BB_OVERBOUGHT', 'SELL', 0.8))
            else:
                print(f"    ➖ Цена в середине BB или полосы узкие")

        # MACD
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        macd_hist = indicators.get('macd_hist', 0)

        if macd is not None and macd_signal is not None:
            print(f"\n  📊 MACD:")
            print(f"    MACD: {macd:.4f}")
            print(f"    Signal: {macd_signal:.4f}")
            print(f"    Histogram: {macd_hist:.4f}")

            if macd > macd_signal and macd_hist > 0:
                print(f"    ✅ MACD_BULLISH (MACD > Signal и Hist > 0) → conf=0.6")
                signals_found.append(('MACD_BULLISH', 'BUY', 0.6))
            elif macd < macd_signal and macd_hist < 0:
                print(f"    ✅ MACD_BEARISH (MACD < Signal и Hist < 0) → conf=0.6")
                signals_found.append(('MACD_BEARISH', 'SELL', 0.6))
            else:
                print(f"    ➖ MACD нейтральный")

        # Volume
        volume_ratio = indicators.get('volume_ratio', 1)
        price_change_1d = indicators.get('price_change_1d', 0)

        print(f"\n  📊 ОБЪЁМ:")
        print(f"    Volume Ratio: {volume_ratio:.2f}")
        print(f"    Price Change 1d: {price_change_1d:.2f}%")

        if volume_ratio > 2.0 and price_change_1d > 1:
            print(f"    ✅ VOLUME_SPIKE (ratio > 2.0 и change > 1%) → conf=0.4")
            signals_found.append(('VOLUME_SPIKE', 'BUY', 0.4))
        else:
            print(f"    ➖ Нет всплеска объёма")

        # 5. Выводим найденные сигналы
        if signals_found:
            print(f"\n🎯 НАЙДЕННЫЕ СИГНАЛЫ ({len(signals_found)}):")
            for i, (name, action, conf) in enumerate(signals_found, 1):
                print(f"  {i}. {name:15} → {action:4} (conf={conf})")

        # 6. Генерируем и анализируем финальный сигнал
        print(f"\n{'─' * 40}")
        print(f"🎯 ГЕНЕРАЦИЯ ФИНАЛЬНОГО СИГНАЛА:")

        signal = self.tech_core.generate_technical_signals(ticker, indicators)

        if signal:
            print(f"✅ СГЕНЕРИРОВАН СИГНАЛ:")
            print(f"   Действие: {signal['action']}")
            print(f"   Уверенность: {signal['confidence']}")
            print(f"   Индикаторы: {signal['indicators']}")

            # Анализируем источник confidence
            conf = signal['confidence']
            source = self._analyze_confidence_source(conf, signals_found)
            print(f"   📍 Источник confidence: {source}")

            # Если confidence = 0.5, показываем детали
            if abs(conf - 0.5) < 0.01:
                print(f"\n   ⚠️  ВНИМАНИЕ: Confidence = 0.5")
                print(f"   Это означает, что сработал только TREND_UP/DOWN сигнал")
                print(f"   или он был сильнее других сигналов")

        else:
            print("❌ Сигнал не сгенерирован (нет однозначных сигналов)")

        return signal

    def _analyze_confidence_source(self, confidence: float, signals: list) -> str:
        """Анализирует источник confidence"""
        if abs(confidence - 0.5) < 0.01:
            return "TREND_UP/DOWN сигнал (фиксированный conf=0.5)"
        elif abs(confidence - 0.7) < 0.01:
            return "RSI сигнал (фиксированный conf=0.7)"
        elif abs(confidence - 0.6) < 0.01:
            return "MACD сигнал (фиксированный conf=0.6)"
        elif abs(confidence - 0.8) < 0.01:
            return "Bollinger Bands сигнал (фиксированный conf=0.8)"
        elif abs(confidence - 0.4) < 0.01:
            return "Volume Spike сигнал (фиксированный conf=0.4)"
        else:
            return f"Неизвестный (возможно несколько сигналов: {confidence})"

    def analyze_conf_05_issue(self):
        """Анализ проблемы с conf=0.5"""
        print(f"\n{'=' * 60}")
        print(f"🔧 АНАЛИЗ ПРОБЛЕМЫ С CONFIDENCE=0.5")
        print(f"{'=' * 60}")

        print("\n📌 Проблема: Сигналы часто имеют confidence=0.5")

        print("\n📖 Причина в коде core_technical_trader.py:")
        print("""
    # В методе generate_technical_signals():

    # Каждый сигнал имеет ФИКСИРОВАННЫЙ confidence:
    signals.append(('RSI_OVERSOLD', 'BUY', 0.7))       # ← всегда 0.7
    signals.append(('RSI_OVERBOUGHT', 'SELL', 0.7))    # ← всегда 0.7  
    signals.append(('MACD_BULLISH', 'BUY', 0.6))       # ← всегда 0.6
    signals.append(('MACD_BEARISH', 'SELL', 0.6))      # ← всегда 0.6
    signals.append(('TREND_UP', 'BUY', 0.5))           # ← всегда 0.5 ← ВОТ ОН!
    signals.append(('TREND_DOWN', 'SELL', 0.5))        # ← всегда 0.5 ← И ОН!
    signals.append(('BB_OVERSOLD', 'BUY', 0.8))        # ← всегда 0.8
    signals.append(('BB_OVERBOUGHT', 'SELL', 0.8))     # ← всегда 0.8

    # Логика агрегации берёт МАКСИМАЛЬНЫЙ confidence:
    if buy_signals and not sell_signals:
        max_conf = max(s[2] for s in buy_signals)  # ← БЕРЁТСЯ МАКСИМУМ!
        # Если только TREND_UP → max_conf = 0.5
        """)

        print("\n🔍 Почему часто conf=0.5:")
        print("1. TREND_UP/DOWN срабатывает чаще других индикаторов")
        print("2. Условия TREND проще: просто сравнение цены со скользящими")
        print("3. Другие индикаторы требуют более строгих условий")
        print("4. В логике агрегации берётся МАКСИМАЛЬНЫЙ confidence")

        print("\n💡 Решение проблемы:")
        print("1. Сделать confidence динамическим (зависит от силы сигнала)")
        print("2. Усреднять confidence при нескольких сигналах")
        print("3. Добавить веса для разных типов индикаторов")
        print("4. Игнорировать слабые сигналы")

    def run_analysis(self):
        """Запуск полного анализа"""
        print("🔍 ЗАПУСК ОТЛАДКИ ТЕХНИЧЕСКИХ РАСЧЁТОВ")
        print("=" * 60)

        # 1. Загружаем тикеры из портфеля
        self.load_tickers_from_portfolio()

        if not self.tickers_to_analyze:
            print("❌ Нет тикеров для анализа")
            return

        # Ограничиваем количество тикеров для анализа
        tickers_to_process = self.tickers_to_analyze[:5]  # Только 5 тикеров

        # 2. Собираем данные и анализируем каждый тикер
        all_signals = []
        successful_tickers = []

        for ticker in tickers_to_process:
            print(f"\n{'─' * 50}")
            print(f"📈 АНАЛИЗ ТИКЕРА: {ticker}")

            # Получаем исторические данные
            success = self.get_historical_data(ticker, days=30)
            if not success:
                print(f"  ⏩ Пропускаем {ticker} - нет данных")
                continue

            # Детальная отладка
            signal = self.debug_indicators_calculation(ticker)
            if signal:
                all_signals.append((ticker, signal))
                successful_tickers.append(ticker)
            else:
                print(f"  ⏩ {ticker}: нет сигналов")

        # 3. Сводка по всем сигналам
        if all_signals:
            print(f"\n{'=' * 60}")
            print(f"📊 ИТОГОВАЯ СВОДКА:")
            print(f"{'=' * 60}")

            print(f"\n📈 Проанализировано тикеров: {len(successful_tickers)}/{len(tickers_to_process)}")
            print(f"🎯 Сгенерировано сигналов: {len(all_signals)}")

            print(f"\n📋 СИГНАЛЫ:")
            for ticker, signal in all_signals:
                print(
                    f"  {ticker:6} → {signal['action']:4} (conf={signal['confidence']:.2f}, ind={signal['indicators']})")

            # Статистика confidence
            if all_signals:
                conf_values = [s[1]['confidence'] for s in all_signals]
                print(f"\n📊 Статистика confidence:")
                print(f"  Среднее: {np.mean(conf_values):.3f}")
                print(f"  Минимум: {min(conf_values):.3f}")
                print(f"  Максимум: {max(conf_values):.3f}")

                # Анализ распределения
                conf_counts = {}
                for conf in conf_values:
                    key = round(conf, 1)  # Округляем до 0.1
                    conf_counts[key] = conf_counts.get(key, 0) + 1

                print(f"\n📈 Распределение confidence:")
                for conf_val, count in sorted(conf_counts.items()):
                    source = self._get_confidence_source_name(conf_val)
                    print(f"  conf={conf_val:.1f}: {count:2} сигналов ({source})")
        else:
            print(f"\n❌ Нет сгенерированных сигналов")

        # 4. Анализ проблемы с conf=0.5
        if all_signals:
            conf_05_count = sum(1 for _, s in all_signals if abs(s['confidence'] - 0.5) < 0.01)
            if conf_05_count > 0:
                self.analyze_conf_05_issue()

        print(f"\n{'=' * 60}")
        print(f"✅ ОТЛАДКА ЗАВЕРШЕНА")
        print(f"{'=' * 60}")

    def _get_confidence_source_name(self, confidence: float) -> str:
        """Получение названия источника confidence"""
        if abs(confidence - 0.5) < 0.01:
            return "TREND"
        elif abs(confidence - 0.7) < 0.01:
            return "RSI"
        elif abs(confidence - 0.6) < 0.01:
            return "MACD"
        elif abs(confidence - 0.8) < 0.01:
            return "BB"
        elif abs(confidence - 0.4) < 0.01:
            return "VOLUME"
        else:
            return "UNKNOWN"


def main():
    """Основная функция"""
    try:
        debugger = TechnicalDebugger()
        debugger.run_analysis()
    except KeyboardInterrupt:
        print("\n\n⚠️  Отладка прервана пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()