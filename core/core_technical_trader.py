"""
Ядро технического анализа - индикаторы и паттерны
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import talib
import json

from utils.logger import setup_logger

logger = setup_logger("TECH_CORE")


class TechnicalTraderCore:
    """Ядро для технического анализа"""

    def __init__(self):
        self.price_history = {}
        self.indicators_cache = {}
        logger.info("Инициализировано ядро технического анализа")

    def update_price_data(self, ticker: str, price: float, volume: int = 0):
        """Обновление исторических данных по цене"""
        if ticker not in self.price_history:
            self.price_history[ticker] = {
                'prices': [],
                'volumes': [],
                'timestamps': [],
                'max_length': 100
            }

        data = self.price_history[ticker]
        data['prices'].append(price)
        data['volumes'].append(volume)
        data['timestamps'].append(datetime.now())

        # Ограничиваем длину истории
        if len(data['prices']) > data['max_length']:
            data['prices'] = data['prices'][-data['max_length']:]
            data['volumes'] = data['volumes'][-data['max_length']:]
            data['timestamps'] = data['timestamps'][-data['max_length']:]

        # Очищаем кэш индикаторов
        if ticker in self.indicators_cache:
            del self.indicators_cache[ticker]

    def calculate_indicators(self, ticker: str) -> Dict[str, float]:
        """Расчет всех индикаторов для тикера"""
        if ticker not in self.price_history or len(self.price_history[ticker]['prices']) < 20:
            return {}

        # Проверяем кэш
        if ticker in self.indicators_cache:
            return self.indicators_cache[ticker]

        prices = np.array(self.price_history[ticker]['prices'], dtype=float)
        volumes = np.array(self.price_history[ticker]['volumes'], dtype=float)

        indicators = {}

        try:
            # 1. Трендовые индикаторы
            if len(prices) >= 10:
                sma_10_result = talib.SMA(prices, timeperiod=10)
                indicators['sma_10'] = sma_10_result[-1] if len(sma_10_result) > 0 and not np.isnan(
                    sma_10_result[-1]) else prices[-1]

                sma_20_result = talib.SMA(prices, timeperiod=20)
                indicators['sma_20'] = sma_20_result[-1] if len(sma_20_result) > 0 and not np.isnan(
                    sma_20_result[-1]) else prices[-1]

                if len(prices) >= 50:
                    sma_50_result = talib.SMA(prices, timeperiod=50)
                    indicators['sma_50'] = sma_50_result[-1] if len(sma_50_result) > 0 and not np.isnan(
                        sma_50_result[-1]) else 0

                ema_12_result = talib.EMA(prices, timeperiod=12)
                indicators['ema_12'] = ema_12_result[-1] if len(ema_12_result) > 0 and not np.isnan(
                    ema_12_result[-1]) else prices[-1]

                ema_26_result = talib.EMA(prices, timeperiod=26)
                indicators['ema_26'] = ema_26_result[-1] if len(ema_26_result) > 0 and not np.isnan(
                    ema_26_result[-1]) else prices[-1]

            # 2. Осцилляторы
            if len(prices) >= 14:
                rsi_result = talib.RSI(prices, timeperiod=14)
                indicators['rsi'] = rsi_result[-1] if len(rsi_result) > 0 and not np.isnan(rsi_result[-1]) else 50.0

            if len(prices) >= 14:
                bbands_result = talib.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2)
                upper, middle, lower = bbands_result

                if len(upper) > 0 and len(middle) > 0 and len(lower) > 0:
                    indicators['bb_upper'] = upper[-1] if not np.isnan(upper[-1]) else prices[-1] * 1.1
                    indicators['bb_middle'] = middle[-1] if not np.isnan(middle[-1]) else prices[-1]
                    indicators['bb_lower'] = lower[-1] if not np.isnan(lower[-1]) else prices[-1] * 0.9
                    indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators[
                        'bb_middle'] if indicators['bb_middle'] > 0 else 0

            # 3. MACD
            if len(prices) >= 26:
                macd_result = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
                macd, macd_signal, macd_hist = macd_result

                if len(macd) > 0 and len(macd_signal) > 0 and len(macd_hist) > 0:
                    indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else 0
                    indicators['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0
                    indicators['macd_hist'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0

            # 4. Volume indicators
            if len(volumes) >= 20 and volumes.sum() > 0:
                volume_sma_result = talib.SMA(volumes, timeperiod=20)
                indicators['volume_sma'] = volume_sma_result[-1] if len(volume_sma_result) > 0 and volume_sma_result[
                    -1] > 0 else 1
                indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma'] if indicators[
                                                                                           'volume_sma'] > 0 else 1

            # 5. Волатильность
            # Создаем массивы для high, low, close (используем цены для всех)
            high_prices = np.array(prices, dtype=float)
            low_prices = np.array(prices, dtype=float)
            close_prices = np.array(prices, dtype=float)

            atr_result = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            indicators['atr'] = atr_result[-1] if len(atr_result) > 0 and not np.isnan(atr_result[-1]) else 0

            # 6. Моментум
            indicators['momentum'] = self._calculate_momentum(prices)
            indicators['trend_strength'] = self._calculate_trend_strength(prices)

            # 7. Текущая цена и изменения
            indicators['current_price'] = prices[-1]
            indicators['price_change_1d'] = ((prices[-1] / prices[-2]) - 1) * 100 if len(prices) >= 2 else 0
            indicators['price_change_5d'] = ((prices[-1] / prices[-5]) - 1) * 100 if len(prices) >= 5 else 0

            # Кэшируем результаты
            self.indicators_cache[ticker] = indicators

            logger.debug(f"Рассчитано {len(indicators)} индикаторов для {ticker}")

        except Exception as e:
            logger.error(f"Ошибка расчета индикаторов для {ticker}: {str(e)}")
            indicators = {}

        return indicators

    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Расчет моментума"""
        if len(prices) < 10:
            return 0.0

        # Простой моментум за 10 периодов
        return ((prices[-1] / prices[-10]) - 1) * 100

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Расчет силы тренда"""
        if len(prices) < 20:
            return 0.0

        # ADX-like тренд
        recent = prices[-10:]
        earlier = prices[-20:-10]

        if len(recent) == 10 and len(earlier) == 10:
            recent_trend = np.polyfit(range(10), recent, 1)[0]
            earlier_trend = np.polyfit(range(10), earlier, 1)[0]

            # Нормализованная сила тренда
            trend_strength = abs(recent_trend) * 100 / np.mean(prices[-20:])
            return trend_strength

        return 0.0

    def generate_technical_signals(self, ticker: str, indicators: Dict) -> Optional[Dict]:
        """Генерация технических сигналов"""
        if not indicators:
            return None

        signals = []
        confidence = 0.0

        # 1. Проверка RSI
        rsi = indicators.get('rsi')
        if rsi:
            if rsi < 30:
                signals.append(('RSI_OVERSOLD', 'BUY', 0.7))
            elif rsi > 70:
                signals.append(('RSI_OVERBOUGHT', 'SELL', 0.7))

        # 2. Проверка MACD
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        if macd and macd_signal:
            if macd > macd_signal and indicators.get('macd_hist', 0) > 0:
                signals.append(('MACD_BULLISH', 'BUY', 0.6))
            elif macd < macd_signal and indicators.get('macd_hist', 0) < 0:
                signals.append(('MACD_BEARISH', 'SELL', 0.6))

        # 3. Проверка скользящих средних
        current = indicators.get('current_price', 0)
        sma_10 = indicators.get('sma_10')
        sma_20 = indicators.get('sma_20')

        if current and sma_10 and sma_20:
            if current > sma_10 > sma_20:
                signals.append(('TREND_UP', 'BUY', 0.5))
            elif current < sma_10 < sma_20:
                signals.append(('TREND_DOWN', 'SELL', 0.5))

        # 4. Проверка Bollinger Bands
        bb_lower = indicators.get('bb_lower')
        bb_upper = indicators.get('bb_upper')

        if current and bb_lower and bb_upper:
            bb_width = indicators.get('bb_width', 0)

            if current <= bb_lower and bb_width > 0.05:  # Широкие полосы
                signals.append(('BB_OVERSOLD', 'BUY', 0.8))
            elif current >= bb_upper and bb_width > 0.05:
                signals.append(('BB_OVERBOUGHT', 'SELL', 0.8))

        # 5. Объем
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2.0 and indicators.get('price_change_1d', 0) > 1:
            signals.append(('VOLUME_SPIKE', 'BUY', 0.4))

        # Агрегируем сигналы
        if signals:
            buy_signals = [s for s in signals if s[1] == 'BUY']
            sell_signals = [s for s in signals if s[1] == 'SELL']

            if buy_signals and not sell_signals:
                # Только buy сигналы
                max_conf = max(s[2] for s in buy_signals)
                return {
                    'ticker': ticker,
                    'action': 'BUY',
                    'confidence': max_conf,
                    'indicators': [s[0] for s in buy_signals],
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'technical_analysis'
                }
            elif sell_signals and not buy_signals:
                # Только sell сигналы
                max_conf = max(s[2] for s in sell_signals)
                return {
                    'ticker': ticker,
                    'action': 'SELL',
                    'confidence': max_conf,
                    'indicators': [s[0] for s in sell_signals],
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'technical_analysis'
                }

        return None

    def analyze_all_tickers(self, prices: Dict[str, float]) -> List[Dict]:
        """Анализ всех тикеров и генерация сигналов"""
        signals = []

        for ticker, price in prices.items():
            # Обновляем данные
            self.update_price_data(ticker, price)

            # Рассчитываем индикаторы
            indicators = self.calculate_indicators(ticker)

            if indicators:
                # Генерируем сигнал
                signal = self.generate_technical_signals(ticker, indicators)
                if signal:
                    signals.append(signal)

                    logger.info(f"Технический сигнал: {ticker} {signal['action']} "
                                f"(conf={signal['confidence']:.2f}, ind={signal['indicators']})")

        return signals

    def get_technical_summary(self, ticker: str) -> Dict:
        """Получение технической сводки по тикеру"""
        if ticker not in self.price_history:
            return {}

        indicators = self.calculate_indicators(ticker)

        summary = {
            'price_history_count': len(self.price_history[ticker]['prices']),
            'indicators_count': len(indicators),
            'last_update': self.price_history[ticker]['timestamps'][-1].isoformat()
            if self.price_history[ticker]['timestamps'] else None
        }

        # Добавляем ключевые индикаторы
        key_indicators = ['rsi', 'macd', 'sma_10', 'sma_20', 'bb_width', 'volume_ratio']
        for key in key_indicators:
            if key in indicators:
                summary[key] = indicators[key]

        return summary