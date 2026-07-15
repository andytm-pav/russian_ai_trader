#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прокси-стакан и микроструктурный анализ.

Использует доступные поля MOEX marketdata:
  BID, OFFER, SPREAD, BIDDEPTH, OFFERDEPTH, NUMBIDS, NUMOFFERS

Создаёт 6 микроструктурных признаков для state vector:
  1. spread_pct — спред в % от цены
  2. bid_ask_imbalance — дисбаланс объёмов (BIDDEPTH - OFFERDEPTH) / (BIDDEPTH + OFFERDEPTH)
  3. order_imbalance — дисбаланс числа заявок (NUMBIDS - NUMOFFERS) / (NUMBIDS + NUMOFFERS)
  4. bid_volume_relative — объём bid относительно скользящего среднего
  5. offer_volume_relative — объём offer относительно скользящего среднего
  6. microstructure_regime — 0=сбалансированный, 1=покупательский, 2=продавецкий

Также генерирует ранние торговые сигналы на основе микроструктуры.
"""
import time
import threading
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple
import requests
from utils.logger import get_logger

logger = get_logger("MICROSTRUCTURE")


class MicrostructureFetcher:
    """Прокси-стакан из доступных полей MOEX marketdata."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.base_url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities"
        
        # Кэш микроструктуры по тикерам
        self._cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 10  # 10 секунд
        
        # История для скользящих средних объёмов
        self._bid_vol_history = defaultdict(lambda: deque(maxlen=50))
        self._offer_vol_history = defaultdict(lambda: deque(maxlen=50))
        
        # Статистика
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
        }
        
        self._lock = threading.RLock()
        logger.info("MicrostructureFetcher инициализирован")

    def fetch_raw(self, ticker: str) -> Dict:
        """Получение сырых данных стакана 1 уровня для одного тикера."""
        self.stats['total_requests'] += 1
        
        url = f"{self.base_url}/{ticker}/marketdata.json"
        params = {
            'iss.meta': 'off',
            'iss.only': 'marketdata',
        }
        try:
            r = self.session.get(url, params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
            md = data.get("marketdata", {})
            cols = md.get("columns", [])
            rows = md.get("data", [])
            if rows:
                d = dict(zip(cols, rows[0]))
                self.stats['successful_fetches'] += 1
                return d
        except Exception as e:
            logger.debug(f"Ошибка получения стакана для {ticker}: {e}")
            self.stats['failed_fetches'] += 1
        return {}

    def get_microstructure(self, ticker: str, securities_data: Dict = None) -> Dict:
        """
        Получение микроструктурных признаков для тикера.
        
        Использует securities_data из get_all_securities() если доступно,
        иначе делает прямой запрос к MOEX.
        
        Возвращает:
            bid, offer, spread, spread_pct,
            bid_volume, offer_volume, num_bids, num_offers,
            imbalance, order_imbalance,
            bid_vol_relative, offer_vol_relative,
            microstructure_regime (0=balanced, 1=buyer, 2=seller)
        """
        with self._lock:
            # Если есть готовые данные из securities — используем
            if securities_data and ticker in securities_data:
                sec = securities_data[ticker]
                bid = sec.get('bid', 0.0)
                offer = sec.get('offer', 0.0)
                spread = sec.get('spread', 0.0)
                price = sec.get('price', 0.0)
            else:
                # Прямой запрос
                md = self.fetch_raw(ticker)
                bid = float(md.get('BID') or 0)
                offer = float(md.get('OFFER') or 0)
                spread = float(md.get('SPREAD') or 0)
                price = float(md.get('LAST') or md.get('MARKETPRICE') or 0)

            # Если данные из marketdata (прямой запрос) — берём глубину
            if not securities_data or ticker not in securities_data:
                md = self.fetch_raw(ticker) if not md else md
                bid_vol = float(md.get('BIDDEPTH') or 0)
                offer_vol = float(md.get('OFFERDEPTH') or 0)
                num_bids = int(md.get('NUMBIDS') or 0)
                num_offers = int(md.get('NUMOFFERS') or 0)
            else:
                bid_vol = 0
                offer_vol = 0
                num_bids = 0
                num_offers = 0

            # Расчёт признаков
            spread_pct = (spread / bid * 100) if bid > 0 else 0.0
            
            # Дисбаланс объёмов
            total_vol = bid_vol + offer_vol
            imbalance = ((bid_vol - offer_vol) / total_vol) if total_vol > 0 else 0.0
            
            # Дисбаланс числа заявок
            total_orders = num_bids + num_offers
            order_imbalance = ((num_bids - num_offers) / total_orders) if total_orders > 0 else 0.0
            
            # Относительные объёмы (к скользящему среднему)
            self._bid_vol_history[ticker].append(bid_vol)
            self._offer_vol_history[ticker].append(offer_vol)
            
            bid_hist = list(self._bid_vol_history[ticker])
            offer_hist = list(self._offer_vol_history[ticker])
            avg_bid = sum(bid_hist) / len(bid_hist) if bid_hist else 1
            avg_offer = sum(offer_hist) / len(offer_hist) if offer_hist else 1
            bid_vol_relative = bid_vol / avg_bid if avg_bid > 0 else 1.0
            offer_vol_relative = offer_vol / avg_offer if avg_offer > 0 else 1.0
            
            # Режим микроструктуры (пороги из конфига через caller)
            balanced_imb = 0.15  # default, переопределяется через config
            balanced_spread = 0.2
            if abs(imbalance) < balanced_imb and spread_pct < balanced_spread:
                regime = 0  # сбалансированный
            elif imbalance > balanced_imb:
                regime = 1  # покупательский (быки давят)
            else:
                regime = 2  # продавецкий (медведи давят)
            
            return {
                'bid': bid,
                'offer': offer,
                'spread': spread,
                'spread_pct': spread_pct,
                'bid_volume': bid_vol,
                'offer_volume': offer_vol,
                'num_bids': num_bids,
                'num_offers': num_offers,
                'imbalance': imbalance,
                'order_imbalance': order_imbalance,
                'bid_vol_relative': bid_vol_relative,
                'offer_vol_relative': offer_vol_relative,
                'microstructure_regime': regime,
                # 🆕 Дополнительные поля для Варианта F (entry cascading)
                'volume_5m': bid_vol + offer_vol,  # прокси: суммарная глубина
                'volume_30m_avg': avg_bid + avg_offer,  # скользящее среднее
                'trade_acceleration': bid_vol_relative - offer_vol_relative,
            }

    def get_microstructure_batch(self, tickers: list, securities_data: Dict = None) -> Dict[str, Dict]:
        """Получение микроструктуры для списка тикеров."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_microstructure(ticker, securities_data)
            except Exception as e:
                logger.debug(f"Ошибка микроструктуры для {ticker}: {e}")
        return results

    def get_microstructure_features_vector(self, ticker: str, securities_data: Dict = None) -> list:
        """
        Возвращает 6 микроструктурных признаков для state vector.
        Нормализованные значения в диапазоне [-1, 1] или [0, 1].
        """
        ms = self.get_microstructure(ticker, securities_data)
        
        # Нормализация:
        # spread_pct: 0-2% → 0-1 (делим на 2)
        # imbalance: -1..1 → как есть
        # order_imbalance: -1..1 → как есть
        # bid_vol_relative: 0-5 → 0-1 (делим на 5, клип)
        # offer_vol_relative: 0-5 → 0-1
        # regime: 0,1,2 → 0,0.5,1
        
        return [
            min(ms['spread_pct'] / 2.0, 1.0),           # 0-1
            ms['imbalance'],                              # -1..1
            ms['order_imbalance'],                        # -1..1
            min(ms['bid_vol_relative'] / 5.0, 1.0),      # 0-1
            min(ms['offer_vol_relative'] / 5.0, 1.0),    # 0-1
            ms['microstructure_regime'] / 2.0,            # 0, 0.5, 1
        ]

    def generate_signal(self, ticker: str, securities_data: Dict = None,
                        buy_threshold: float = 0.3,
                        sell_threshold: float = -0.3,
                        max_spread_pct: float = 0.2,
                        low_liquidity_spread: float = 0.5) -> Optional[Dict]:
        """
        Генерация раннего торгового сигнала на основе микроструктуры.
        Пороги передаются из конфига.
        """
        ms = self.get_microstructure(ticker, securities_data)

        # Сильный покупательский дисбаланс → BUY
        if ms['imbalance'] > buy_threshold and ms['spread_pct'] < max_spread_pct:
            return {
                'action': 'BUY',
                'confidence': min(abs(ms['imbalance']), 1.0),
                'reason': 'microstructure_buy',
                'imbalance': ms['imbalance'],
                'spread_pct': ms['spread_pct'],
            }

        # Сильный продавецкий дисбаланс → SELL
        if ms['imbalance'] < sell_threshold and ms['spread_pct'] < max_spread_pct:
            return {
                'action': 'SELL',
                'confidence': min(abs(ms['imbalance']), 1.0),
                'reason': 'microstructure_sell',
                'imbalance': ms['imbalance'],
                'spread_pct': ms['spread_pct'],
            }

        # Широкий спред → HOLD (низкая ликвидность)
        if ms['spread_pct'] > low_liquidity_spread:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'reason': 'microstructure_low_liquidity',
                'spread_pct': ms['spread_pct'],
            }

        return None


# Синглтон
microstructure_fetcher = MicrostructureFetcher()
