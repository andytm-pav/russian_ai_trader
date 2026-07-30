#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вариант F: Каскадный confirmer входа.

Пайплайн для каждого тикера из вселенной:
  1. Hawkes-триггер: прогноз числа bullish-событий на 4 часа
  2. Техническое подтверждение (через wait_minutes): RSI 50-70, momentum, BB
  3. Микроструктурное подтверждение (через wait_minutes): imbalance, spread, volume
  4. Хаос-фильтр: RQA DET, L_max, kurtosis, branching ratio η

Все настройки в config/settings.json → entry_cascading.
Никакого хардкода — все пороги из конфига.

Возвращает список EntrySignal для тикеров, прошедших все фазы.
"""
import time
import math
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from utils.logger import get_logger

logger = get_logger("ENTRY_CASCADE")


@dataclass
class EntrySignal:
    ticker: str
    confidence: float
    hawkes_prob_bull: float
    hawkes_net: float
    stop_loss_pct: float
    take_profit_pct: float
    target_weight: float
    reasons: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class EntryCascadingConfirmer:
    """Каскадный confirmer входа (Вариант F)."""

    def __init__(self, config: Dict = None):
        if config is None:
            config = {}
        self.enabled = config.get('enabled', True)

        # Подконфиги
        self.hawkes_cfg = config.get('hawkes_trigger', {})
        self.tech_cfg = config.get('technical_confirmation', {})
        self.ms_cfg = config.get('microstructure_confirmation', {})
        self.chaos_cfg = config.get('chaos_filter', {})
        self.portfolio_cfg = config.get('portfolio_constraints', {})
        self.cooldown = config.get('cooldown_seconds_between_entries', 60)

        # 🆕 v16.3: Конфигурируемые веса и параметры (вместо хардкода)
        self.conf_weights = config.get('confidence_weights', {
            'prob_bull': 0.35, 'trigger_strength': 0.20, 'tech_strength': 0.20,
            'ms_strength': 0.15, 'chaos_strength': 0.10
        })
        self.trigger_normalizer = config.get('trigger_normalizer', 3.0)
        self.ms_imbalance_normalizer = config.get('ms_imbalance_normalizer', 0.8)
        self.base_stop_loss_pct = config.get('base_stop_loss_pct', 2.5)
        self.min_target_weight = config.get('min_target_weight', 0.05)
        self.target_weight_multiplier = config.get('target_weight_multiplier', 0.15)

        # 🆕 v16.9: Cleanup устаревших фаз
        self.phase_cleanup_seconds = config.get('phase_cleanup_seconds', 3600)  # 1 час

        # Кэш фаз: {ticker: {'phase': int, 'trigger_time': float, 'trigger_price': float,
        #                    'tech_confirm_time': float, 'last_entry_time': float}}
        self._phases = {}

        logger.info(f"EntryCascadingConfirmer init: enabled={self.enabled}, "
                    f"cooldown={self.cooldown}s")

    # 🆕 v16 Фаза 2.4: Группы тикеров для диверсификации
    # Тикеры одной компании (разные классы акций) считаются как одна позиция
    TICKER_GROUPS = {
        'SBER': ['SBER', 'SBERP'],
        'SNGS': ['SNGS', 'SNGSP'],
        'TATN': ['TATN', 'TATNP'],
        'MTLR': ['MTLR', 'MTLRP'],
        'MSRS': ['MSRS', 'MSRSS'],
    }

    def _get_ticker_group(self, ticker: str) -> Optional[str]:
        """Возвращает группу тикера или None."""
        for group, members in self.TICKER_GROUPS.items():
            if ticker in members:
                return group
        return None

    def screen(self, universe: List[str],
               get_hawkes_forecast, get_indicators, get_microstructure,
               get_chaos_metrics, get_price,
               held_tickers: List[str] = None,
               portfolio_state: Dict = None) -> List[EntrySignal]:
        """
        Скрининг вселенной тикеров.

        Аргументы — callback-функции (без хардкода источников данных):
          get_hawkes_forecast(ticker, horizon) -> {bull_expected, bear_expected, prob_bull, net_signal}
          get_indicators(ticker) -> {rsi, bb_position, momentum, ...}
          get_microstructure(ticker) -> {imbalance, spread_pct, volume_5m, volume_30m_avg, ...}
          get_chaos_metrics(ticker) -> {rqa_DET, rqa_L_max, kurtosis, hurst, atr_pct, ...}
          get_price(ticker) -> float
          held_tickers — список удерживаемых тикеров
          portfolio_state — {available_capital, total_capital, sector_weights}
        """
        if not self.enabled:
            return []

        # 🆕 v16.9: Cleanup устаревших фаз (trigger_time старше phase_cleanup_seconds)
        # Фаза >= PHASE_SIGNAL_DONE означает, что сигнал уже сформирован — не cleanup
        PHASE_SIGNAL_DONE = 4
        now_ts = time.time()
        stale_tickers = []
        for t, state in self._phases.items():
            trigger_time = state.get('trigger_time', 0)
            if trigger_time > 0 and (now_ts - trigger_time) > self.phase_cleanup_seconds:
                if state.get('phase', 0) < PHASE_SIGNAL_DONE:
                    stale_tickers.append(t)
        for t in stale_tickers:
            del self._phases[t]
        if stale_tickers:
            logger.debug(f"[ENTRY_F] Cleanup: удалено {len(stale_tickers)} устаревших фаз")

        held_tickers = held_tickers or []
        portfolio_state = portfolio_state or {}
        signals = []
        self._screen_log = []  # 🆕 детальный лог скрининга

        for ticker in universe:
            try:
                price = get_price(ticker)
                if not price or price <= 0:
                    continue
                if ticker in held_tickers:
                    self._screen_log.append({'ticker': ticker, 'result': 'skip', 'reason': 'уже в портфеле'})
                    continue

                # Кулдаун
                last_entry = self._phases.get(ticker, {}).get('last_entry_time', 0)
                if time.time() - last_entry < self.cooldown:
                    self._screen_log.append({'ticker': ticker, 'result': 'skip', 'reason': f'кулдаун {self.cooldown}с'})
                    continue

                # 🆕 v16 Фаза 2.2: Запрет BUY в вечерней сессии (после 19:00 MSK)
                # В тестах можно отключить через ec._test_mode_skip_evening_check = True
                if not getattr(self, '_test_mode_skip_evening_check', False):
                    evening_cutoff_hour = 19  # MSK
                    from datetime import datetime, timezone, timedelta
                    MSK = timezone(timedelta(hours=3))
                    now_msk = datetime.now(MSK)
                    if now_msk.hour >= evening_cutoff_hour:
                        self._screen_log.append({
                            'ticker': ticker, 'result': 'skip',
                            'reason': f'вечерняя сессия {now_msk.strftime("%H:%M")} MSK >= {evening_cutoff_hour}:00'
                        })
                        continue

                # 🆕 v16 Фаза 2.1: Контроль ликвидности
                # Проверяем микроструктуру: spread и volume
                ms_preview = None
                if get_microstructure:
                    try:
                        ms_preview = get_microstructure(ticker) or {}
                    except Exception:
                        ms_preview = {}

                if ms_preview:
                    spread_pct = ms_preview.get('spread_pct', 0)
                    vol_30m = ms_preview.get('volume_30m_avg', 0)
                    liq_cfg_max_spread = self.ms_cfg.get('max_spread_pct_entry', 0.5) if hasattr(self, 'ms_cfg') else 0.5
                    liq_cfg_min_vol = self.ms_cfg.get('min_volume_30m_rub', 1_000_000) if hasattr(self, 'ms_cfg') else 1_000_000

                    if spread_pct > liq_cfg_max_spread:
                        self._screen_log.append({
                            'ticker': ticker, 'result': 'skip',
                            'reason': f'low_liq spread={spread_pct:.2f}%>{liq_cfg_max_spread}%'
                        })
                        continue

                    if 0 < vol_30m < liq_cfg_min_vol:
                        self._screen_log.append({
                            'ticker': ticker, 'result': 'skip',
                            'reason': f'low_liq vol_30m={vol_30m:.0f}<{liq_cfg_min_vol}'
                        })
                        continue

                # 🆕 v16 Фаза 2.4: Диверсификация SBER/SBERP (считать как одну позицию)
                # Если SBER уже в портфеле — не покупать SBERP, и наоборот
                ticker_group = self._get_ticker_group(ticker)
                if ticker_group:
                    held_in_group = [t for t in held_tickers if self._get_ticker_group(t) == ticker_group and t != ticker]
                    if held_in_group:
                        self._screen_log.append({
                            'ticker': ticker, 'result': 'skip',
                            'reason': f'group {ticker_group} уже в портфеле: {held_in_group}'
                        })
                        continue

                signal = self._evaluate_ticker(
                    ticker, price,
                    get_hawkes_forecast, get_indicators, get_microstructure,
                    get_chaos_metrics, held_tickers, portfolio_state
                )
                if signal:
                    signals.append(signal)
                    self._screen_log.append({
                        'ticker': ticker, 'result': 'PASS',
                        'confidence': signal.confidence,
                        'reasons': signal.reasons[:3],
                    })
                else:
                    # Логируем почему отклонён
                    phase = self._phases.get(ticker, {}).get('phase', 0)
                    fc = get_hawkes_forecast(ticker, 4)
                    ind = get_indicators(ticker) or {}
                    self._screen_log.append({
                        'ticker': ticker,
                        'result': 'REJECT',
                        'phase': phase,
                        'hawkes_bull': fc.get('bull_expected', 0) if fc else 0,
                        'hawkes_prob': fc.get('prob_bull', 0) if fc else 0,
                        'rsi': ind.get('rsi', 50),
                        'momentum': ind.get('momentum', 0),
                        'bb_pos': ind.get('bb_position', 0.5),
                    })
            except Exception as e:
                logger.debug(f"Entry screen error {ticker}: {e}")

        # 🆕 Выводим топ-5 отклонённых для диагностики с разбором причин
        rejected = [s for s in self._screen_log if s.get('result') == 'REJECT']
        if rejected:
            # Группируем по причинам для сводки
            from collections import Counter
            reason_counter = Counter()
            for r in rejected:
                # Определяем главную причину отклонения
                phase = r.get('phase', 0)
                hawkes_prob = r.get('hawkes_prob', 0)
                hawkes_bull = r.get('hawkes_bull', 0)
                rsi = r.get('rsi', 50)
                mom = r.get('momentum', 0)
                bb_pos = r.get('bb_pos', 0.5)

                if phase > 0:
                    reason_counter[f'phase_{phase}_failed'] += 1
                    continue

                # Phase 0 — не прошёл ни hawkes, ни tech trigger
                # 🆕 v15.5: Правильная классификация причин
                # 1) Hawkes не обучен (нет событий)
                if hawkes_prob < 0.05 and hawkes_bull < 0.05:
                    reason_counter['hawkes_not_trained'] += 1
                # 2) Hawkes обучен, но не сработал триггер
                #    (bull < min_bull OR prob < min_prob OR bull/bear < ratio)
                elif hawkes_prob < 0.5 or hawkes_bull < 0.5:
                    reason_counter['hawkes_weak_signal'] += 1
                # 3) Технический триггер не прошёл (RSI вне диапазона)
                elif not (30 <= rsi <= 70):
                    reason_counter[f'rsi_out_of_range'] += 1
                # 4) Momentum слишком низкий
                elif abs(mom) < 0.1:
                    reason_counter['momentum_low'] += 1
                # 5) BB position вне диапазона
                elif not (0.2 <= bb_pos <= 0.8):
                    reason_counter['bb_out_of_range'] += 1
                else:
                    reason_counter['other'] += 1

            # Сводная статистика
            summary = ', '.join(f'{k}:{v}' for k, v in reason_counter.most_common())
            logger.info(f"[ENTRY_F] REJECT summary: {len(rejected)} tickers rejected. "
                       f"Reasons: {summary}")

            # Детальный лог топ-5
            for r in rejected[:5]:
                logger.info(f"[ENTRY_F] REJECT {r['ticker']}: phase={r['phase']} "
                           f"hawkes_bull={r['hawkes_bull']:.2f} prob={r['hawkes_prob']:.2f} "
                           f"RSI={r['rsi']:.0f} mom={r['momentum']:.2f} BB={r['bb_pos']:.2f}")

        # 🆕 Если НИ ОДИН тикер не прошёл — отдельное предупреждение
        if not signals and len(self._screen_log) > 0:
            skipped_count = sum(1 for s in self._screen_log if s.get('result') == 'skip')
            reject_count = len(rejected)
            logger.warning(
                f"[ENTRY_F] ⚠️ NO TICKERS PASSED. "
                f"Skipped: {skipped_count}, Rejected: {reject_count}, "
                f"Passed: 0. "
                f"Top reject reason: {summary if rejected else 'no rejects (all skipped)'}"
            )

        # Сортировка по убыванию уверенности
        signals.sort(key=lambda s: -s.confidence)

        # Применение портфельных ограничений
        max_positions = self.portfolio_cfg.get('max_positions', 5)
        signals = signals[:max_positions]
        return signals

    def _evaluate_ticker(self, ticker: str, price: float,
                         get_hawkes_forecast, get_indicators, get_microstructure,
                         get_chaos_metrics, held_tickers, portfolio_state) -> Optional[EntrySignal]:
        """Полный каскад оценки для одного тикера."""
        reasons = []

        # ─── ФАЗА 1: Hawkes-триггер ───
        fc = get_hawkes_forecast(ticker, 4)
        if not fc:
            return None
        bull = fc.get('bull_expected', 0)
        bear = fc.get('bear_expected', 0)
        prob_bull = fc.get('prob_bull', 0)
        net = fc.get('net_signal', 0)

        bull_to_bear_ratio = self.hawkes_cfg.get('bull_to_bear_ratio', 1.5)
        min_bull = self.hawkes_cfg.get('min_bull_expected', 0.5)
        min_prob = self.hawkes_cfg.get('min_prob_bull', 0.5)

        # Если Хокс не обучен (мало событий), prob_bull=0 — пропускаем триггер,
        # но не блокируем тикер: используем технические индикаторы как backup-триггер
        hawkes_triggered = (bull > bull_to_bear_ratio * max(bear, 0.01)
                            and bull > min_bull and prob_bull > min_prob)

        # Backup-триггер: технические индикаторы (если Хокс не обучен)
        # 🆕 v15.5: Поддерживаем ДВА режима технического триггера:
        #   1. Трендовый (RSI 50-70, mom > min_momentum_pct) — для растущего рынка
        #   2. Разворотный (RSI < 30, mom < -0.5) — для перепроданности (вход на отскок)
        tech_ind = get_indicators(ticker) or {}
        rsi = tech_ind.get('rsi', 50)
        mom = tech_ind.get('momentum', 0)
        bb_pos = tech_ind.get('bb_position', 0.5)

        # 🆕 Читаем min_momentum_pct из конфига (по умолчанию 0.1%)
        min_mom = self.tech_cfg.get('min_momentum_pct', 0.1)
        rsi_min = self.tech_cfg.get('rsi_min', 50)
        rsi_max = self.tech_cfg.get('rsi_max', 70)
        bb_min = self.tech_cfg.get('bb_position_min', 0.4)
        bb_max = self.tech_cfg.get('bb_position_max', 0.8)

        # 🆕 Разворотный режим (oversold bounce): RSI < 30 + BB в нижней зоне
        rsi_oversold = self.tech_cfg.get('rsi_oversold_threshold', 30)
        bb_oversold = self.tech_cfg.get('bb_position_oversold_max', 0.2)

        # Трендовый триггер (рост)
        trend_triggered = (rsi_min <= rsi <= rsi_max and mom > min_mom and bb_min <= bb_pos <= bb_max)
        # Разворотный триггер (перепроданность)
        oversold_triggered = (rsi < rsi_oversold and bb_pos < bb_oversold)
        tech_triggered = trend_triggered or oversold_triggered

        if not hawkes_triggered and not tech_triggered:
            self._phases[ticker] = {'phase': 0, 'last_entry_time': self._phases.get(ticker, {}).get('last_entry_time', 0)}
            return None

        trigger_source = 'hawkes' if hawkes_triggered else ('tech_oversold' if oversold_triggered else 'tech_trend')
        reasons.append(f"trigger={trigger_source}")

        # Триггер сработал — фиксируем время и цену
        state = self._phases.get(ticker, {})
        # 🆕 v15.5: Получаем wait_min ВСЕГДА, до проверки phase
        wait_min = self.tech_cfg.get('wait_minutes', 15)

        if state.get('phase', 0) < 1:
            state['phase'] = 1
            state['trigger_time'] = time.time()
            state['trigger_price'] = price
            state['trigger_indicators'] = tech_ind
            state['trigger_source'] = trigger_source
            self._phases[ticker] = state
            logger.debug(f"[ENTRY] {ticker} phase 1 triggered ({trigger_source}): "
                        f"bull={bull:.2f}, prob={prob_bull:.2f}, RSI={rsi:.0f}, mom={mom:.2f}")
            # 🆕 v14.5: При wait_minutes=0 не выходим — продолжаем к phase 2
            if wait_min > 0:
                return None  # ждём техническое подтверждение

        # ─── ФАЗА 2: Техническое подтверждение ───
        # wait_min уже получен выше
        # Если wait_min=0 — пропускаем проверку времени (мгновенное подтверждение)
        if wait_min > 0 and time.time() - state.get('trigger_time', 0) < wait_min * 60:
            return None  # ещё не прошло время

        ind_now = tech_ind  # уже получили выше
        rsi = ind_now.get('rsi', 50)
        mom_now = ind_now.get('momentum', 0)
        bb_pos = ind_now.get('bb_position', 0.5)

        # Проверка: цена выросла минимум на min_momentum_pct с момента триггера
        trigger_price = state.get('trigger_price', price)
        momentum_pct = (price - trigger_price) / trigger_price * 100 if trigger_price > 0 else 0

        rsi_min = self.tech_cfg.get('rsi_min', 50)
        rsi_max = self.tech_cfg.get('rsi_max', 70)
        bb_min = self.tech_cfg.get('bb_position_min', 0.4)
        bb_max = self.tech_cfg.get('bb_position_max', 0.8)
        min_mom = self.tech_cfg.get('min_momentum_pct', 0.1)

        # Если wait_min=0, не требуем momentum_pct (он будет ~0)
        # 🆕 v15.5: Если триггер был oversold — phase 2 проверяет что RSI/BB всё ещё в oversold зоне
        # (цена не отскочила обратно вверх)
        trigger_src = state.get('trigger_source', 'tech_trend')

        if trigger_src == 'tech_oversold':
            # Для oversold-входа: проверяем что перепроданность сохраняется
            # (RSI всё ещё < oversold threshold, BB всё ещё в нижней зоне)
            tech_ok = (rsi < rsi_oversold and bb_pos < bb_oversold)
            if not tech_ok and wait_min > 0:
                # Цена отскочила — требуем momentum_pct для подтверждения отскока
                tech_ok = (rsi_min <= rsi <= rsi_max and
                           bb_min <= bb_pos <= bb_max and
                           momentum_pct >= min_mom)
        elif wait_min > 0:
            # Трендовый режим с ожиданием — требуем momentum_pct
            tech_ok = (rsi_min <= rsi <= rsi_max and
                       bb_min <= bb_pos <= bb_max and
                       momentum_pct >= min_mom)
        else:
            # Трендовый режим без ожидания — мгновенное подтверждение
            tech_ok = (rsi_min <= rsi <= rsi_max and
                       bb_min <= bb_pos <= bb_max)
        if not tech_ok:
            self._phases[ticker] = {'phase': 0, 'last_entry_time': self._phases.get(ticker, {}).get('last_entry_time', 0)}
            return None

        state['phase'] = 2
        state['tech_confirm_time'] = time.time()
        state['tech_indicators'] = ind_now
        self._phases[ticker] = state
        logger.debug(f"[ENTRY] {ticker} phase 2 tech OK: RSI={rsi:.0f}, BB={bb_pos:.2f}, "
                    f"mom={momentum_pct:.2f}%")

        # ─── ФАЗА 3: Микроструктурное подтверждение ───
        ms_wait_min = self.ms_cfg.get('wait_minutes', 5)
        if ms_wait_min > 0 and time.time() - state.get('tech_confirm_time', 0) < ms_wait_min * 60:
            return None

        ms = get_microstructure(ticker)
        if ms is None:
            ms = {}
        imbalance = ms.get('imbalance', 0)
        spread_pct = ms.get('spread_pct', 1.0)
        vol_5m = ms.get('volume_5m', 0)
        vol_baseline = ms.get('volume_30m_avg', 1)
        vol_ratio = vol_5m / vol_baseline if vol_baseline > 0 else 0

        ms_min_imb = self.ms_cfg.get('min_imbalance', 0.2)
        ms_max_spread = self.ms_cfg.get('max_spread_pct', 0.3)
        ms_min_vol_ratio = self.ms_cfg.get('min_volume_ratio_5m', 1.2)
        # 🆕 v15.6: Если volume_5m или volume_30m_avg равны 0 — данные MS недоступны
        ms_min_imb_abs = self.ms_cfg.get('min_imbalance_abs', 0.05)  # минимум |imbalance|

        # Если данных микроструктуры нет (вне торгов) — пропускаем фазу 3
        ms_data_available = (ms.get('bid', 0) > 0 or ms.get('offer', 0) > 0
                            or ms.get('imbalance', 0) != 0)

        # 🆕 v15.6: volume_5m=0 или volume_30m_avg=0 — данные MS недоступны
        ms_volume_available = (vol_5m > 0 and vol_baseline > 0)

        if not ms_data_available or not ms_volume_available:
            # Вне торгов или нет данных MS — пропускаем микроструктурную фазу
            ms_ok = True
            imbalance = 0
            spread_pct = 0
            vol_ratio = 1.0
            reasons.append("ms_skipped_no_data")
        else:
            # 🆕 v15.6: Ослабленная проверка:
            # Требуем ТОЛЬКО spread и volume (не блокируем по imbalance на падающем рынке)
            # imbalance используется только для расчёта confidence, не для блокировки
            ms_ok = (spread_pct <= ms_max_spread and
                     vol_ratio >= ms_min_vol_ratio)
            # Дополнительно: если |imbalance| очень мал (боковик) — тоже OK
            if not ms_ok and abs(imbalance) < ms_min_imb_abs:
                ms_ok = True
                reasons.append("ms_low_imbalance_ok")

        if not ms_ok:
            # 🆕 v15.6: Добавили debug-лог для phase 3 failure (раньше был silent)
            logger.debug(f"[ENTRY] {ticker} phase 3 ms FAIL: imb={imbalance:.2f} "
                        f"(need>={ms_min_imb}), spread={spread_pct:.3f} (need<={ms_max_spread}), "
                        f"vol_ratio={vol_ratio:.2f} (need>={ms_min_vol_ratio}), "
                        f"ms_data_available={ms_data_available}")
            self._phases[ticker] = {'phase': 0, 'last_entry_time': self._phases.get(ticker, {}).get('last_entry_time', 0)}
            return None

        state['phase'] = 3
        self._phases[ticker] = state
        logger.debug(f"[ENTRY] {ticker} phase 3 ms OK: imb={imbalance:.2f}, "
                    f"spread={spread_pct:.3f}, vol_ratio={vol_ratio:.2f}")

        # ─── ФАЗА 4: Хаос-фильтр ───
        chaos = get_chaos_metrics(ticker)
        if chaos is None:
            chaos = {}
        det = chaos.get('rqa_DET', 0.3)
        l_max = chaos.get('rqa_L_max', 6)
        kurt = chaos.get('kurtosis', 10)
        hurst = chaos.get('hurst', 0.5)
        atr_pct = chaos.get('atr_pct', 1.0)

        # Branching ratio из Хокса
        try:
            from core.hawkes_signal import hawkes_signal
            eta_bull = hawkes_signal.get_branching_ratio(ticker, 'bull') if hawkes_signal else 0
        except Exception:
            eta_bull = 0

        min_det = self.chaos_cfg.get('min_rqa_DET', 0.25)
        min_lmax = self.chaos_cfg.get('min_rqa_L_max', 4)
        max_kurt = self.chaos_cfg.get('max_kurtosis', 100)
        min_eta = self.chaos_cfg.get('min_hawkes_branching_ratio', 0.3)

        # 🆕 v15.7: Для oversold-триггеров eta_bull обычно низкий (рынок падает)
        # → не блокируем по eta_bull если триггер был oversold
        trigger_src_for_chaos = state.get('trigger_source', 'tech_trend')
        is_oversold_trigger = (trigger_src_for_chaos == 'tech_oversold')

        # Если eta_bull=0 (Хокс не обучен) — не блокируем по этому критерию
        chaos_fail = (det < min_det or l_max < min_lmax or kurt > max_kurt)
        # 🆕 v15.7: Проверяем eta_bull только для НЕ-oversold триггеров
        if not is_oversold_trigger and eta_bull > 0 and eta_bull < min_eta:
            chaos_fail = True
        if chaos_fail:
            self._phases[ticker] = {'phase': 0, 'last_entry_time': self._phases.get(ticker, {}).get('last_entry_time', 0)}
            logger.debug(f"[ENTRY] {ticker} chaos filter fail: DET={det:.2f}, "
                        f"L_max={l_max}, kurt={kurt:.1f}, eta_bull={eta_bull:.2f}, "
                        f"trigger={trigger_src_for_chaos} (oversold_eta_skip={is_oversold_trigger})")
            return None

        # ─── ВСЕ ФАЗЫ ПРОЙДЕНЫ → ФОРМИРОВАНИЕ СИГНАЛА ───

        # Корреляционная проверка с удерживаемыми
        max_corr = self.portfolio_cfg.get('max_correlation_with_held', 0.7)
        corr_matrix = portfolio_state.get('correlation_matrix', {})
        max_corr_actual = 0
        for held in held_tickers:
            max_corr_actual = max(max_corr_actual,
                                  abs(corr_matrix.get(ticker, {}).get(held, 0)))
        if max_corr_actual > max_corr:
            self._phases[ticker] = {'phase': 0, 'last_entry_time': self._phases.get(ticker, {}).get('last_entry_time', 0)}
            logger.debug(f"[ENTRY] {ticker} correlation too high: {max_corr_actual:.2f}")
            return None

        # Секторная диверсификация
        max_per_sector = self.portfolio_cfg.get('max_per_sector', 2)
        sector_weights = portfolio_state.get('sector_weights', {})
        ticker_sector = portfolio_state.get('ticker_sectors', {}).get(ticker, 'unknown')
        if sector_weights.get(ticker_sector, 0) >= max_per_sector:
            self._phases[ticker] = {'phase': 0, 'last_entry_time': self._phases.get(ticker, {}).get('last_entry_time', 0)}
            logger.debug(f"[ENTRY] {ticker} sector {ticker_sector} full")
            return None

        # Уверенность
        trigger_strength = min(1.0, bull / self.trigger_normalizer) if hawkes_triggered else 0.5
        tech_strength = (1 - abs(rsi - 57) / 25) * (1 - abs(bb_pos - 0.55) / 0.4)
        ms_strength = min(1.0, abs(imbalance) / self.ms_imbalance_normalizer) if ms_data_available else 0.5
        chaos_strength = (det + min(l_max / 20, 1)) / 2
        confidence = (prob_bull * self.conf_weights.get('prob_bull', 0.35) +
                      trigger_strength * self.conf_weights.get('trigger_strength', 0.20) +
                      tech_strength * self.conf_weights.get('tech_strength', 0.20) +
                      ms_strength * self.conf_weights.get('ms_strength', 0.15) +
                      chaos_strength * self.conf_weights.get('chaos_strength', 0.10))
        confidence = max(0, min(1, confidence))

        # 🆕 v16 Фаза 3.1: 5-я фаза — ML-стратегия прогноза цены
        try:
            from core.ml_price_strategy import ml_price_strategy
            if ml_price_strategy and ml_price_strategy.enabled:
                # Нужен price_predictor — пробуем получить из portfolio_state
                price_predictor = portfolio_state.get('price_predictor')
                if price_predictor:
                    ml_result = ml_price_strategy.evaluate(
                        ticker=ticker,
                        current_price=price,
                        price_predictor=price_predictor,
                        chaos_metrics=chaos,
                        hawkes_forecast={'bull_expected': bull, 'bear_expected': 0,
                                         'prob_bull': prob_bull, 'net_signal': bull},
                    )

                    if ml_result['action'] == 'BLOCK':
                        self._phases[ticker] = {'phase': 0,
                                              'last_entry_time': self._phases.get(ticker, {}).get('last_entry_time', 0)}
                        logger.debug(f"[ENTRY] {ticker} ML BLOCK: {ml_result['reason']}")
                        return None

                    if ml_result['action'] == 'BOOST':
                        old_conf = confidence
                        confidence = min(1.0, confidence * ml_result['confidence_multiplier'])
                        reasons.append(f"ml_boost={ml_result['predicted_return_pct']:+.1f}%")
                        logger.debug(f"[ENTRY] {ticker} ML BOOST: {old_conf:.2f} → {confidence:.2f} "
                                    f"({ml_result['reason']})")
        except Exception as e:
            logger.debug(f"ML strategy integration error: {e}")

        min_conf = self.portfolio_cfg.get('min_confidence', 0.5)  # 🆕 v16 Фаза 2.3: 0.6 → 0.5
        if confidence < min_conf:
            self._phases[ticker] = {'phase': 0, 'last_entry_time': self._phases.get(ticker, {}).get('last_entry_time', 0)}
            logger.debug(f"[ENTRY] {ticker} confidence {confidence:.2f} < {min_conf}")
            return None

        # Stop-loss с учётом тяжёлых хвостов
        kurt_penalty = min(5.0, (kurt - 3) * 0.02)
        stop_loss_pct = self.base_stop_loss_pct + kurt_penalty

        # Take-profit = 2 × stop (RR = 2:1)
        take_profit_pct = stop_loss_pct * 2

        # Целевой вес
        max_weight = self.portfolio_cfg.get('max_position_weight_pct', 20) / 100
        target_weight = min(max_weight, max(self.min_target_weight, confidence * self.target_weight_multiplier))

        # Сбрасываем фазу
        self._phases[ticker] = {'phase': 0, 'last_entry_time': time.time()}

        reasons.extend([
            f"hawkes_bull={bull:.2f}",
            f"prob_bull={prob_bull:.2f}",
            f"rsi={rsi:.0f}",
            f"mom_15m={momentum_pct:.2f}%",
            f"imbalance={imbalance:.2f}",
            f"DET={det:.2f}",
            f"L_max={l_max}",
            f"kurt={kurt:.0f}",
            f"eta_bull={eta_bull:.2f}",
        ])
        logger.info(f"✅ [ENTRY] {ticker} ALL PHASES OK | "
                    f"confidence={confidence:.2f} | stop={stop_loss_pct:.2f}% | "
                    f"target_weight={target_weight*100:.1f}%")

        return EntrySignal(
            ticker=ticker,
            confidence=confidence,
            hawkes_prob_bull=prob_bull,
            hawkes_net=net,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            target_weight=target_weight,
            reasons=reasons,
        )

    def reset_ticker(self, ticker: str):
        """Сброс фазы при закрытии позиции."""
        if ticker in self._phases:
            del self._phases[ticker]

    def get_stats(self) -> Dict:
        return {
            'enabled': self.enabled,
            'tracked_tickers': len(self._phases),
            'phase_distribution': dict(defaultdict(int, {
                self._phases[t].get('phase', 0): 1 for t in self._phases
            })),
        }


# ─── СИНГЛТОН ───
_config = {}
try:
    _settings_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'settings.json'
    )
    if os.path.exists(_settings_path):
        with open(_settings_path, 'r', encoding='utf-8') as _f:
            _settings = json.load(_f)
            _config = _settings.get('entry_cascading', {})
except Exception as e:
    logger.warning(f"Не удалось загрузить entry_cascading config: {e}")

entry_confirmer = EntryCascadingConfirmer(_config)
