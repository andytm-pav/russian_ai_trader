"""
LLM-коуч для обучения торговой модели
"""
import json
import re
import time
from datetime import datetime
from typing import Dict, Optional
from utils.logger import get_logger
from models.llm_providers import create_provider

logger = get_logger("LLM_COACH")


class LLMCoach:
    """Коуч на основе LLM для помощи RL-модели"""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.coach_interval_cycles = config.get("coach_interval_cycles", 20)
        self.coach_action_weight = config.get("coach_action_weight", 0.3)
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.6)
        self.max_coach_experiences = config.get("max_coach_experiences", 1000)
        self.max_consecutive_timeouts = config.get("max_consecutive_timeouts", 5)
        self.cooldown_minutes = config.get("cooldown_after_timeouts_minutes", 10)

        self.provider = None
        self.consecutive_timeouts = 0
        self.cooldown_until = None
        self.coach_experiences = []

        if self.enabled:
            self._init_provider()

        logger.info(f"LLMCoach инициализирован (enabled={self.enabled})")

    def _init_provider(self):
        """Инициализация провайдера"""
        try:
            provider_config = self.config.get("provider", {})
            self.provider = create_provider(provider_config)
            if self.provider.is_available():
                logger.info(f"Провайдер {provider_config.get('type')} готов")
            else:
                logger.warning("Провайдер недоступен")
        except Exception as e:
            logger.error(f"Ошибка инициализации провайдера: {e}")
            self.provider = None

    def is_available(self) -> bool:
        """Проверяет, доступен ли коуч"""
        if not self.enabled:
            return False
        if self.cooldown_until and time.time() < self.cooldown_until:
            return False
        return self.provider is not None and self.provider.is_available()

    def get_coach_advice(self, snapshot: Dict) -> Optional[Dict]:
        """Получает совет от LLM-коуча"""
        if not self.is_available():
            return None

        # Обновляем таймаут провайдера из конфига
        provider_config = self.config.get("provider", {})
        if self.provider and "timeout" in provider_config:
            self.provider.timeout = provider_config["timeout"]

        prompt = self._build_context_prompt(snapshot)
        action = snapshot.get("suggested_action", "HOLD")
        rule = snapshot.get("rule_triggered", "7")

        full_prompt = f"""{prompt}

Сработало правило №{rule}.

Рекомендуемое действие: {action}

Согласен ли ты с этим решением? Если нет, предложи другое.
Ответь СТРОГО в JSON без лишнего текста:
{{"action": "{action}", "confidence": 0.0-1.0, "rule_triggered": "{rule}", "rationale": "краткое обоснование"}}"""

        try:
            start_time = time.time()
            logger.debug(f"Промпт коучу:\n{full_prompt}")
            response = self.provider.generate(full_prompt)
            elapsed = time.time() - start_time

            advice = self._parse_response(response)
            if advice:
                self.consecutive_timeouts = 0
                logger.debug(f"Коуч ответил за {elapsed:.1f}с: {advice.get('action')} conf={advice.get('confidence')}")
                return advice
            else:
                return None

        except (TimeoutError, Exception) as e:
            self.consecutive_timeouts += 1
            logger.warning(f"Коуч не ответил: {e}. Таймаутов подряд: {self.consecutive_timeouts}")

            if self.consecutive_timeouts >= self.max_consecutive_timeouts:
                self.cooldown_until = time.time() + self.cooldown_minutes * 60
                logger.warning(f"Коуч отключён на {self.cooldown_minutes} мин из-за {self.consecutive_timeouts} таймаутов")
                self.consecutive_timeouts = 0

            return None

    def _build_context_prompt(self, snapshot: Dict) -> str:
        """Формирует промпт из снимка состояния"""
        ticker = snapshot.get("ticker", "N/A")
        price = snapshot.get("price", 0)
        rsi = snapshot.get("rsi", 50)
        bb_pos = snapshot.get("bb_position", 0.5)
        momentum = snapshot.get("momentum", 0)
        has_pos = snapshot.get("has_position", False)
        pnl = snapshot.get("pnl_pct", 0)
        imoex = snapshot.get("imoex", 0)
        imoex_change = snapshot.get("imoex_change", 0)
        brent = snapshot.get("brent", 0)
        brent_change = snapshot.get("brent_change", 0)
        rvi = snapshot.get("rvi", 0)
        usd_rub = snapshot.get("usd_rub", 0)
        news_title = snapshot.get("news_title", "нет новостей")
        news_sentiment = snapshot.get("news_sentiment", 0)
        positions_count = snapshot.get("positions_count", 0)
        max_positions = snapshot.get("max_positions", 10)
        cash = snapshot.get("cash", 0)
        exposure = snapshot.get("exposure", 0)
        stop_loss = snapshot.get("stop_loss", 0)
        take_profit = snapshot.get("take_profit", 0)
        volume = snapshot.get("volume", 0)

        # Цена и позиция
        if has_pos:
            pos_line = (f"У вас ОТКРЫТА позиция по {ticker}. "
                        f"Цена: {price:.2f}₽. PnL: {pnl:+.1f}%. "
                        f"Стоп-лосс: {stop_loss:.2f}₽. Тейк-профит: {take_profit:.2f}₽.")
        else:
            pos_line = f"У вас НЕТ позиции по {ticker}. Текущая цена: {price:.2f}₽."

        # Портфель
        port_line = (f"Портфель: {positions_count}/{max_positions} позиций, "
                     f"свободный кэш: {cash:.0f}₽, риск (exposure): {exposure:.1%}.")

        # Рынок
        market_line = (f"IMOEX: {imoex:.0f} ({imoex_change:+.1f}%), "
                       f"Brent: ${brent:.1f} ({brent_change:+.1f}%), "
                       f"RVI: {rvi:.1f}, USD/RUB: {usd_rub:.2f}.")

        # Новость
        sentiment_word = "позитивная" if news_sentiment > 0.1 else "негативная" if news_sentiment < -0.1 else "нейтральная"
        news_line = f"Ключевая новость ({sentiment_word}): {news_title}"

        # Объём
        volume_line = f"Объём торгов: {volume:,.0f}₽" if volume > 0 else ""

        return f"""Ты — ассистент трейдера. Проверь, правильно ли сработало правило.

    ДАННЫЕ:
    - {pos_line}
    - {port_line}
    - RSI={rsi:.1f}, полоса Боллинджера={bb_pos:.2f}, momentum={momentum:+.1f}%
    - {market_line}
    - {news_line}
    {f'- {volume_line}' if volume_line else ''}

    АНАЛИЗ:"""

    def _parse_response(self, text: str) -> Optional[Dict]:
        """Извлекает JSON из ответа модели"""
        try:
            json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', text)
            if json_match:
                advice = json.loads(json_match.group())
                if advice.get("confidence", 0) < self.min_confidence_threshold:
                    logger.debug(f"Совет отклонён: confidence={advice.get('confidence')} < {self.min_confidence_threshold}")
                    return None
                return advice
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Ошибка парсинга ответа коуча: {e}")
        return None

    def add_coach_experience(self, experience: Dict):
        """Добавляет опыт коуча в буфер"""
        if len(self.coach_experiences) >= self.max_coach_experiences:
            self.coach_experiences.pop(0)
        self.coach_experiences.append(experience)

    def _precompute_rule(self, snapshot: Dict) -> tuple:
        """Предвычисляет правило на основе данных (Python, не LLM)"""
        rsi = snapshot.get("rsi", 50)
        bb_pos = snapshot.get("bb_position", 0.5)
        pnl = snapshot.get("pnl_pct", 0)
        has_pos = snapshot.get("has_position", False)
        imoex_change = snapshot.get("imoex_change", 0)

        if has_pos and pnl > 3 and (rsi > 65 or bb_pos > 0.8):
            return 1, "SELL", f"позиция с прибылью {pnl:.1f}%, RSI={rsi:.1f}, BB={bb_pos:.2f}"
        elif has_pos and pnl < -2:
            return 2, "SELL", f"позиция с убытком {pnl:.1f}%, ограничить убыток"
        elif rsi < 35 and bb_pos < 0.2:
            return 3, "BUY", f"RSI={rsi:.1f} (перепродан), BB={bb_pos:.2f}, возможен отскок"
        elif rsi > 75:
            return 4, "HOLD", f"RSI={rsi:.1f} > 75, BUY запрещён"
        elif has_pos and pnl > 1 and imoex_change < -0.5:
            return 5, "SELL", f"прибыль {pnl:.1f}%, рынок падает {imoex_change:.1f}%"
        elif not has_pos and imoex_change < -1:
            return 6, "HOLD", f"нет позиции, рынок падает {imoex_change:.1f}%"
        else:
            return 7, "HOLD", "сигналы разнонаправленные"