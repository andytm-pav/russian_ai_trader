"""
Главный модуль Smart Broker с интеграцией всех компонентов
"""

import json
import torch
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from core.core_news_trader import NewsTraderCore
from core.core_technical_trader import TechnicalTraderCore
from core.risk_manager import RiskManager
from core.trading_hours_scheduler import TradingScheduler
from fetchers.moex_fetcher import MoexFetcher
from fetchers.rss_fetcher import RSSFetcher
from utils.portfolio_manager import PortfolioManager
from utils.logger import get_logger

from models.trader_model import trader_model_instance
from models.trader_model import MIN_EXPERIENCES_FOR_LEARNING
logger = None

class SmartPortfolioBroker:
    """Умный брокер с интеграцией всех торговых ядер"""

    def __init__(self, settings: Dict):
        self.rl_config = None
        self.settings = settings
        self.moex = MoexFetcher()
        self.rss_fetcher = RSSFetcher()
        self.news_core = NewsTraderCore()
        self.technical_core = TechnicalTraderCore()
        self.risk_manager = RiskManager()
        self.scheduler = TradingScheduler()
        self.portfolio = PortfolioManager()
        self.model = trader_model_instance

        global logger
        # Инициализируем глобальный логгер с настройками
        logger = get_logger('SMART_BROKER')



        # ✅ ИНИЦИАЛИЗАЦИЯ НОВЫХ КОНФИГОВ
        self.profit_config = settings.get("profit_optimization", {})
        self.rl_config = self._load_rl_config()

        # ✅ ЗАГРУЖАЕМ КОНФИГИ СЕНТИМЕНТА
        self.sentiment_config = self.rl_config.get("sentiment_integration", {})
        self.market_sentiment_weight = self.sentiment_config.get("market_sentiment_weight", 0.3)
        self.ticker_sentiment_weight = self.sentiment_config.get("ticker_sentiment_weight", 0.4)
        self.reward_sentiment_bonus = self.sentiment_config.get("reward_sentiment_bonus", 0.5)

        # ✅ ИНИЦИАЛИЗАЦИЯ TRAINER (исправление ошибки)
        self.trainer = None
        if settings.get("enable_background_training", True):
            from models.trainer import model_trainer_instance
            self.trainer = model_trainer_instance

        # ✅ ЗАГРУЗКА КОНСТАНТ ДЛЯ ПРИБЫЛИ
        self.extreme_pnl_threshold = self.profit_config.get("extreme_pnl_threshold", 0.08)
        self.max_reward = self.profit_config.get("max_reward", 1000.0)
        self.min_reward = self.profit_config.get("min_reward", -1000.0)
        self.reward_scaling_profit = self.profit_config.get("reward_scaling_profit", 100.0)
        self.fast_learning_cycles = self.profit_config.get("fast_learning_cycles", 5)
        self.strategy_adaptation_cycles = self.profit_config.get("strategy_adaptation_cycles", 20)

        # Торговые переменные
        self.last_cycle_time = 0
        self.cycle_count = 0
        self.trading_enabled = True
        self.signals_cache = []
        self.current_tickers = []

        # Запуск компонентов
        self._initialize_components()

        print(f"[SmartBroker] Инициализирован. Капитал: {self.portfolio.cash:,.0f}₽")
        print(f"[SmartBroker] Модель sentiment: {self.model.market_sentiment:.3f}")
        print(f"[SmartBroker] Макс. позиций: {settings['max_positions']}")
        print(f"[SmartBroker] Конфиг прибыли загружен: {len(self.profit_config) > 0}")
        print(f"[SmartBroker] RL конфиг загружен: {len(self.rl_config) > 0}")

        self.ticker_states = {}  # {ticker: last_state}
        self.pending_experiences = []  # Опыты для обучения
        self.strategy_tracker = defaultdict(list)  # История стратегий по тикерам

    def _load_rl_config(self) -> Dict:
        """Загрузка RL конфига"""
        try:
            with open("config/rl_config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Не удалось загрузить RL конфиг: {e}")
            return {}


    def _execute_trading_decisions(self, signals: List[Dict],
                                   prices: Dict[str, float], securities: Dict):
        """Исполнение торговых решений с RL"""
        logger.debug(f"[DEBUG] Исполнение {len(signals)} сигналов")

        executed_count = 0

        for signal in signals[:5]:  # Обрабатываем только топ-5 сигналов
            ticker = signal['ticker']
            action_str = signal['action']  # 'BUY', 'SELL'
            confidence = signal['confidence']

            if ticker not in prices:
                continue

            price = prices[ticker]
            action_idx = {'BUY': 0, 'HOLD': 1, 'SELL': 2}[action_str]

            # Логируем каждый сигнал
            logger.debug(f"[DEBUG] Сигнал: {ticker} {action_str} conf={confidence:.2f}")

            # 1. Получаем или создаем состояние для тикера
            if ticker in self.ticker_states:
                current_state = self.ticker_states[ticker]
            else:
                # Создаем начальное состояние
                current_state = self._create_initial_state(ticker, price, securities.get(ticker, {}))
                self.ticker_states[ticker] = current_state

            if action_str == 'BUY':
                # 2. Выбор стратегии для покупки
                strategy, stop_loss, take_profit = self._select_buy_strategy(
                    ticker, price, confidence, current_state
                )

                # 3. Исполнение
                quantity, risk_amount = self.risk_manager.calculate_position_size(
                    ticker=ticker,
                    price=price,
                    stop_loss=stop_loss,
                    confidence=confidence,
                )

                logger.debug(f"[DEBUG] {ticker}: quantity={quantity}, risk_amount={risk_amount}")

                # Применяем стратегический множитель к результату RiskManager
                if quantity > 0:
                    # Проверяем наличие стратегии в модели
                    if hasattr(self.model, 'strategies') and strategy in self.model.strategies:
                        strategy_multiplier = self.model.strategies[strategy].get('risk_multiplier', 1.0)
                    else:
                        strategy_multiplier = 1.0
                        logger.warning(f"Стратегия {strategy} не найдена в модели")

                    # Безопасное применение множителя
                    if strategy_multiplier != 1.0:
                        # Используем округление для точности
                        adjusted_quantity = max(1, int(round(quantity * strategy_multiplier)))

                        # Проверка ликвидности (предотвращение превышения ADV)
                        security_info = securities.get(ticker, {})
                        adv = security_info.get('volume', 0)
                        if adv > 0:
                            max_by_adv = int(adv * 0.05)  # Макс 5% от дневного объема
                            adjusted_quantity = min(adjusted_quantity, max_by_adv)
                            logger.debug(f"Лимит ликвидности {ticker}: {adjusted_quantity} акций")

                        quantity = adjusted_quantity

                    logger.info(f"Стратегия {strategy}: {quantity} акций (множитель: {strategy_multiplier:.2f}×)")

                sentiment_data = self.news_core.get_enhanced_sentiment(ticker)

                logger.debug(f"[DEBUG] {ticker}: cash={self.portfolio.cash}, price={price}, total={quantity * price}")

                if quantity > 0 and self.portfolio.buy(ticker, quantity, price):
                    # 4. Запись RL-опыта с сентиментом
                    self._record_rl_experience(
                        ticker=ticker,
                        state=current_state,
                        action=action_idx,
                        strategy=strategy,
                        price=price,
                        quantity=quantity,
                        sentiment_data=sentiment_data  # Добавляем сентимент
                    )

                    # 5. Обновляем позицию информацией о стратегии
                    self.portfolio.positions[ticker]['strategy'] = strategy
                    self.portfolio.positions[ticker]['stop_loss'] = stop_loss
                    self.portfolio.positions[ticker]['take_profit'] = take_profit
                    self.portfolio.positions[ticker]['entry_state'] = current_state.cpu().numpy().tolist()

                    executed_count += 1
                    logger.debug(f"[DEBUG] Куплено: {ticker} {quantity} @ {price:.2f}")


            elif action_str == 'SELL':
                if ticker in self.portfolio.positions:
                    # 6. Продажа - завершение RL-опыта
                    logger.debug(f"[DEBUG] Продажа {ticker}, есть позиция")
                    self._complete_rl_experience(ticker, price)

                    # Исполнение продажи
                    pos = self.portfolio.positions[ticker]
                    qty = pos['qty'] // 2 if pos['qty'] > 1 else pos['qty']

                    if qty > 0 and self.portfolio.sell(ticker, qty, price):
                        # Обновляем стратегию
                        strategy = pos.get('strategy', 'balanced')
                        executed_count += 1
                        logger.debug(f"[DEBUG] Продано: {ticker} {qty} @ {price:.2f}")

                        # Записываем результат стратегии
                        if hasattr(self.model, 'record_strategy_outcome'):
                            self.model.record_strategy_outcome(
                                strategy_name=strategy,
                                action='SELL',
                                pnl=(price - pos['avg_price']) * qty,
                                hold_time=time.time() - pos.get('buy_time', time.time())
                            )
        logger.debug(f"[DEBUG] Всего исполнено сделок: {executed_count}")
        return executed_count

    def _record_rl_experience(self, ticker: str, state: torch.Tensor,
                              action: int, strategy: str, price: float,
                              quantity: int, sentiment_data: Dict = None):
        """Запись опыта с маркировкой важности"""
        experience = {
            'ticker': ticker,
            'start_state': state.cpu(),
            'action': action,
            'strategy': strategy,
            'entry_price': price,
            'entry_time': time.time(),
            'quantity': quantity,
            'sentiment_data': sentiment_data,
            'completed': False,
            'is_priority': self._is_priority_experience(sentiment_data)  # Новое поле
        }

        # Также отправляем в trainer если это приоритетный опыт
        if hasattr(self, 'trainer') and experience['is_priority']:
            self.trainer.add_priority_experience(experience)

    def _is_priority_experience(self, sentiment_data: Optional[Dict]) -> bool:
        """Определение приоритетности опыта через конфиги"""
        if not sentiment_data:
            return False

        # ✅ ИСПОЛЬЗУЕМ КОНФИГ СТРАТЕГИЙ
        strategy_config = self.model.strategy_config
        priority_config = strategy_config.get("strategy_selection", {}).get("priority_learning", {})

        if not priority_config.get("enable_priority_training", True):
            return False

        # ✅ КОНСТАНТЫ ИЗ КОНФИГА
        sentiment_threshold = priority_config.get("sentiment_threshold", 0.6)
        news_count_threshold = priority_config.get("news_count_threshold", 3)

        # ✅ РАСЧЕТ ПАРАМЕТРОВ
        sentiment_strength = abs(sentiment_data.get('sentiment', 0))
        impact_level = sentiment_data.get('impact_level', 'low_impact')
        news_count = sentiment_data.get('news_count', 0)

        # ✅ КРИТЕРИИ ПРИОРИТЕТНОСТИ
        is_priority = (
                sentiment_strength > sentiment_threshold or
                impact_level in priority_config.get("impact_levels", ["high_impact"]) or
                news_count >= news_count_threshold
        )

        return is_priority

    def _complete_rl_experience(self, ticker: str, exit_price: float):
        """Завершение RL-опыта при продаже С ИСПРАВЛЕНИЕМ АРГУМЕНТОВ"""
        logger.debug (f"[DEBUG] Завершение RL опыта для {ticker} @ {exit_price}")
        logger.debug (f"[DEBUG] pending_experiences: {len(self.pending_experiences)}")
        # Находим незавершенный опыт для этого тикера
        for exp in self.pending_experiences:
            if exp['ticker'] == ticker and not exp['completed']:
                # Рассчитываем reward
                pnl = (exit_price - exp['entry_price']) * exp['quantity']
                hold_time = (time.time() - exp['entry_time']) / 3600  # в часах

                # Создаем следующее состояние
                next_state = self._create_next_state(ticker, exit_price)


                # Нормализованный reward
                reward = self._calculate_reward(pnl, hold_time, exp['strategy'])


                # ✅ ИСПРАВЛЕННЫЙ ВЫЗОВ С ПРАВИЛЬНОЙ ПОСЛЕДОВАТЕЛЬНОСТЬЮ
                self.model.remember_experience(
                    state=exp['start_state'].to(self.model.device),  # ✅ гарантируем device
                    action=exp['action'],
                    reward=reward,
                    next_state=next_state,
                    done=True,
                    news_features=None,  # ✅ явно указываем None
                )


                # ✅ ДОПОЛНИТЕЛЬНО: обновляем market_conditions в записи модели если нужно
                if 'sentiment_data' in exp:

                    # ✅ ПОЛУЧАЕМ РЫНОЧНЫЙ СЕНТИМЕНТ НА МОМЕНТ СДЕЛКИ
                    market_sentiment_data = self.news_core.get_market_sentiment()
                    market_sentiment = market_sentiment_data.get('sentiment', 0.0)

                    # Записываем результат в модель с market_sentiment
                    self.model.record_trade_outcome(
                        ticker=ticker,
                        action='SELL',
                        entry_price=exp['entry_price'],
                        exit_price=exit_price,
                        hold_time=hold_time,
                        news_sentiment=exp['sentiment_data'].get('sentiment', 0),
                        market_conditions={
                            'strategy': exp['strategy'],
                            'confidence': exp.get('confidence', 0.5),
                            'reward': reward,
                            'pnl': pnl,
                            'market_sentiment': market_sentiment  # ✅ ДОБАВЛЯЕМ
                        },
                        strategy=exp['strategy'],
                        market_sentiment=market_sentiment  # ✅ ПЕРЕДАЕМ В НОВЫЙ ПАРАМЕТР
                    )

                exp['completed'] = True
                logger.debug(f"RL опыт завершен: {ticker}, reward={reward:.3f}, pnl={pnl:.2f}")
                break

    def _calculate_reward(self, pnl: float, hold_time: float, strategy: str) -> float:
        """Расчет награды ОПТИМИЗИРОВАННЫЙ ДЛЯ МАКСИМАЛЬНОЙ ПРИБЫЛИ"""

        # ✅ ИСПОЛЬЗУЕМ КОНФИГИ
        reward_calc = self.rl_config.get("reward_calculation", {})

        # 1. БАЗОВАЯ НАГРАДА ЗА ПРИБЫЛЬ (самое важное)
        base_reward_multiplier = reward_calc.get("base_reward_multiplier", 100.0)
        base_reward = pnl * base_reward_multiplier

        # 2. БОНУС ЗА БЫСТРУЮ ПРИБЫЛЬ
        strategy_params = self.model.strategies.get(strategy, self.model.strategies['balanced'])
        target_time = strategy_params.get('target_hold_time_hours', 6)

        speed_bonus_multiplier = reward_calc.get("speed_bonus_multiplier", 50.0)

        if pnl > 0:
            # БЫСТРАЯ прибыль - ВЫСОКИЙ бонус
            speed_bonus = max(0, (target_time - hold_time) / target_time) * speed_bonus_multiplier
        else:
            # Быстрый убыток - большой штраф
            speed_bonus = min(0, (target_time - hold_time) / target_time) * speed_bonus_multiplier * 2

        # 3. ШТРАФ ЗА ПРОСИЖИВАНИЕ В УБЫТКАХ
        time_penalty_multiplier = reward_calc.get("time_penalty_multiplier", 200.0)

        if pnl < 0 and hold_time > target_time * 1.5:
            time_penalty = -abs(pnl) * time_penalty_multiplier
        else:
            time_penalty = 0.0

        # 4. СТРАТЕГИЧЕСКИЙ МНОЖИТЕЛЬ
        strategy_multiplier = strategy_params.get('risk_multiplier', 1.0)

        final_reward = (base_reward + speed_bonus + time_penalty) * strategy_multiplier

        # 5. Лимиты из конфига
        max_reward = self.max_reward
        min_reward = self.min_reward

        # ✅ ОГРАНИЧЕНИЕ ДИАПАЗОНА
        limited_reward = max(min_reward, min(max_reward, final_reward))

        logger.debug(f"Reward расчет: pnl={pnl:.2f}, base={base_reward:.2f}, "
                     f"speed={speed_bonus:.2f}, penalty={time_penalty:.2f}, "
                     f"final={limited_reward:.2f}")

        return limited_reward

    def _create_initial_state(self, ticker: str, price: float, security_info: Dict) -> torch.Tensor:
        # Используем существующий метод модели
        momentum = security_info.get('momentum', 0.0)
        sentiment = self.news_core.get_current_sentiment(ticker)

        # Получаем технические данные
        indicators = self.technical_core.calculate_indicators(ticker)

        # Получаем новости
        news_items = self.rss_fetcher.get_news_for_ticker(ticker, limit=3)
        news_texts = [n.get('title', '') + ' ' + n.get('summary', '') for n in news_items]
        news_features = self.model.encode_news(news_texts)

        # ✅ ПОЛУЧАЕМ РЫНОЧНЫЙ СЕНТИМЕНТ
        market_sentiment_data = self.news_core.get_market_sentiment()
        market_sentiment = market_sentiment_data.get('sentiment', 0.0)

        # ✅ ЕДИНСТВЕННЫЙ ВЫЗОВ build_state_vector со ВСЕМИ параметрами
        state = self.model.build_state_vector(
            ticker=ticker,
            price=price,
            momentum=momentum,
            sentiment=sentiment,
            news_features=news_features,
            market_data={
                'volume': indicators.get('volume_ratio', 1.0),
                'spread': 0.01,
                'rsi': indicators.get('rsi', 50),
                'volatility': indicators.get('atr', 0) / price if price > 0 else 0.1,
                'sma_10_ratio': indicators.get('sma_10', price) / price if price > 0 else 1.0,
                'sma_20_ratio': indicators.get('sma_20', price) / price if price > 0 else 1.0,
                'bb_position': (price - indicators.get('bb_lower', price)) /
                               (indicators.get('bb_upper', price * 1.1) - indicators.get('bb_lower', price * 0.9))
                if indicators.get('bb_upper', price * 1.1) > indicators.get('bb_lower', price * 0.9) else 0.5,
                # ✅ ДОБАВЛЯЕМ недостающие поля из trader_model.py
                'liquidity': 0.5,  # или расчитать из security_info
                'market_cap': security_info.get('market_cap', 0),
                'pe_ratio': security_info.get('pe_ratio', 15)
            },
            market_sentiment=market_sentiment  # ✅ ПЕРЕДАЕМ РЫНОЧНЫЙ СЕНТИМЕНТ
        )

        return state

    def _select_buy_strategy(self, ticker: str, price: float,
                             confidence: float, state: torch.Tensor) -> Tuple[str, float, float]:
        """Выбор стратегии для покупки с учетом тональности"""

        # 1. Получаем сентимент и выбираем стратегию на его основе
        enhanced_sentiment = self.news_core.get_enhanced_sentiment(ticker)
        sentiment_score = enhanced_sentiment.get('sentiment', 0.0)
        sentiment_category = enhanced_sentiment.get('sentiment_category', 'neutral')

        # Выбираем стратегию на основе сентимента
        sentiment_strategy = self.model.choose_strategy_based_on_sentiment(
            ticker=ticker,
            sentiment=sentiment_score,
            current_strategy='balanced'
        )

        # 2. Контекст рынка для выбора стратегии моделью
        market_context = {
            'market_sentiment': self.model.market_sentiment,
            'volatility': self.model.volatility_index,
            'confidence': confidence,
            'time_of_day': datetime.now().hour / 24.0,
            'ticker_sentiment': sentiment_score,
            'sentiment_category': sentiment_category
        }

        # 3. Модель выбирает финальную стратегию
        action, final_strategy, strategy_confidence = self.model.choose_action_with_strategy(
            state=state,
            ticker=ticker,
            price=price,
            market_context=market_context
        )

        # 4. Настройка параметров с учетом тональности
        strategy_config = self.model.strategies.get(final_strategy, self.model.strategies['balanced'])

        # Корректировка на основе тональности
        sentiment_config = self.model.strategy_config.get('sentiment_integration', {})
        risk_adjustment = sentiment_config.get('risk_adjustment', {})

        # Получаем множители для текущей категории сентимента
        category_adjustment = risk_adjustment.get(sentiment_category, {})
        stop_loss_multiplier = category_adjustment.get('stop_loss_multiplier', 1.0)
        take_profit_multiplier = category_adjustment.get('take_profit_multiplier', 1.0)

        # Применяем множители
        base_stop_loss = strategy_config.get('stop_loss_percent', 2.5)
        base_take_profit = strategy_config.get('take_profit_percent', 5.0)

        adjusted_stop_loss = base_stop_loss * stop_loss_multiplier
        adjusted_take_profit = base_take_profit * take_profit_multiplier

        # Расчет конечных значений
        stop_loss = price * (1 - adjusted_stop_loss / 100)
        take_profit = price * (1 + adjusted_take_profit / 100)

        # Логирование
        logger.debug(
            f"Стратегия {ticker}: {final_strategy} "
            f"(sentiment: {sentiment_score:.3f}, category: {sentiment_category}) "
            f"SL={adjusted_stop_loss:.1f}%, TP={adjusted_take_profit:.1f}% "
            f"(multipliers: {stop_loss_multiplier:.1f}/{take_profit_multiplier:.1f})"
        )

        return final_strategy, stop_loss, take_profit

    def _periodic_learning(self):
        """ОПТИМИЗИРОВАННОЕ онлайн-обучение для БЫСТРОЙ прибыли"""
        try:
            # ✅ ИСПОЛЬЗУЕМ КОНФИГИ
            enable_extreme = self.profit_config.get("enable_extreme_learning", True)
            extreme_threshold = self.extreme_pnl_threshold

            # 1. ТОЛЬКО критические сделки
            critical_trades = []
            for ticker, pos in self.portfolio.positions.items():
                current_price = self.moex.get_price(ticker)
                if not current_price:
                    continue

                entry_price = pos.get('avg_price', 0)
                if entry_price <= 0:
                    continue

                pnl_pct = (current_price - entry_price) / entry_price * 100

                # ✅ КРИТЕРИЙ ИЗ КОНФИГА
                if abs(pnl_pct) > extreme_threshold * 100:  # конвертируем в проценты
                    critical_trades.append({
                        'ticker': ticker,
                        'pnl_pct': pnl_pct,
                        'is_profit': pnl_pct > 0
                    })

            # 2. СУПЕР-БЫСТРОЕ обучение на критических сделках
            if enable_extreme and critical_trades and len(self.model.memory) > 32:
                # Создаем ЭКСТРЕМАЛЬНЫЙ батч
                extreme_batch = self._create_extreme_batch(critical_trades)

                if extreme_batch:
                    # АГРЕССИВНОЕ обучение
                    loss = self._train_extreme(extreme_batch)

                    if loss is not None:
                        profit_count = sum(1 for t in critical_trades if t['is_profit'])
                        loss_count = len(critical_trades) - profit_count

                        logger.info(f"[ЭКСТРЕМ-обучение] {len(critical_trades)} сделок "
                                    f"(прибыль: {profit_count}, убытки: {loss_count}) Loss={loss:.6f}")

            # 3. Регулярное обучение из конфига
            learning_config = self.rl_config.get("learning", {})
            batch_size = learning_config.get("batch_size", 32)

            if self.cycle_count % self.fast_learning_cycles == 0 and len(self.model.memory) > batch_size:
                regular_loss = self.model.learn_from_experience(batch_size=batch_size)
                if regular_loss:
                    logger.debug(f"Регулярное обучение: Loss={regular_loss:.6f}")

            # 4. АДАПТАЦИЯ СТРАТЕГИЙ для максимальной прибыли
            enable_aggressive = self.profit_config.get("enable_aggressive_adaptation", True)
            if enable_aggressive and self.cycle_count % self.strategy_adaptation_cycles == 0:
                self._adapt_strategies_for_profit()

        except Exception as e:
            logger.error(f"Ошибка оптимизированного обучения: {e}")

    def _train_extreme(self, extreme_batch):
        """АГРЕССИВНОЕ обучение на экстремальных сделках с использованием конфигов"""
        try:
            if not extreme_batch:
                return None

            # ✅ КОНФИГИ ОБУЧЕНИЯ
            learning_config = self.rl_config.get("learning", {})
            aggressive_multiplier = learning_config.get("aggressive_lr_multiplier", 3.0)
            training_steps = learning_config.get("extreme_training_steps", 3)

            # ВРЕМЕННО увеличиваем learning rate для быстрой адаптации
            original_lr = self.model.policy_optimizer.param_groups[0]['lr']
            self.model.policy_optimizer.param_groups[0]['lr'] = original_lr * aggressive_multiplier

            try:
                # Безопасная подготовка батча
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

                for exp in extreme_batch:
                    if 'state' in exp and 'action' in exp and 'reward' in exp and 'next_state' in exp and 'done' in exp:
                        states.append(exp['state'])
                        actions.append(exp['action'])
                        rewards.append(exp['reward'])
                        next_states.append(exp['next_state'])
                        dones.append(exp['done'])

                if not states:
                    return None

                # Конвертируем в тензоры
                states_tensor = torch.stack(states).to(self.model.device)
                actions_tensor = torch.LongTensor(actions).to(self.model.device)
                rewards_tensor = torch.FloatTensor(rewards).to(self.model.device)
                next_states_tensor = torch.stack(next_states).to(self.model.device)
                dones_tensor = torch.FloatTensor(dones).to(self.model.device)

                # АГРЕССИВНОЕ обучение (несколько шагов)
                total_loss = 0
                successful_steps = 0

                for step in range(training_steps):
                    loss = self.model.learn_from_experience_custom(
                        states=states_tensor,
                        actions=actions_tensor,
                        rewards=rewards_tensor,
                        next_states=next_states_tensor,
                        dones=dones_tensor
                    )
                    if loss is not None:
                        total_loss += loss
                        successful_steps += 1

                avg_loss = total_loss / successful_steps if successful_steps > 0 else None

                return avg_loss

            finally:
                # Восстанавливаем оригинальный LR
                self.model.policy_optimizer.param_groups[0]['lr'] = original_lr

        except Exception as e:
            logger.error(f"Ошибка экстремального обучения: {e}")
            return None

    def _create_extreme_batch(self, critical_trades, max_size=16):
        """Создание экстремального батча для СУПЕР-БЫСТРОГО обучения"""
        try:
            extreme_experiences = []
            critical_tickers = {trade['ticker'] for trade in critical_trades}

            # ✅ ИСПОЛЬЗУЕМ КОНФИГ
            learning_config = self.rl_config.get("learning", {})
            max_size = learning_config.get("extreme_batch_size", 16)

            # Ищем последние опыты с критическими тикерами
            memory_list = list(self.model.memory)

            # Проходим с конца (самые свежие сделки)
            for exp in reversed(memory_list):
                if len(extreme_experiences) >= max_size:
                    break

                # Проверяем, связан ли опыт с критическим тикером
                exp_ticker = None
                if 'ticker' in exp:
                    exp_ticker = exp['ticker']
                elif 'market_conditions' in exp and 'ticker' in exp['market_conditions']:
                    exp_ticker = exp['market_conditions']['ticker']
                elif hasattr(exp, 'get') and callable(exp.get):
                    exp_ticker = exp.get('ticker')

                if exp_ticker in critical_tickers:
                    # Создаем модифицированную копию
                    modified_exp = exp.copy() if hasattr(exp, 'copy') else dict(exp)

                    # ✅ БЕЗОПАСНОЕ УВЕЛИЧЕНИЕ REWARD (для float и Tensor)
                    if 'reward' in modified_exp:
                        reward_value = modified_exp['reward']
                        reward_multiplier = 2.0 if 'is_profit' not in modified_exp or modified_exp.get(
                            'is_profit') else 2.5

                        # Обрабатываем и float, и torch.Tensor
                        if isinstance(reward_value, (int, float)):
                            modified_exp['reward'] = reward_value * reward_multiplier
                        elif isinstance(reward_value, torch.Tensor):
                            modified_exp['reward'] = reward_value.clone() * reward_multiplier
                        else:
                            # Если неизвестный тип, пропускаем
                            continue

                    extreme_experiences.append(modified_exp)

            logger.debug(f"Создан экстремальный батч: {len(extreme_experiences)}/{max_size} записей")
            return extreme_experiences

        except Exception as e:
            logger.error(f"Ошибка создания экстремального батча: {e}")
            return []
    def _create_priority_batch(self, critical_trades, max_size=8):
        """Создание приоритетного батча для онлайн-обучения"""
        if not self.model.memory or len(self.model.memory) < 20:
            return []

        # Берем последние сделки из памяти
        recent_memory = list(self.model.memory)[-50:]  # Последние 50 записей

        # Фильтруем по критическим тикерам
        critical_tickers = {trade['ticker'] for trade in critical_trades}
        priority_experiences = []

        for exp in recent_memory:
            # Находим тикер в опыте (может храниться в разных полях)
            exp_ticker = None

            # Ищем тикер в разных возможных полях
            if 'ticker' in exp:
                exp_ticker = exp['ticker']
            elif 'market_conditions' in exp and 'ticker' in exp['market_conditions']:
                exp_ticker = exp['market_conditions']['ticker']

            if exp_ticker in critical_tickers:
                priority_experiences.append(exp)

            if len(priority_experiences) >= max_size:
                break

        return priority_experiences

    def _train_online_priority(self, priority_batch, batch_size=8):
        """Быстрое онлайн-обучение на приоритетном батче"""
        try:
            # Подготовка данных
            states = torch.stack([exp['state'] for exp in priority_batch]).to(self.model.device)
            actions = torch.LongTensor([exp['action'] for exp in priority_batch]).to(self.model.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in priority_batch]).to(self.model.device)
            next_states = torch.stack([exp['next_state'] for exp in priority_batch]).to(self.model.device)
            dones = torch.FloatTensor([exp['done'] for exp in priority_batch]).to(self.model.device)

            # Быстрое обучение (1 эпоха)
            self.model.policy_net.train()

            # Прямой проход
            current_probs, current_values = self.model.policy_net(states)

            with torch.no_grad():
                _, next_values = self.model.policy_net(next_states)

            # Целевые значения
            target_values = rewards + (1 - dones) * self.model.gamma * next_values

            # Loss
            value_loss = torch.nn.SmoothL1Loss()(current_values, target_values.detach())

            dist = torch.distributions.Categorical(current_probs)
            log_probs = dist.log_prob(actions)
            advantages = (target_values - current_values).detach()
            policy_loss = -(log_probs * advantages).mean()

            # Общий loss
            total_loss = value_loss + policy_loss

            # Оптимизация
            self.model.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy_net.parameters(), 0.5)
            self.model.policy_optimizer.step()

            self.model.policy_net.eval()

            return total_loss.item()

        except Exception as e:
            logger.error(f"Ошибка онлайн-обучения: {e}")
            return None

    def _has_high_impact_news(self, ticker):
        """Проверка наличия высоковлиятельных новостей"""
        try:
            sentiment_data = self.news_core.get_enhanced_sentiment(ticker)
            sentiment_strength = abs(sentiment_data.get('sentiment', 0))
            impact_level = sentiment_data.get('impact_level', 'low_impact')

            return (sentiment_strength > 0.5 or impact_level == 'high_impact')
        except:
            return False

    def _evaluate_strategies(self):
        """Оценка эффективности стратегий"""
        strategies_report = {}

        for strategy_name, perf in self.model.strategy_performance.items():
            if perf['total_trades'] > 0:
                strategies_report[strategy_name] = {
                    'total_trades': perf['total_trades'],
                    'win_rate': f"{perf['win_rate'] * 100:.1f}%",
                    'avg_pnl': f"{perf['avg_pnl']:.2f}",
                    'total_pnl': f"{perf['total_pnl']:.2f}"
                }

        if strategies_report:
            logger.info(f"Эффективность стратегий: {json.dumps(strategies_report, indent=2)}")

    def _adapt_strategies(self):
        """Адаптация параметров стратегий на основе эффективности"""
        for strategy_name, perf in self.model.strategy_performance.items():
            if perf['total_trades'] >= 10:
                win_rate = perf['win_rate']
                avg_pnl = perf['avg_pnl']

                # Адаптируем risk multiplier на основе эффективности
                current_multiplier = self.model.strategies[strategy_name]['risk_multiplier']

                if win_rate > 0.6 and avg_pnl > 0:
                    # Успешная стратегия - увеличиваем риск
                    new_multiplier = min(current_multiplier * 1.1, 2.0)
                elif win_rate < 0.4 or avg_pnl < 0:
                    # Неуспешная стратегия - уменьшаем риск
                    new_multiplier = max(current_multiplier * 0.9, 0.5)
                else:
                    new_multiplier = current_multiplier

                self.model.strategies[strategy_name]['risk_multiplier'] = new_multiplier

                logger.debug(f"Адаптация {strategy_name}: "
                             f"risk_multiplier {current_multiplier:.2f} -> {new_multiplier:.2f}")

    def _initialize_components(self):
        """Инициализация всех компонентов"""
        # Устанавливаем начальные настройки для компонентов
        self.portfolio.initial_capital = self.settings.get("initial_capital_rub", 10000)
        self.portfolio.max_positions = self.settings.get("max_positions", 5)

        # Запуск непрерывного сбора новостей
        if self.settings.get("enable_news_core", True):
            self.news_core.start_continuous_fetching()

        # Загрузка состояния портфеля
        self._load_portfolio_state()

        # Запуск планировщика задач
        self.scheduler.schedule_daily_tasks(
            pre_market_callback=self.pre_session_analysis,
            market_open_callback=self.on_market_open,
            market_close_callback=self.on_market_close,
            post_market_callback=self.post_session_analysis
        )

        logger.info("SmartBroker: все компоненты инициализированы")

    def _load_portfolio_state(self):
        """Загрузка состояния портфеля из файла"""
        try:
            with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)

                # Восстанавливаем портфель
                self.portfolio.positions = state.get('positions', {})
                self.portfolio.cash = state.get('cash', self.settings["initial_capital_rub"])

                # Обновляем время покупки
                for ticker, pos in self.portfolio.positions.items():
                    if 'buy_time' not in pos:
                        pos['buy_time'] = time.time()

                logger.info(f"Загружено состояние портфеля: {len(self.portfolio.positions)} позиций, "
                            f"{self.portfolio.cash:,.0f}₽ кэша")

        except Exception as e:
            logger.warning(f"Не удалось загрузить состояние портфеля: {e}")
            # Создаем начальное состояние
            self._save_portfolio_state()

    def _save_portfolio_state(self):
        """Сохранение состояния портфеля"""
        try:
            state = {
                'total_value': self.portfolio.get_total_value({}),
                'cash': self.portfolio.cash,
                'positions': self.portfolio.positions,
                'last_update': datetime.now().isoformat(),
                'initial_capital': self.settings["initial_capital_rub"]
            }

            with open('data/portfolio_state.json', 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Ошибка сохранения состояния портфеля: {e}")

    def pre_session_analysis(self):
        """Предсессионный анализ"""
        logger.info("=" * 60)
        logger.info("ПРЕДСЕССИОННЫЙ АНАЛИЗ")
        logger.info("=" * 60)

        # Сброс дневных метрик риска
        self.risk_manager.reset_daily_metrics()

        # Анализ новостей
        self.news_core.fetch_all_news()

        # Получение списка ликвидных бумаг
        securities = self.moex.get_all_securities()
        logger.info(f"Загружено бумаг для анализа: {len(securities)}")

        # Обновление рыночного настроения
        self.analyze_sentiment(force=True)

        logger.info("Предсессионный анализ завершен")

    def on_market_open(self):
        """Действия при открытии рынка"""
        logger.info("РЫНОК ОТКРЫТ")
        self.trading_enabled = True

        # Проверка стоп-лоссов и тейк-профитов
        prices = self._get_current_prices()
        if prices:
            self.check_stops_and_tp(prices)

    def on_market_close(self):
        """Действия при закрытии рынка"""
        logger.info("РЫНОК ЗАКРЫТ")
        self.trading_enabled = False

        # Сохранение состояния
        self._save_portfolio_state()
        self.model.save_model()

        # Генерация дневного отчета
        self._generate_daily_report()

    def post_session_analysis(self):
        """Послерыночный анализ"""
        logger.info("ПОСЛЕРЫНОЧНЫЙ АНАЛИЗ")

        # Анализ результатов дня
        total_value = self.portfolio.get_total_value({})
        initial = self.settings["initial_capital_rub"]
        daily_pnl = total_value - self.portfolio.cash - sum(
            p['qty'] * p['avg_price'] for p in self.portfolio.positions.values()
        )

        logger.info(f"Итоги дня: Капитал {total_value:,.0f}₽, PnL: {daily_pnl:+,.0f}₽")
        logger.info(f"С начала: {((total_value / initial) - 1) * 100:+.2f}%")

    def run_cycle(self):
        """Основной торговый цикл"""
        if not self.trading_enabled or not self.scheduler.is_trading_time():
            return

        cycle_start = time.time()
        self.cycle_count += 1

        try:
            logger.debug(f"=== Торговый цикл #{self.cycle_count} ===")

            # 1. Получение данных
            securities = self.moex.get_all_securities()
            if not securities:
                logger.warning("Не удалось получить список бумаг")
                return

            # Берем топ-N по объему
            top_n = 120
            tickers = sorted(securities.items(),
                             key=lambda x: x[1].get('volume', 0),
                             reverse=True)[:top_n]
            tickers = [t[0] for t in tickers]
            self.current_tickers = tickers

            # 2. Получение текущих цен
            prices = {}
            for ticker in tickers:
                price = self.moex.get_price(ticker)
                if price:
                    prices[ticker] = price

            if len(prices) < 10:
                logger.warning(f"Слишком мало цен: {len(prices)}")
                return

            # 3. Генерация сигналов от всех ядер
            all_signals = []

            # Новостные сигналы
            if self.settings.get("enable_news_core", True):
                news_signals = self.news_core.generate_trading_signals(prices)
                all_signals.extend(news_signals)
                logger.debug(f"Сгенерировано новостных сигналов: {len(news_signals)}")

            # Технические сигналы
            if self.settings.get("enable_technical_core", True):
                tech_signals = self.technical_core.analyze_all_tickers(prices)
                all_signals.extend(tech_signals)
                logger.debug(f"Сгенерировано технических сигналов: {len(tech_signals)}")

            # 4. Агрегация и фильтрация сигналов
            filtered_signals = self._aggregate_signals(all_signals)
            self.signals_cache = filtered_signals[:10]  # Кэшируем топ-10

            # 5. Проверка стоп-лоссов и тейк-профитов
            self.check_stops_and_tp(prices)

            # 6. Исполнение торговых решений
            if filtered_signals and self.risk_manager.check_daily_limits():
                self._execute_trading_decisions(filtered_signals, prices, securities)

            # 7. Ребалансировка портфеля
            if self.cycle_count % 5 == 0:  # Каждые 5 циклов
                self._rebalance_portfolio(prices, securities)

            # 8. Периодическое обучение модели
            if self.cycle_count % 10 == 0:
                self._periodic_learning()

            # 9. Адаптация стратегий для прибыли
            if self.cycle_count % self.strategy_adaptation_cycles == 0:
                self._adapt_strategies_for_profit()

            # 10. Сохранение состояния
            if self.cycle_count % 20 == 0:
                self._save_portfolio_state()


            # Лог цикла
            cycle_time = time.time() - cycle_start
            total_value = self.portfolio.get_total_value(prices)

            logger.info(f"Цикл #{self.cycle_count} завершен за {cycle_time:.1f}с | "
                        f"Портфель: {total_value:,.0f}₽ | "
                        f"Позиций: {len(self.portfolio.positions)} | "
                        f"Сигналов: {len(filtered_signals)}")

        except Exception as e:
            logger.error(f"Критическая ошибка в run_cycle: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _get_recent_critical_trades(self):
        """Получение критически важных сделок для онлайн-обучения"""
        critical_trades = []

        # Критерии критической сделки:
        # 1. Крупный убыток (>5%)
        # 2. Крупная прибыль (>10%)
        # 3. Сделка на сильных новостях (sentiment > |0.5|)
        # 4. Сделка с аномальной волатильностью

        for ticker, pos in self.portfolio.positions.items():
            # Получаем текущую цену
            current_price = self.moex.get_price(ticker)
            if not current_price:
                continue

            entry_price = pos.get('avg_price', 0)
            if entry_price <= 0:
                continue

            # Расчет PnL
            pnl_pct = (current_price - entry_price) / entry_price * 100

            # Проверка на критичность
            is_critical = (
                    abs(pnl_pct) > 5 or  # Крупный PnL
                    pos.get('strategy') in ['news_aggressive', 'momentum'] or  # Важные стратегии
                    self._has_high_impact_news(ticker)  # Сильные новости
            )

            if is_critical:
                critical_trades.append({
                    'ticker': ticker,
                    'pnl_pct': pnl_pct,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'strategy': pos.get('strategy'),
                    'time_held': time.time() - pos.get('buy_time', time.time())
                })

        return critical_trades

    def _get_current_prices(self) -> Dict[str, float]:
        """Получение текущих цен для портфеля"""
        prices = {}

        for ticker in list(self.portfolio.positions.keys()):
            price = self.moex.get_price(ticker)
            if price:
                prices[ticker] = price

        return prices

    def _aggregate_signals(self, signals: List[Dict]) -> List[Dict]:
        """Агрегация и фильтрация сигналов"""
        if not signals:
            return []

        # Группируем по тикерам
        ticker_signals = defaultdict(list)
        for signal in signals:
            ticker = signal['ticker']
            ticker_signals[ticker].append(signal)

        # Агрегируем сигналы для каждого тикера
        aggregated = []

        for ticker, sig_list in ticker_signals.items():
            # Подсчитываем голоса за каждое действие
            action_scores = defaultdict(float)

            for sig in sig_list:
                action = sig['action']
                confidence = sig.get('confidence', 0.5)

                # Взвешиваем по уверенности и типу сигнала
                weight = confidence
                if sig.get('reason') == 'news_analysis':
                    weight *= 1.2  # Новостные сигналы важнее

                action_scores[action] += weight

            # Выбираем действие с максимальным score
            if action_scores:
                best_action = max(action_scores.items(), key=lambda x: x[1])

                # Только если уверенность выше порога
                if best_action[1] > self.settings.get('model_confidence_threshold', 0.65):
                    aggregated_signal = {
                        'ticker': ticker,
                        'action': best_action[0],
                        'confidence': best_action[1] / len(sig_list),  # Нормализуем
                        'source_count': len(sig_list),
                        'sources': [s.get('reason', 'unknown') for s in sig_list],
                        'timestamp': datetime.now().isoformat()
                    }
                    aggregated.append(aggregated_signal)

        # Сортируем по уверенности
        aggregated.sort(key=lambda x: x['confidence'], reverse=True)

        return aggregated

    def _record_trade_for_learning(self, ticker: str, action: str, price: float,
                                   quantity: int, confidence: float, pnl: float = 0.0):
        """Запись сделки для обучения модели"""
        try:
            # Получаем сентимент для тикера
            sentiment = 0.0
            if hasattr(self.news_core, 'get_current_sentiment'):
                sentiment = self.news_core.get_current_sentiment(ticker)

            # Записываем в модель
            if action == 'BUY':
                self.model.record_trade_outcome(
                    ticker=ticker,
                    action=action,
                    entry_price=price,
                    exit_price=price,  # При покупке exit = entry
                    hold_time=0.0,
                    news_sentiment=sentiment,
                    market_conditions={
                        'reason': 'signal_execution',
                        'confidence': confidence,
                        'quantity': quantity
                    }
                )
            elif action == 'SELL':
                pos = self.portfolio.positions.get(ticker, {})
                entry_price = pos.get('avg_price', price)
                hold_time = (time.time() - pos.get('buy_time', time.time())) / 3600  # в часах

                self.model.record_trade_outcome(
                    ticker=ticker,
                    action=action,
                    entry_price=entry_price,
                    exit_price=price,
                    hold_time=hold_time,
                    news_sentiment=sentiment,
                    market_conditions={
                        'reason': 'signal_execution',
                        'confidence': confidence,
                        'pnl': pnl,
                        'quantity': quantity
                    }
                )

        except Exception as e:
            logger.error(f"Ошибка записи сделки для обучения: {e}")

    def check_stops_and_tp(self, prices: Dict[str, float]):
        """Проверка стоп-лоссов и тейк-профитов"""
        cfg = self.settings

        for ticker, pos in list(self.portfolio.positions.items()):
            if ticker not in prices:
                continue

            price = prices[ticker]
            entry_price = pos['avg_price']

            if entry_price <= 0:
                continue

            change_pct = (price - entry_price) / entry_price * 100
            hold_time = time.time() - pos.get('buy_time', time.time())

            # Стоп-лосс
            if change_pct <= -cfg.get('stop_loss_percent', 3.0):
                qty = pos['qty']
                if self.portfolio.sell(ticker, qty, price):
                    pnl = (price - entry_price) * qty

                    self.risk_manager.update_trade_result(
                        ticker=ticker,
                        action='STOP_LOSS',
                        quantity=qty,
                        price=price,
                        pnl=pnl
                    )

                    logger.warning(f"СТОП-ЛОСС: {ticker} {qty} @ {price:.2f} "
                                   f"({change_pct:+.1f}%, PnL: {pnl:+.0f}₽)")

            # Тейк-профит
            elif change_pct >= cfg.get('take_profit_percent', 6.0):
                qty = pos['qty'] // 2  # Продаем половину
                if qty > 0:
                    if self.portfolio.sell(ticker, qty, price):
                        pnl = (price - entry_price) * qty

                        self.risk_manager.update_trade_result(
                            ticker=ticker,
                            action='TAKE_PROFIT',
                            quantity=qty,
                            price=price,
                            pnl=pnl
                        )

                        logger.info(f"ТЕЙК-ПРОФИТ: {ticker} {qty} @ {price:.2f} "
                                    f"({change_pct:+.1f}%, PnL: {pnl:+.0f}₽)")

    def _rebalance_portfolio(self, prices: Dict, securities: Dict):
        """Ребалансировка портфеля"""
        current_count = len(self.portfolio.positions)
        target_count = self.settings.get('target_positions', 4)

        # Если позиций меньше целевого - докупаем
        if current_count < target_count and self.portfolio.cash > self.settings["min_cash_per_trade"]:
            # Используем ранжирование от модели
            ticker_sentiment = {t: 0.0 for t in prices.keys()}
            news_by_ticker = {t: [] for t in prices.keys()}

            candidates = self.model.rank_candidates(
                prices=prices,
                securities=securities,
                ticker_sentiment=ticker_sentiment,
                news_by_ticker=news_by_ticker
            )

            bought = 0
            for ticker, score in candidates:
                if bought >= 2:  # Макс 2 позиции за ребалансировку
                    break

                if ticker in self.portfolio.positions:
                    continue

                price = prices[ticker]
                max_qty = int(self.portfolio.cash * 0.15 / price)  # Макс 15% кэша

                if max_qty > 0:
                    qty = max(1, max_qty // 2)  # Берем половину от максимального

                    if self.portfolio.buy(ticker, qty, price):
                        self.portfolio.positions[ticker]['buy_time'] = time.time()
                        bought += 1

                        logger.info(f"Ребалансировка BUY: {ticker} {qty} @ {price:.2f} "
                                    f"(score: {score:.2f})")



    def _generate_daily_report(self):
        """Генерация дневного отчета"""
        try:
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'portfolio_value': self.portfolio.get_total_value({}),
                'cash': self.portfolio.cash,
                'positions_count': len(self.portfolio.positions),
                'daily_pnl': self.risk_manager.daily_pnl,
                'daily_trades': self.risk_manager.daily_trades,
                'market_sentiment': self.model.market_sentiment,
                'signals_generated': len(self.signals_cache),
                'risk_metrics': self.risk_manager.get_risk_metrics()
            }

            # Сохраняем отчет
            report_file = f"data/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Дневной отчет сохранен: {report_file}")

        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")

    def analyze_sentiment(self, force: bool = False):
        """Анализ рыночного настроения"""
        try:
            # Используем RSS фетчер для новостей
            news_items = self.rss_fetcher.fetch_all_news()

            if not news_items:
                return

            # Анализ тональности
            sentiments = self.news_core.analyze_news_sentiment(news_items)
            market_sentiment = sentiments.get('MARKET', 0.0)

            # Обновляем в модели
            self.model.update_market_sentiment(market_sentiment)

            logger.info(f"Рыночное настроение обновлено: {market_sentiment:+.3f}")

        except Exception as e:
            logger.error(f"Ошибка анализа сентимента: {e}")

    def _create_next_state(self, ticker: str, exit_price: float) -> torch.Tensor:
        """Создание следующего состояния для RL с ГАРАНТИЕЙ ПРАВИЛЬНОГО DEVICE"""
        try:
            # Получаем текущую цену
            current_price = self.moex.get_price(ticker)
            if not current_price:
                current_price = exit_price

            # Получаем информацию о бумаге
            securities = self.moex.get_all_securities()
            security_info = securities.get(ticker, {})

            # Получаем сентимент
            sentiment = 0.0
            if hasattr(self.news_core, 'get_current_sentiment'):
                try:
                    sentiment = self.news_core.get_current_sentiment(ticker)
                except:
                    pass

            # Получаем технические индикаторы
            indicators = {}
            try:
                indicators = self.technical_core.calculate_indicators(ticker)
            except:
                pass

            # Получаем новости
            news_items = []
            try:
                news_items = self.rss_fetcher.get_news_for_ticker(ticker, limit=3)
            except:
                pass

            news_texts = [n.get('title', '') + ' ' + n.get('summary', '') for n in news_items]
            news_features = self.model.encode_news(news_texts)

            # Используем существующий метод модели для создания состояния
            state = self.model.build_state_vector(
                ticker=ticker,
                price=current_price,
                momentum=security_info.get('momentum', 0.0),
                sentiment=sentiment,
                news_features=news_features,
                market_data={
                    'volume': indicators.get('volume_ratio', 1.0) if indicators else 1.0,
                    'spread': 0.01,
                    'rsi': indicators.get('rsi', 50) if indicators else 50,
                    'volatility': indicators.get('atr', 0) / current_price if indicators and current_price > 0 else 0.1,
                    'sma_10_ratio': indicators.get('sma_10',
                                                   current_price) / current_price if indicators and current_price > 0 else 1.0,
                    'sma_20_ratio': indicators.get('sma_20',
                                                   current_price) / current_price if indicators and current_price > 0 else 1.0,
                    'bb_position': (
                        (current_price - indicators.get('bb_lower', current_price * 0.9)) /
                        (indicators.get('bb_upper', current_price * 1.1) - indicators.get('bb_lower',
                                                                                          current_price * 0.9))
                        if indicators and indicators.get('bb_upper', current_price * 1.1) > indicators.get('bb_lower',
                                                                                                           current_price * 0.9)
                        else 0.5
                    ) if indicators else 0.5
                }
            )

            # ✅ ГАРАНТИРУЕМ ПРАВИЛЬНЫЙ DEVICE
            return state.to(self.model.device)

        except Exception as e:
            logger.error(f"Ошибка создания следующего состояния для {ticker}: {e}")
            # ✅ ВОЗВРАЩАЕМ СОСТОЯНИЕ ПО УМОЛЧАНИЮ НА ПРАВИЛЬНОМ DEVICE
            state_dim = getattr(self.model.policy_net, 'expected_state_dim',
                                getattr(self.model.policy_net, 'state_dim', 156))
            return torch.zeros(state_dim, dtype=torch.float32).to(self.model.device)

    def _adapt_strategies_for_profit(self):
        """АГРЕССИВНАЯ адаптация стратегий для максимальной прибыли с использованием конфигов"""
        try:
            adaptation_config = self.rl_config.get("strategy_adaptation", {})
            min_trades = adaptation_config.get("min_trades_for_fast_adaptation", 5)

            for strategy_name, perf in self.model.strategy_performance.items():
                if perf['total_trades'] >= min_trades:
                    win_rate = perf['win_rate']
                    avg_pnl = perf['avg_pnl']
                    current_multiplier = self.model.strategies[strategy_name]['risk_multiplier']

                    # ✅ ПОРОГИ ИЗ КОНФИГА
                    win_high = adaptation_config.get("aggressive_win_rate_high", 0.7)
                    win_low = adaptation_config.get("aggressive_win_rate_low", 0.3)
                    profit_thresh = adaptation_config.get("profit_threshold_fast", 0.02)
                    loss_thresh = adaptation_config.get("loss_threshold_fast", -0.01)
                    max_risk = adaptation_config.get("max_risk_multiplier", 2.0)
                    min_risk = adaptation_config.get("min_risk_multiplier", 0.5)

                    # ✅ ЛОГИКА АДАПТАЦИИ
                    if win_rate > win_high and avg_pnl > profit_thresh:
                        # ОЧЕНЬ успешная стратегия - СИЛЬНО увеличиваем риск
                        increase_factor = adaptation_config.get("aggressive_increase_factor", 1.3)
                        new_multiplier = min(current_multiplier * increase_factor, max_risk)

                        logger.info(f"🔥 АГРЕССИВНОЕ УВЕЛИЧЕНИЕ {strategy_name}: "
                                    f"{current_multiplier:.2f} → {new_multiplier:.2f} "
                                    f"(WR: {win_rate:.1%}, PnL: {avg_pnl:.2%})")

                    elif win_rate < win_low or avg_pnl < loss_thresh:
                        # ПЛОХАЯ стратегия - СИЛЬНО уменьшаем риск
                        decrease_factor = adaptation_config.get("aggressive_decrease_factor", 0.7)
                        new_multiplier = max(current_multiplier * decrease_factor, min_risk)

                        logger.warning(f"⚠ АГРЕССИВНОЕ УМЕНЬШЕНИЕ {strategy_name}: "
                                       f"{current_multiplier:.2f} → {new_multiplier:.2f} "
                                       f"(WR: {win_rate:.1%}, PnL: {avg_pnl:.2%})")
                    else:
                        new_multiplier = current_multiplier
                        logger.debug(f"Стратегия {strategy_name} без изменений: {current_multiplier:.2f}")

                    self.model.strategies[strategy_name]['risk_multiplier'] = new_multiplier

            # ✅ СОХРАНЯЕМ ОБНОВЛЕННЫЕ СТРАТЕГИИ
            self.model.save_model()
            logger.info("Адаптация стратегий завершена и сохранена")

        except Exception as e:
            logger.error(f"Ошибка адаптации стратегий: {e}")

    def get_portfolio_summary(self) -> Dict:
        """Получение сводки портфеля"""
        prices = self._get_current_prices()
        total_value = self.portfolio.get_total_value(prices)

        summary = {
            'total_value': total_value,
            'cash': self.portfolio.cash,
            'positions_count': len(self.portfolio.positions),
            'initial_capital': self.settings["initial_capital_rub"],
            'total_pnl': total_value - self.settings["initial_capital_rub"],
            'pnl_percent': ((total_value / self.settings["initial_capital_rub"]) - 1) * 100,
            'current_signals': self.signals_cache[:5],
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'session_info': self.scheduler.get_session_info(),
            'last_update': datetime.now().isoformat()
        }

        # Детали по позициям
        positions_detail = []
        for ticker, pos in self.portfolio.positions.items():
            current_price = prices.get(ticker, pos['avg_price'])
            position_value = pos['qty'] * current_price

            positions_detail.append({
                'ticker': ticker,
                'quantity': pos['qty'],
                'avg_price': pos['avg_price'],
                'current_price': current_price,
                'position_value': position_value,
                'pnl': (current_price - pos['avg_price']) * pos['qty'],
                'pnl_percent': ((current_price / pos['avg_price']) - 1) * 100,
                'weight': (position_value / total_value * 100) if total_value > 0 else 0
            })

        summary['positions'] = positions_detail

        return summary

    def update_settings(self, new_settings: Dict):
        """Обновление настроек системы"""
        try:
            # Обновляем основные настройки
            self.settings.update(new_settings)

            # Специфичные обновления компонентов
            updates_to_apply = []

            if 'initial_capital_rub' in new_settings:
                self.portfolio.initial_capital = new_settings['initial_capital_rub']
                updates_to_apply.append(f"initial_capital_rub={new_settings['initial_capital_rub']}")

            if 'max_positions' in new_settings:
                self.portfolio.max_positions = new_settings['max_positions']
                updates_to_apply.append(f"max_positions={new_settings['max_positions']}")

            # Обновление Risk Manager
            if 'risk_per_trade_percent' in new_settings:
                if hasattr(self.risk_manager, 'config'):
                    self.risk_manager.config['risk_per_trade_percent'] = new_settings['risk_per_trade_percent']
                    updates_to_apply.append(f"risk_per_trade_percent={new_settings['risk_per_trade_percent']}%")

            # Обновление параметров стоп-лоссов
            stop_loss_params = ['stop_loss_percent', 'take_profit_percent', 'max_position_weight_percent']
            for param in stop_loss_params:
                if param in new_settings:
                    if hasattr(self.risk_manager, 'config') and param in self.risk_manager.config:
                        self.risk_manager.config[param] = new_settings[param]
                        updates_to_apply.append(f"{param}={new_settings[param]}")

            if updates_to_apply:
                logger.info(f"Настройки обновлены: {', '.join(updates_to_apply)}")
            else:
                logger.info(f"Настройки обновлены: {new_settings}")

        except Exception as e:
            logger.error(f"Ошибка обновления настроек брокера: {e}")

    def shutdown(self):
        """Корректное завершение работы"""
        logger.info("Завершение работы SmartBroker...")

        # Сохраняем память даже если мало опытов
        if len(self.model.memory) > 0:
            self.model.save_memory()

        # Останавливаем сбор новостей
        self.news_core.stop_continuous_fetching()

        # Сохраняем состояние
        self._save_portfolio_state()
        self.model.save_model()

        logger.info("SmartBroker завершил работу")