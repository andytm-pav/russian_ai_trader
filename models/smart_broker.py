"""
Главный модуль Smart Broker с интеграцией всех компонентов
"""

import json
import torch
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from core.core_technical_trader import TechnicalTraderCore
from core.risk_manager import RiskManager
from core.trading_hours_scheduler import TradingScheduler
from fetchers.moex_fetcher import MoexFetcher
from fetchers.news_fetcher import OptimizedNewsFetcher  # ✅ ИСПРАВЛЕНО
from utils.portfolio_manager import PortfolioManager
from utils.logger import get_logger
from models.trader_model import NEWS_ENCODED_DIM

from models.trader_model import trader_model_instance
from utils.lot_validator import LotValidator

logger = None
# NEWS_ENCODED_DIM = 128

class SmartPortfolioBroker:
    """Умный брокер с интеграцией всех торговых ядер"""

    def __init__(self, settings: Dict):
        self.rl_config = None
        self.settings = settings
        self.moex = MoexFetcher()

        # ✅ ИСПРАВЛЕНО: используем OptimizedNewsFetcher вместо RSSFetcher и NewsTraderCore
        self.news_fetcher = OptimizedNewsFetcher("config/rss_sources.json")

        self.technical_core = TechnicalTraderCore()
        self.risk_manager = RiskManager()
        self.scheduler = TradingScheduler()
        self.auction_mode = False

        self.auction_mode = False
        self.portfolio = PortfolioManager()
        self.model = trader_model_instance
        self.model.market_period = "closed"

        # ✅ ЗАГРУЗКА НАСТРОЕК Т-БАНКА
        moex_config = settings.get('moex_schedule', {})
        commission_config = moex_config.get('commission', {})
        self.tbank_config = commission_config.get('tbank', {})

        # Параметры с значениями по умолчанию
        self.tbank_check_start = self.tbank_config.get('check_start_hour', 9)
        self.tbank_check_end = self.tbank_config.get('check_end_hour', 14)
        self.tbank_check_interval = self.tbank_config.get('check_interval_cycles', 60)
        self.tbank_settlement_time = self.tbank_config.get('settlement_time', "14:00")

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

        # ✅ Добавляем статистику новостного фетчера
        print(f"[SmartBroker] NewsFetcher: {self.news_fetcher.stats}")

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

    def _get_ticker_sentiment(self, ticker: str) -> float:
        """Получение сентимента для тикера из оптимизированного фетчера"""
        try:
            # Получаем новости для тикера
            news_items = self.news_fetcher.search_news(ticker=ticker, limit=5)

            if not news_items:
                return 0.0

            # Анализируем сентимент
            news_with_sentiment = self.news_fetcher.analyze_sentiment_batch(news_items)

            # Усредняем сентимент
            sentiments = [n.get('sentiment', 0.0) for n in news_with_sentiment]
            return sum(sentiments) / len(sentiments) if sentiments else 0.0

        except Exception as e:
            logger.error(f"Ошибка получения сентимента для {ticker}: {e}")
            return 0.0

    def _get_market_sentiment(self) -> float:
        """Получение рыночного сентимента из оптимизированного фетчера"""
        try:
            # Получаем все свежие новости
            news_items = self.news_fetcher.get_last_news(limit=50)

            if not news_items:
                return 0.0

            # Анализируем сентимент
            news_with_sentiment = self.news_fetcher.analyze_sentiment_batch(news_items)

            # Усредняем сентимент
            sentiments = [n.get('sentiment', 0.0) for n in news_with_sentiment]
            return sum(sentiments) / len(sentiments) if sentiments else 0.0

        except Exception as e:
            logger.error(f"Ошибка получения рыночного сентимента: {e}")
            return 0.0

    def _generate_news_signals(self, prices: Dict[str, float]) -> List[Dict]:
        """Генерация сигналов на основе новостей"""
        signals = []

        try:
            # Получаем новости с сентиментом
            news_items = self.news_fetcher.get_last_news(limit=100)

            if not news_items:
                return signals

            # Анализируем сентимент для всех новостей
            news_with_sentiment = self.news_fetcher.analyze_sentiment_batch(news_items)

            # Группируем по тикерам
            ticker_sentiments = defaultdict(list)

            for news in news_with_sentiment:
                sentiment = news.get('sentiment', 0.0)

                # Проверяем упоминания тикеров
                title = news.get('title', '')
                for ticker in prices.keys():
                    if ticker in title.upper():
                        ticker_sentiments[ticker].append(sentiment)
                        break

            # Генерируем сигналы
            for ticker, price in prices.items():
                sentiments = ticker_sentiments.get(ticker, [])

                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)

                    # Порог срабатывания
                    if abs(avg_sentiment) > 0.3:
                        signals.append({
                            'ticker': ticker,
                            'action': 'BUY' if avg_sentiment > 0 else 'SELL',
                            'confidence': abs(avg_sentiment),
                            'price': price,
                            'reason': 'news_analysis',
                            'sentiment': avg_sentiment,
                            'news_count': len(sentiments)
                        })

            logger.debug(f"Сгенерировано новостных сигналов: {len(signals)}")

        except Exception as e:
            logger.error(f"Ошибка генерации новостных сигналов: {e}")

        return signals

    def _execute_trading_decisions(self, signals: List[Dict],
                                   prices: Dict[str, float], securities: Dict):
        """Исполнение торговых решений с RL и учетом лотности"""
        logger.debug(f"[DEBUG] Исполнение {len(signals)} сигналов")

        executed_count = 0

        for signal in signals[:5]:  # Обрабатываем только топ-5 сигналов
            ticker = signal['ticker']
            action_str = signal['action']  # 'BUY', 'SELL'
            confidence = signal['confidence']

            if ticker not in prices or ticker not in securities:
                continue

            # ✅ Получение параметров лотности
            security_info = securities[ticker]
            lot_size = security_info.get('lot_size', 1)
            min_step = security_info.get('min_step', 0.01)

            price = prices[ticker]

            # ✅ КОРРЕКТИРОВКА ЦЕНЫ
            if min_step > 0:
                price, price_adjusted = LotValidator.validate_and_adjust_price(price, min_step)
                if price_adjusted:
                    logger.debug(f"Цена {ticker} скорректирована по min_step {min_step}")

            action_idx = {'BUY': 0, 'HOLD': 1, 'SELL': 2}[action_str]

            # Логируем каждый сигнал
            logger.debug(f"[DEBUG] Сигнал: {ticker} {action_str} conf={confidence:.2f}")

            # 1. Получаем или создаем состояние для тикера
            if ticker in self.ticker_states:
                current_state = self.ticker_states[ticker]
            else:
                # Создаем начальное состояние
                current_state = self._create_initial_state(ticker, price, security_info)
                self.ticker_states[ticker] = current_state

            if action_str == 'BUY':
                # 2. Выбор стратегии для покупки
                strategy, stop_loss, take_profit = self._select_buy_strategy(
                    ticker, price, confidence, current_state
                )

                # 3. Получение множителя стратегии
                strategy_multiplier = 1.0
                if hasattr(self.model, 'strategies') and strategy in self.model.strategies:
                    strategy_multiplier = self.model.strategies[strategy].get('risk_multiplier', 1.0)
                else:
                    logger.warning(f"Стратегия {strategy} не найдена в модели")

                # 4. Расчет размера позиции с учетом лотности
                quantity, risk_amount = self.risk_manager.calculate_position_size(
                    ticker=ticker,
                    price=price,
                    stop_loss=stop_loss,
                    confidence=confidence,
                    lot_size=lot_size
                )

                # ✅ Применяем стратегический множитель
                if strategy_multiplier != 1.0:
                    quantity = max(lot_size, int(round(quantity * strategy_multiplier)))
                    logger.debug(f"Стратегический множитель {strategy}: {strategy_multiplier:.2f}x → {quantity} акций")

                logger.debug(f"[DEBUG] {ticker}: quantity={quantity}, risk_amount={risk_amount}, lot_size={lot_size}")

                # ✅ ПРОВЕРКА ЛОТНОСТИ
                if lot_size > 1:
                    original_quantity = quantity
                    quantity, qty_adjusted = LotValidator.validate_and_adjust_quantity(quantity, lot_size)

                    if qty_adjusted:
                        logger.debug(
                            f"Количество {ticker} скорректировано с {original_quantity} до {quantity} (лот: {lot_size})")

                    # Если после корректировки 0, пробуем взять 1 лот
                    if quantity == 0:
                        # Проверяем, можем ли взять 1 лот
                        one_lot_cost = lot_size * price
                        if one_lot_cost <= self.portfolio.cash * 0.5:  # Не больше 50% кэша
                            quantity = lot_size
                            logger.debug(f"✋ Берем минимальный лот: {quantity} (стоимость: {one_lot_cost:.2f}₽)")
                        else:
                            logger.warning(
                                f"❌ Недостаточно средств для 1 лота {ticker}: нужно {one_lot_cost:.2f}₽, есть {self.portfolio.cash:.2f}₽")
                            continue

                if quantity > 0:
                    # ✅ Получаем сентимент из оптимизированного фетчера
                    ticker_sentiment = self._get_ticker_sentiment(ticker)
                    sentiment_data = {
                        'sentiment': ticker_sentiment,
                        'news_count': signal.get('news_count', 0),
                        'impact_level': 'high_impact' if abs(ticker_sentiment) > 0.5 else 'medium_impact'
                    }

                    logger.debug(f"[DEBUG] {ticker}: cash={self.portfolio.cash}, price={price}, total={quantity * price}")

                    # ✅ ОДИН вызов buy с передачей всех параметров
                    if self.portfolio.buy(ticker, quantity, price, strategy,
                                          lot_size=lot_size,
                                          min_step=min_step,
                                          stop_loss=stop_loss,
                                          take_profit=take_profit):

                        # 5. Запись RL-опыта с сентиментом
                        self._record_rl_experience(
                            ticker=ticker,
                            state=current_state,
                            action=action_idx,
                            strategy=strategy,
                            price=price,
                            quantity=quantity,
                            sentiment_data=sentiment_data
                        )

                        # 6. Обновляем позицию
                        self.portfolio.positions[ticker]['strategy'] = strategy
                        self.portfolio.positions[ticker]['stop_loss'] = stop_loss
                        self.portfolio.positions[ticker]['take_profit'] = take_profit
                        self.portfolio.positions[ticker]['entry_state'] = current_state.cpu().numpy().tolist()

                        executed_count += 1
                        logger.debug(f"[DEBUG] Куплено: {ticker} {quantity} @ {price:.2f}")




            elif action_str == 'SELL':

                if ticker in self.portfolio.positions:
                    logger.debug(f"[DEBUG] Продажа {ticker}, есть позиция")

                    # Получаем параметры из позиции
                    pos = self.portfolio.positions[ticker]
                    pos_lot_size = pos.get('lot_size', 1)
                    pos_min_step = pos.get('min_step', 0.01)

                    # ✅ СОХРАНЯЕМ ДАННЫЕ ДО ПРОДАЖИ
                    pos_info = {
                        'avg_price': pos['avg_price'],
                        'strategy': pos.get('strategy', 'balanced'),
                        'lot_size': pos.get('lot_size', 1),
                        'qty': pos['qty'],
                        'buy_time': pos.get('buy_time', time.time())
                    }

                    # Корректировка цены продажи
                    sell_price = price

                    if pos_min_step > 0:
                        sell_price, price_adjusted = LotValidator.validate_and_adjust_price(price, pos_min_step)

                        if price_adjusted:
                            logger.debug(f"Цена продажи {ticker} скорректирована")

                    # Исполнение продажи
                    qty = pos['qty'] // 2 if pos['qty'] > 1 else pos['qty']

                    # Корректировка по лотности
                    if pos_lot_size > 1:
                        qty, qty_adjusted = LotValidator.validate_and_adjust_quantity(qty, pos_lot_size)

                        if qty_adjusted:
                            logger.debug(f"Количество продажи {ticker} скорректировано по лотности")

                        if qty == 0:
                            qty = pos_lot_size

                    if qty > 0:
                        success, pnl_with_commission = self.portfolio.sell(ticker, qty, sell_price)

                        if success:
                            strategy = pos.get('strategy', 'balanced')
                            executed_count += 1
                            logger.debug(
                                f"[DEBUG] Продано: {ticker} {qty} @ {sell_price:.2f} (PnL с комиссией: {pnl_with_commission:.2f})")

                            # ✅ ПЕРЕДАЁМ pos_info В _complete_rl_experience!

                            self._complete_rl_experience(
                                ticker,
                                sell_price,
                                actual_pnl=pnl_with_commission,
                                pos_info=pos_info  #
                            )

                            # Записываем результат стратегии
                            if hasattr(self.model, 'record_strategy_outcome'):
                                self.model.record_strategy_outcome(
                                    strategy_name=strategy,
                                    action='SELL',
                                    pnl=pnl_with_commission,
                                    hold_time=time.time() - pos.get('buy_time', time.time())
                                )

        logger.debug(f"[DEBUG] Всего исполнено сделок: {executed_count}")
        return executed_count

    def _record_rl_experience(self, ticker: str, state: torch.Tensor,
                              action: int, strategy: str, price: float,
                              quantity: int, sentiment_data: Dict = None):

        print(f"\n📝 _record_rl_experience for {ticker}")
        print(f"   state.shape: {state.shape}, dtype: {state.dtype}")
        print(f"   action: {action}, strategy: {strategy}")

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
            'is_priority': self._is_priority_experience(sentiment_data)
        }
        self.pending_experiences.append(experience)
        print(f"   ✅ Добавлено в pending_experiences: {len(self.pending_experiences)}")

        if hasattr(self, 'trainer') and experience['is_priority']:
            self.trainer.add_priority_experience(experience)

    def _is_priority_experience(self, sentiment_data: Optional[Dict]) -> bool:
        """Определение приоритетности опыта через конфиги"""
        if not sentiment_data:
            return False

        strategy_config = self.model.strategy_config
        priority_config = strategy_config.get("strategy_selection", {}).get("priority_learning", {})

        if not priority_config.get("enable_priority_training", True):
            return False

        sentiment_threshold = priority_config.get("sentiment_threshold", 0.6)
        news_count_threshold = priority_config.get("news_count_threshold", 3)

        sentiment_strength = abs(sentiment_data.get('sentiment', 0))
        impact_level = sentiment_data.get('impact_level', 'low_impact')
        news_count = sentiment_data.get('news_count', 0)

        is_priority = (
                sentiment_strength > sentiment_threshold or
                impact_level in priority_config.get("impact_levels", ["high_impact"]) or
                news_count >= news_count_threshold
        )

        return is_priority

    def _complete_rl_experience(self, ticker: str, exit_price: float, actual_pnl: float = None, pos_info: dict = None):
        logger.debug(f"[DEBUG] Завершение RL опыта для {ticker} @ {exit_price}")

        print(f"\n🔍🔍🔍 _complete_rl_experience for {ticker}")
        print(f"   pending_experiences before: {len(self.pending_experiences)}")
        for i, exp in enumerate(self.pending_experiences):
            print(f"   exp {i}: ticker={exp['ticker']}, completed={exp['completed']}")

        found = False
        for exp in self.pending_experiences:
            if exp['ticker'] == ticker and not exp['completed']:
                found = True
                print(f"   ✅ НАЙДЕН ОПЫТ!")
                print(f"   exp['start_state'].shape: {exp['start_state'].shape}")
                print(f"   exp['strategy']: {exp['strategy']}")

        # -----------------------------------------------------------------------------

        for exp in self.pending_experiences:
            if exp['ticker'] == ticker and not exp['completed']:
                # Если передан actual_pnl - используем его (с комиссией)
                if actual_pnl is not None:
                    pnl = actual_pnl
                else:
                    # Старый расчет (без комиссии) - только для обратной совместимости
                    pnl = (exit_price - exp['entry_price']) * exp['quantity']
                hold_time = (time.time() - exp['entry_time']) / 3600

                # 2. Создаем следующее базовое состояние (150)
                next_base_state = self._create_next_state(ticker, exit_price)

                # 3. Получаем параметры стратегии из опыта
                strategy_params = self.model.strategies[exp['strategy']]

                # 4. 🔥 ПРЕОБРАЗУЕМ ОБА СОСТОЯНИЯ В 156
                # Начальное состояние (было сохранено как 150)
                start_base_state = exp['start_state'].to(self.model.device)
                full_start_state = self.model._create_strategy_state(
                    start_base_state,
                    strategy_params
                )

                # Следующее состояние (тоже преобразуем в 156)
                full_next_state = self.model._create_strategy_state(
                    next_base_state,
                    strategy_params
                )

                # 5. Расчет награды
                reward = self._calculate_reward(pnl, hold_time, exp['strategy'])

                # 6. 🔥 РАССЧИТЫВАЕМ TD-ERROR ДЛЯ ПРИОРИТЕТА (используем полные состояния)
                state_value = self.model.get_state_value(full_start_state)
                next_state_value = self.model.get_state_value(full_next_state)
                td_error = reward + self.model.gamma * next_state_value - state_value

                # 7. 🔥 ОПРЕДЕЛЯЕМ КРИТИЧЕСКИЕ ОШИБКИ
                is_critical = False
                critical_reason = ""

                entry_price = exp['entry_price']
                qty = exp['quantity']

                if pnl < -500:  # Крупный убыток в рублях
                    is_critical = True
                    critical_reason = "large_loss_rub"
                elif pnl / (entry_price * qty) < -0.05:  # Убыток >5%
                    is_critical = True
                    critical_reason = "large_loss_percent"
                elif hold_time < 0.5 and pnl < 0:  # Быстрая убыточная сделка
                    is_critical = True
                    critical_reason = "quick_loss"
                elif hold_time > 24 and pnl < 0:  # Долгая убыточная позиция
                    is_critical = True
                    critical_reason = "stuck_position"

                # 8. Сохраняем опыт с TD-error (оба состояния 156)

                print(f"   🔥 ВЫЗЫВАЕМ remember_experience!")

                # Рассчитываем PnL в процентах перед вызовом
                entry_price = exp['entry_price']
                if entry_price > 0:
                    pnl_percent = (exit_price - entry_price) / entry_price  # ✅ проценты
                else:
                    pnl_percent = 0.0
                pnl_rub = pnl

                self.model.remember_experience(
                    state=full_start_state,  # ✅ 156
                    action=exp['action'],
                    reward=reward,
                    next_state=full_next_state,  # ✅ 156
                    done=True,
                    news_features=None,
                    td_error=td_error,
                    sentiment_data=exp.get('sentiment_data'),
                    pnl_rub = pnl_rub,  # ✅  рубли для лога
                    pnl_percent = pnl_percent
                )
                print(f"   ✅ remember_experience ВЫЗВАН")

                # 9. 🔥 ДЛЯ КРИТИЧЕСКИХ ОШИБОК - ДУБЛИРУЕМ С ВЫСОКИМ ПРИОРИТЕТОМ
                if is_critical:
                    # Искусственно увеличиваем TD-error для приоритета
                    self.model.remember_experience(
                        state=full_start_state,  # ✅ 156
                        action=exp['action'],
                        reward=reward * 2,  # Усиливаем награду для обучения
                        next_state=full_next_state,  # ✅ 156
                        done=True,
                        news_features=None,
                        td_error=td_error * 3, # 🔥 УСИЛЕННЫЙ ПРИОРИТЕТ
                        sentiment_data=exp.get('sentiment_data'),
                        pnl_rub=pnl_rub,  # рубли для лога
                        pnl_percent=pnl_percent # проценты для лога
                    )
                    logger.warning(f"🔥 КРИТИЧЕСКАЯ ОШИБКА: {ticker} - {critical_reason} (PnL: {pnl:.2f})")

                # 10. Запись в модель для статистики
                if 'sentiment_data' in exp:
                    market_sentiment = self._get_market_sentiment()


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
                            'pnl': pnl_percent,
                            'pnl_rub': pnl,
                            'market_sentiment': market_sentiment,
                            'is_critical': is_critical,
                            'critical_reason': critical_reason
                        },
                        strategy=exp['strategy'],
                        market_sentiment=market_sentiment
                    )

                exp['completed'] = True
                print(f"   ✅ Опыт завершен")
                logger.debug(f"RL опыт завершен: {ticker}, reward={reward:.3f}, pnl={pnl:.2f}")

                # 💾 Принудительное сохранение каждые 3 опыта
                if len(self.model.memory) % 3 == 0:
                    print(f"💾 Сохраняем память после {len(self.model.memory)} опытов")
                    self.model.save_memory()

                break



        if not found:
            print(f"   ❌ ОПЫТ ДЛЯ {ticker} НЕ НАЙДЕН!")
            # ✅ ВАЖНО: создаем опыт "на лету" для позиций из файла
            if ticker in self.portfolio.positions:
                print(f"   ⚠️ Создаем опыт на лету для {ticker} (из портфеля)")
                pos = self.portfolio.positions[ticker]

                # Создаем базовое состояние (150)
                dummy_state = self._create_initial_state(
                    ticker,
                    pos['avg_price'],
                    {'lot_size': pos.get('lot_size', 1)}
                )
                print(f"   dummy_state.shape: {dummy_state.shape}")

                exp = {
                    'ticker': ticker,
                    'start_state': dummy_state.cpu(),
                    'action': 2,  # SELL
                    'strategy': pos.get('strategy', 'balanced'),
                    'entry_price': pos['avg_price'],
                    'entry_time': pos.get('buy_time', time.time() - 3600),
                    'quantity': pos['qty'],
                    'sentiment_data': {'sentiment': 0, 'news_count': 0},
                    'completed': False
                }
                self.pending_experiences.append(exp)
                print(f"   ✅ Создан фиктивный опыт, теперь {len(self.pending_experiences)} опытов")
                # Повторяем вызов
                self._complete_rl_experience(ticker, exit_price)
                return

        print(f"   pending_experiences after: {len(self.pending_experiences)}")
        self.pending_experiences = [exp for exp in self.pending_experiences if not exp['completed']]

    def _calculate_reward(self, pnl: float, hold_time: float, strategy: str) -> float:

        global logger

        """
        ЕДИНАЯ функция награды, объединяющая оба подхода
        """
        global logger
        if logger is None:
            from utils.logger import get_logger
            logger = get_logger('SMART_BROKER')

        # 1. Получаем параметры из конфига
        reward_calc = self.rl_config.get("reward_calculation", {})
        base_reward_multiplier = reward_calc.get("base_reward_multiplier", 100.0)
        speed_bonus_multiplier = reward_calc.get("speed_bonus_multiplier", 50.0)
        time_penalty_multiplier = reward_calc.get("time_penalty_multiplier", 200.0)

        # 2. Параметры стратегии
        strategy_params = self.model.strategies.get(strategy, self.model.strategies['balanced'])
        target_time = strategy_params.get('target_hold_time_hours', 6)

        # 3. БАЗОВАЯ НАГРАДА ЗА ПРИБЫЛЬ
        base_reward = pnl * base_reward_multiplier

        # 4. БОНУС/ШТРАФ ЗА СКОРОСТЬ (объединенная логика)
        if pnl > 0:
            # Прибыль: чем быстрее, тем лучше (НО не скальпинг)
            if hold_time < 0.25:  # меньше 15 минут - это скальпинг
                speed_component = -abs(pnl) * 2  # штраф за скальпинг
                logger.debug(f"⚠️ Штраф за скальпинг: {speed_component:.2f}")
            else:
                # Нормальная быстрая прибыль - бонус
                speed_ratio = max(0, (target_time - hold_time) / target_time)
                speed_component = speed_ratio * speed_bonus_multiplier
                logger.debug(f"🏆 Бонус за быструю прибыль: {speed_component:.2f}")
        else:
            # Убыток: чем быстрее, тем лучше (меньше убыток)
            if hold_time < 0.25:
                # Быстрый убыток - меньший штраф
                speed_component = abs(pnl) * speed_bonus_multiplier * 0.5
                logger.debug(f"✅ Быстрый убыток (меньший штраф): {speed_component:.2f}")
            else:
                # Долгий убыток - большой штраф
                speed_component = -abs(pnl) * time_penalty_multiplier
                logger.debug(f"⚠️ Штраф за долгий убыток: {speed_component:.2f}")

        # 5. ШТРАФ ЗА КОНЦЕНТРАЦИЮ
        concentration_penalty = 0
        if hasattr(self, 'portfolio') and hasattr(self.portfolio, 'positions'):
            positions_count = len(self.portfolio.positions)
            if positions_count > 5:
                concentration_penalty = -10 * (positions_count - 5)
                logger.debug(f"⚠️ Штраф за концентрацию ({positions_count} позиций): {concentration_penalty:.2f}")

        # 6. БОНУС ЗА ТЕРПЕНИЕ (для долгих прибыльных)
        patience_bonus = 0
        if pnl > 0 and hold_time > target_time * 2:
            patience_bonus = pnl * 30
            logger.debug(f"🏆 Бонус за терпение: {patience_bonus:.2f}")

        # 7. Объединяем все компоненты
        pre_multiplier_reward = (base_reward + speed_component +
                                 concentration_penalty + patience_bonus)

        # 8. Применяем стратегический множитель
        final_reward = pre_multiplier_reward * strategy_params.get('risk_multiplier', 1.0)

        # 9. Лимиты из конфига
        max_reward = self.max_reward
        min_reward = self.min_reward
        limited_reward = max(min_reward, min(max_reward, final_reward))

        logger.debug(f"Reward расчет: pnl={pnl:.2f}, hold={hold_time:.2f}ч, "
                     f"base={base_reward:.2f}, speed={speed_component:.2f}, "
                     f"concentration={concentration_penalty:.2f}, patience={patience_bonus:.2f}, "
                     f"strategy_mult={strategy_params.get('risk_multiplier', 1.0):.2f}, "
                     f"final={limited_reward:.2f}")

        return limited_reward

    def _create_initial_state(self, ticker: str, price: float, security_info: Dict) -> torch.Tensor:
        """Создание начального состояния для RL"""
        # Используем существующий метод модели
        momentum = security_info.get('momentum', 0.0)

        # ✅ Получаем сентимент из оптимизированного фетчера
        sentiment = self._get_ticker_sentiment(ticker)

        # Получаем технические данные
        indicators = self.technical_core.calculate_indicators(ticker)

        # Получаем новости
        news_items = self.news_fetcher.search_news(ticker=ticker, limit=3)
        news_texts = [n.get('title', '') + ' ' + n.get('summary', '') for n in news_items]
        news_features = self.model.encode_news(news_texts)

        # ✅ Получаем рыночный сентимент
        market_sentiment = self._get_market_sentiment()

        # ✅ ЕДИНСТВЕННЫЙ ВЫЗОВ build_state_vector
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
                'liquidity': 0.5,
                'market_cap': security_info.get('market_cap', 0),
                'pe_ratio': security_info.get('pe_ratio', 15)
            },
            market_sentiment=market_sentiment
        )

        return state

    def _select_buy_strategy(self, ticker: str, price: float,
                             confidence: float, base_state: torch.Tensor) -> Tuple[str, float, float]:
        # ✅ Проверяем тип base_state
        print(f"   base_state.dtype: {base_state.dtype}")
        if base_state.dtype != torch.float32:
            print(f"   ⚠️ КОНВЕРТИРУЕМ base_state в float32")
            base_state = base_state.to(dtype=torch.float32)

        """
        Выбор стратегии для покупки - МОДЕЛЬ САМА ПРИНИМАЕТ РЕШЕНИЯ
        base_state: базовое состояние размерности 150 (из ticker_states)
        """

        # 1. Получаем объективные данные (не правила!)
        sentiment_score = self._get_ticker_sentiment(ticker)

        # 2. Контекст рынка - только данные, без категорий
        market_context = {
            'market_sentiment': self.model.market_sentiment,
            'volatility': self.model.volatility_index,
            'confidence': confidence,
            'time_of_day': datetime.now().hour / 24.0,
            'ticker_sentiment': sentiment_score,
            # НЕТ sentiment_category - модель сама разберется!
        }

        # 3. Модель САМА выбирает стратегию и действие
        # choose_action_with_strategy внутри вызовет _create_strategy_state,
        # который превратит base_state (150) в full_state (156)
        action, final_strategy, strategy_confidence = self.model.choose_action_with_strategy(
            state=base_state,  # передаем 150
            ticker=ticker,
            price=price,
            market_context=market_context
        )

        # 4. Получаем параметры выбранной стратегии из конфига
        # (это не ручное управление, а базовые настройки, которые модель будет менять через адаптацию)
        strategy_config = self.model.strategies.get(final_strategy, self.model.strategies['balanced'])

        # 5. Базовые значения стоп-лосса и тейк-профита
        base_stop_loss = strategy_config.get('stop_loss_percent', 2.5)
        base_take_profit = strategy_config.get('take_profit_percent', 5.0)

        # 6. Адаптация под волатильность (объективный рыночный фактор)
        volatility_factor = 1.0
        if hasattr(self.model, 'volatility_index'):
            # Чем выше волатильность, тем шире стоп-лосс
            volatility_factor = 1.0 + self.model.volatility_index

        # 7. Адаптация под уверенность модели (чем увереннее, тем агрессивнее)
        confidence_factor = 0.5 + strategy_confidence  # от 0.5 до 1.5

        # 8. Итоговые параметры (без ручных множителей из конфига!)
        adjusted_stop_loss = base_stop_loss * volatility_factor / confidence_factor
        adjusted_take_profit = base_take_profit * volatility_factor * confidence_factor

        # Ограничиваем разумные пределы
        adjusted_stop_loss = max(0.5, min(10.0, adjusted_stop_loss))
        adjusted_take_profit = max(1.0, min(20.0, adjusted_take_profit))

        stop_loss = price * (1 - adjusted_stop_loss / 100)
        take_profit = price * (1 + adjusted_take_profit / 100)

        logger.debug(
            f"🤖 Стратегия {ticker}: {final_strategy} "
            f"(sentiment: {sentiment_score:.3f}, confidence: {strategy_confidence:.2f}) "
            f"SL={adjusted_stop_loss:.1f}%, TP={adjusted_take_profit:.1f}%"
        )

        return final_strategy, stop_loss, take_profit

    def _periodic_learning(self):
        """ОПТИМИЗИРОВАННОЕ онлайн-обучение с приоритетами"""
        try:
            # 🔥 ДИАГНОСТИКА
            print(f"\n📊 ДИАГНОСТИКА ПАМЯТИ:")
            print(f"   memory size: {len(self.model.memory)}")
            if hasattr(self.model, 'prioritized_buffer'):
                print(f"   prioritized_buffer size: {self.model.prioritized_buffer.size}")

            # ✅ ИСПРАВЛЕНО: определяем переменную!
            enable_extreme = self.profit_config.get("enable_extreme_learning", True)

            # 1. ПРИОРИТЕТНОЕ ОБУЧЕНИЕ
            if hasattr(self.model, 'learn_from_prioritized'):
                if self.model.prioritized_buffer.size >= 32:
                    priority_loss = self.model.learn_from_prioritized(batch_size=32)
                    if priority_loss:
                        logger.debug(f"Приоритетное обучение: Loss={priority_loss:.6f}")
                else:
                    print(f"   ⚠️ Недостаточно опытов в prioritized_buffer: {self.model.prioritized_buffer.size} < 32")

            # 2. Критические сделки
            critical_trades = []
            for ticker, pos in self.portfolio.positions.items():
                current_price = self.moex.get_price(ticker)
                if not current_price:
                    continue

                entry_price = pos.get('avg_price', 0)
                if entry_price <= 0:
                    continue

                pnl_pct = (current_price - entry_price) / entry_price * 100

                if abs(pnl_pct) > self.extreme_pnl_threshold * 100:
                    critical_trades.append({
                        'ticker': ticker,
                        'pnl_pct': pnl_pct,
                        'is_profit': pnl_pct > 0
                    })

            # 3. Обучение на критических сделках
            if enable_extreme and critical_trades and len(self.model.memory) > 32:
                extreme_batch = self._create_extreme_batch(critical_trades)
                if extreme_batch:
                    loss = self._train_extreme(extreme_batch)
                    if loss:
                        profit_count = sum(1 for t in critical_trades if t['is_profit'])
                        loss_count = len(critical_trades) - profit_count
                        logger.info(f"[ЭКСТРЕМ-обучение] {len(critical_trades)} сделок "
                                    f"(прибыль: {profit_count}, убытки: {loss_count}) Loss={loss:.6f}")

            # 4. Регулярное обучение
            if self.cycle_count % (self.fast_learning_cycles * 2) == 0:
                if len(self.model.memory) >= 32:
                    regular_loss = self.model.learn_from_experience(batch_size=32)
                    if regular_loss:
                        logger.debug(f"Регулярное обучение: Loss={regular_loss:.6f}")

            # 5. Адаптация стратегий
            if self.cycle_count % self.strategy_adaptation_cycles == 0:
                self._adapt_strategies_for_profit()

        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _train_extreme(self, extreme_batch):
        """АГРЕССИВНОЕ обучение на экстремальных сделках"""
        try:
            if not extreme_batch:
                return None

            learning_config = self.rl_config.get("learning", {})
            aggressive_multiplier = learning_config.get("aggressive_lr_multiplier", 3.0)
            training_steps = learning_config.get("extreme_training_steps", 3)

            # ВРЕМЕННО увеличиваем learning rate
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

                # АГРЕССИВНОЕ обучение
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
        """Создание экстремального батча"""
        try:
            extreme_experiences = []
            critical_tickers = {trade['ticker'] for trade in critical_trades}

            learning_config = self.rl_config.get("learning", {})
            max_size = learning_config.get("extreme_batch_size", 16)

            memory_list = list(self.model.memory)

            for exp in reversed(memory_list):
                if len(extreme_experiences) >= max_size:
                    break

                exp_ticker = None
                if 'ticker' in exp:
                    exp_ticker = exp['ticker']
                elif 'market_conditions' in exp and 'ticker' in exp['market_conditions']:
                    exp_ticker = exp['market_conditions']['ticker']
                elif hasattr(exp, 'get') and callable(exp.get):
                    exp_ticker = exp.get('ticker')

                if exp_ticker in critical_tickers:
                    modified_exp = exp.copy() if hasattr(exp, 'copy') else dict(exp)

                    if 'reward' in modified_exp:
                        reward_value = modified_exp['reward']
                        reward_multiplier = 2.0 if 'is_profit' not in modified_exp or modified_exp.get('is_profit') else 2.5

                        if isinstance(reward_value, (int, float)):
                            modified_exp['reward'] = reward_value * reward_multiplier
                        elif isinstance(reward_value, torch.Tensor):
                            modified_exp['reward'] = reward_value.clone() * reward_multiplier
                        else:
                            continue

                    extreme_experiences.append(modified_exp)

            logger.debug(f"Создан экстремальный батч: {len(extreme_experiences)}/{max_size} записей")
            return extreme_experiences

        except Exception as e:
            logger.error(f"Ошибка создания экстремального батча: {e}")
            return []

    def _create_priority_batch(self, critical_trades, max_size=8):
        """Создание приоритетного батча"""
        if not self.model.memory or len(self.model.memory) < 20:
            return []

        recent_memory = list(self.model.memory)[-50:]
        critical_tickers = {trade['ticker'] for trade in critical_trades}
        priority_experiences = []

        for exp in recent_memory:
            exp_ticker = None
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
            states = torch.stack([exp['state'] for exp in priority_batch]).to(self.model.device)
            actions = torch.LongTensor([exp['action'] for exp in priority_batch]).to(self.model.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in priority_batch]).to(self.model.device)
            next_states = torch.stack([exp['next_state'] for exp in priority_batch]).to(self.model.device)
            dones = torch.FloatTensor([exp['done'] for exp in priority_batch]).to(self.model.device)

            self.model.policy_net.train()

            current_probs, current_values = self.model.policy_net(states)

            with torch.no_grad():
                _, next_values = self.model.policy_net(next_states)

            target_values = rewards + (1 - dones) * self.model.gamma * next_values

            value_loss = torch.nn.SmoothL1Loss()(current_values, target_values.detach())

            dist = torch.distributions.Categorical(current_probs)
            log_probs = dist.log_prob(actions)
            advantages = (target_values - current_values).detach()
            policy_loss = -(log_probs * advantages).mean()

            total_loss = value_loss + policy_loss

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
            sentiment = self._get_ticker_sentiment(ticker)
            return abs(sentiment) > 0.5
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



    def _initialize_components(self):
        """Инициализация всех компонентов"""
        self.portfolio.initial_capital = self.settings.get("initial_capital_rub", 10000)
        self.portfolio.max_positions = self.settings.get("max_positions", 5)

        # ✅ Запуск непрерывного сбора новостей (опционально)
        if self.settings.get("enable_news_core", True):
            # NewsFetcher сам управляет кэшем, запускаем фоновый сбор
            import threading
            def background_fetch():
                while True:
                    self.news_fetcher.get_last_news(limit=100)
                    time.sleep(300)  # Каждые 5 минут

            fetch_thread = threading.Thread(target=background_fetch, daemon=True)
            fetch_thread.start()
            logger.info("Запущен фоновый сбор новостей")

        # Загрузка состояния портфеля
        self._load_portfolio_state()

        # Запуск планировщика задач
        self.scheduler.schedule_daily_tasks(
            pre_market_callback=self.pre_session_analysis,
            market_open_callback=self.on_market_open,
            market_close_callback=self.on_market_close,
            post_market_callback=self.post_session_analysis,
            clearing_liquidity_callback=self.check_clearing_liquidity,
            z0_deadline_callback=self.execute_z0_deadline,
            clearing_17_callback=self.process_clearing_17,
            clearing_19_callback=self.process_clearing_19,
            commission_callback = self.process_pending_commissions
        )

        logger.info("SmartBroker: все компоненты инициализированы")

    def _load_portfolio_state(self):
        """Загрузка состояния портфеля из файла"""
        try:
            with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)

                self.portfolio.positions = state.get('positions', {})
                self.portfolio.cash = state.get('cash', self.settings["initial_capital_rub"])

                for ticker, pos in self.portfolio.positions.items():
                    if 'buy_time' not in pos:
                        pos['buy_time'] = time.time()

                logger.info(f"Загружено состояние портфеля: {len(self.portfolio.positions)} позиций, "
                            f"{self.portfolio.cash:,.0f}₽ кэша")

        except Exception as e:
            logger.warning(f"Не удалось загрузить состояние портфеля: {e}")
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

        self.risk_manager.reset_daily_metrics()

        # Анализ новостей
        self.news_fetcher.get_last_news(limit=100)

        securities = self.moex.get_all_securities()
        logger.info(f"Загружено бумаг для анализа: {len(securities)}")

        # Обновление рыночного настроения
        market_sentiment = self._get_market_sentiment()
        self.model.update_market_sentiment(market_sentiment)

        logger.info("Предсессионный анализ завершен")

    def on_market_open(self):
        """Действия при открытии рынка"""
        logger.info("РЫНОК ОТКРЫТ")
        self.trading_enabled = True

        prices = self._get_current_prices()
        if prices:
            self.check_stops_and_tp(prices)

    def on_market_close(self):
        """Действия при закрытии рынка"""
        logger.info("РЫНОК ЗАКРЫТ")
        self.trading_enabled = False

        self._save_portfolio_state()
        self.model.save_model()

    def post_session_analysis(self):
        """Послерыночный анализ"""
        logger.info("ПОСЛЕРЫНОЧНЫЙ АНАЛИЗ")

        total_value = self.portfolio.get_total_value({})
        initial = self.settings["initial_capital_rub"]
        daily_pnl = total_value - self.portfolio.cash - sum(
            p['qty'] * p['avg_price'] for p in self.portfolio.positions.values()
        )

        logger.info(f"Итоги дня: Капитал {total_value:,.0f}₽, PnL: {daily_pnl:+,.0f}₽")
        logger.info(f"С начала: {((total_value / initial) - 1) * 100:+.2f}%")

        self._generate_daily_report()

    def run_cycle(self):
        """Основной торговый цикл"""

        # 🔥 ДИАГНОСТИКА КАЖДЫЙ ЦИКЛ
        print(f"\n{'=' * 60}")
        print(f"ЦИКЛ #{self.cycle_count}")
        print(f"pending_experiences: {len(self.pending_experiences)}")
        print(f"позиций: {len(self.portfolio.positions)}")
        print(f"{'=' * 60}")

        # Проверка комиссий
        current_hour = datetime.now().hour
        if (self.tbank_check_start <= current_hour < self.tbank_check_end and
                self.cycle_count % self.tbank_check_interval == 0):
            self.process_pending_commissions()

        if not self.trading_enabled or not self.scheduler.is_trading_time():
            return

        period_info = self.scheduler.can_trade_now()
        self.auction_mode = period_info['current_period'] in ['auction_open', 'auction_close']

        if hasattr(self, 'model') and self.model:
            self.model.market_period = period_info['current_period']

        if self.cycle_count % 10 == 0:
            logger.info(f"Текущий период MOEX: {period_info['current_period']}, "
                        f"аукцион: {self.auction_mode}")

        if not period_info['can_place_orders']:
            logger.debug(f"Торговля недоступна: период {period_info['current_period']}")
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

            # ✅ Новостные сигналы (исправлено!)
            if self.settings.get("enable_news_core", True):
                news_signals = self._generate_news_signals(prices)
                all_signals.extend(news_signals)
                logger.debug(f"Сгенерировано новостных сигналов: {len(news_signals)}")

            # Технические сигналы
            if self.settings.get("enable_technical_core", True):
                tech_signals = self.technical_core.analyze_all_tickers(prices)
                all_signals.extend(tech_signals)
                logger.debug(f"Сгенерировано технических сигналов: {len(tech_signals)}")

            # 4. Агрегация и фильтрация сигналов
            filtered_signals = self._aggregate_signals(all_signals)
            self.signals_cache = filtered_signals[:10]

            # 5. Проверка стоп-лоссов и тейк-профитов
            self.check_stops_and_tp(prices)

            # 6. Исполнение торговых решений
            if filtered_signals and self.risk_manager.check_daily_limits():
                self._execute_trading_decisions(filtered_signals, prices, securities)

            # 7. Ребалансировка портфеля
            if self.cycle_count % 5 == 0:
                self._rebalance_portfolio(prices, securities)

            # 8. Периодическое обучение модели
            if self.cycle_count % 10 == 0:
                self._periodic_learning()

            # 9. Адаптация стратегий
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
        """Получение критически важных сделок"""
        critical_trades = []

        for ticker, pos in self.portfolio.positions.items():
            current_price = self.moex.get_price(ticker)
            if not current_price:
                continue

            entry_price = pos.get('avg_price', 0)
            if entry_price <= 0:
                continue

            pnl_pct = (current_price - entry_price) / entry_price * 100

            is_critical = (
                    abs(pnl_pct) > 5 or
                    pos.get('strategy') in ['news_aggressive', 'momentum'] or
                    self._has_high_impact_news(ticker)
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

        ticker_signals = defaultdict(list)
        for signal in signals:
            ticker = signal['ticker']
            ticker_signals[ticker].append(signal)

        aggregated = []

        for ticker, sig_list in ticker_signals.items():
            action_scores = defaultdict(float)

            for sig in sig_list:
                action = sig['action']
                confidence = sig.get('confidence', 0.5)

                weight = confidence
                if sig.get('reason') == 'news_analysis':
                    weight *= 1.2

                action_scores[action] += weight

            if action_scores:
                best_action = max(action_scores.items(), key=lambda x: x[1])

                if best_action[1] > self.settings.get('model_confidence_threshold', 0.65):
                    aggregated_signal = {
                        'ticker': ticker,
                        'action': best_action[0],
                        'confidence': best_action[1] / len(sig_list),
                        'source_count': len(sig_list),
                        'sources': [s.get('reason', 'unknown') for s in sig_list],
                        'timestamp': datetime.now().isoformat()
                    }
                    aggregated.append(aggregated_signal)

        aggregated.sort(key=lambda x: x['confidence'], reverse=True)

        return aggregated

    def _record_trade_for_learning(self, ticker: str, action: str, price: float,
                                   quantity: int, confidence: float, pnl: float = 0.0):
        """Запись сделки для обучения модели"""
        try:
            sentiment = self._get_ticker_sentiment(ticker)

            if action == 'BUY':
                self.model.record_trade_outcome(
                    ticker=ticker,
                    action=action,
                    entry_price=price,
                    exit_price=price,
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
                hold_time = (time.time() - pos.get('buy_time', time.time())) / 3600

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
        """Проверка стоп-лоссов и тейк-профитов с учетом лотности"""
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

            lot_size = pos.get('lot_size', 1)
            min_step = pos.get('min_step', 0.01)

            if min_step > 0:
                price = round(price / min_step) * min_step

            # Стоп-лосс
            if change_pct <= -cfg.get('stop_loss_percent', 3.0):
                qty = pos['qty']

                if lot_size > 1 and qty % lot_size != 0:
                    qty = (qty // lot_size) * lot_size
                    if qty == 0:
                        qty = lot_size

                if qty > 0 and self.portfolio.sell(ticker, qty, price):
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
                qty = pos['qty'] // 2

                if lot_size > 1:
                    qty = (qty // lot_size) * lot_size
                    if qty == 0:
                        qty = lot_size

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
        """Ребалансировка портфеля с учетом лотности"""
        current_count = len(self.portfolio.positions)
        target_count = self.settings.get('target_positions', 4)

        if current_count < target_count and self.portfolio.cash > self.settings["min_cash_per_trade"]:
            # Получаем сентимент для всех тикеров
            ticker_sentiment = {t: self._get_ticker_sentiment(t) for t in prices.keys()}

            # Строим список кандидатов
            candidates = []
            for ticker, price in prices.items():
                if ticker in self.portfolio.positions:
                    continue

                sentiment = ticker_sentiment.get(ticker, 0)
                volume = securities.get(ticker, {}).get('volume', 0)

                # Простая эвристика: позитивный сентимент + объем
                score = sentiment * 100 + volume / 1e6
                candidates.append((ticker, score))

            candidates.sort(key=lambda x: x[1], reverse=True)

            bought = 0
            for ticker, score in candidates[:5]:
                if bought >= 2:
                    break

                security_info = securities.get(ticker, {})
                lot_size = security_info.get('lot_size', 1)
                min_step = security_info.get('min_step', 0.01)

                price = prices[ticker]

                if min_step > 0:
                    price = round(price / min_step) * min_step

                max_qty_by_cash = int(self.portfolio.cash * 0.15 / price)

                if lot_size > 1:
                    max_qty_by_cash = (max_qty_by_cash // lot_size) * lot_size

                qty = max(lot_size, max_qty_by_cash // 2)

                if lot_size > 1 and qty % lot_size != 0:
                    qty = (qty // lot_size) * lot_size

                if qty >= lot_size:
                    if self.portfolio.buy(ticker, qty, price, 'balanced',
                                          lot_size=lot_size,
                                          min_step=min_step):
                        self.portfolio.positions[ticker]['buy_time'] = time.time()
                        bought += 1

                        logger.info(f"Ребалансировка BUY: {ticker} {qty} @ {price:.2f} "
                                    f"(лот: {lot_size}, score: {score:.2f})")

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
                'market_sentiment': self._get_market_sentiment(),
                'signals_generated': len(self.signals_cache),
                'risk_metrics': self.risk_manager.get_risk_metrics()
            }

            report_file = f"data/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Дневной отчет сохранен: {report_file}")

        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")

    def analyze_sentiment(self, force: bool = False):
        """Анализ рыночного настроения"""
        try:
            market_sentiment = self._get_market_sentiment()
            self.model.update_market_sentiment(market_sentiment)
            logger.info(f"Рыночное настроение обновлено: {market_sentiment:+.3f}")
        except Exception as e:
            logger.error(f"Ошибка анализа сентимента: {e}")

    def _create_next_state(self, ticker: str, exit_price: float) -> torch.Tensor:
        """Создание следующего состояния для RL"""
        try:
            current_price = self.moex.get_price(ticker)
            if not current_price:
                current_price = exit_price

            securities = self.moex.get_all_securities()
            security_info = securities.get(ticker, {})

            sentiment = self._get_ticker_sentiment(ticker)

            indicators = {}
            try:
                indicators = self.technical_core.calculate_indicators(ticker)
            except:
                pass

            news_items = self.news_fetcher.search_news(ticker=ticker, limit=3)
            news_texts = [n.get('title', '') + ' ' + n.get('summary', '') for n in news_items]
            news_features = self.model.encode_news(news_texts)

            market_sentiment = self._get_market_sentiment()

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
                },
                market_sentiment=market_sentiment
            )

            return state.to(self.model.device)

        except Exception as e:
            logger.error(f"Ошибка создания следующего состояния для {ticker}: {e}")
            # ✅ ИСПРАВЛЕНО: создаем состояние правильной размерности
            # Пытаемся получить текущую цену для базового состояния
            try:
                current_price = self.moex.get_price(ticker) or exit_price
                security_info = securities.get(ticker, {}) if 'securities' in locals() else {}

                # Создаем простое состояние с правильной размерностью
                dummy_state = self.model.build_state_vector(
                    ticker=ticker,
                    price=current_price,
                    momentum=0.0,
                    sentiment=0.0,
                    news_features=torch.zeros(1, NEWS_ENCODED_DIM).to(self.model.device),
                    market_data={
                        'volume': 1.0,
                        'spread': 0.01,
                        'rsi': 50,
                        'volatility': 0.1,
                        'sma_10_ratio': 1.0,
                        'sma_20_ratio': 1.0,
                        'bb_position': 0.5,
                        'liquidity': 0.5,
                        'market_cap': 0,
                        'pe_ratio': 15
                    },
                    market_sentiment=self._get_market_sentiment() if hasattr(self, '_get_market_sentiment') else 0.0
                )
                return dummy_state.to(self.model.device)
            except:
                # Если совсем ничего не работает - создаем нулевой тензор 156
                logger.warning(f"⚠️ Создание нулевого состояния для {ticker}")
                return torch.zeros(156, dtype=torch.float32).to(self.model.device)

    def _adapt_strategies_for_profit(self):
        """Адаптация стратегий - ТОЛЬКО на основе реальных результатов"""
        try:
            for strategy_name, perf in self.model.strategy_performance.items():
                if perf['total_trades'] >= 5:  # минимальная статистика
                    win_rate = perf['win_rate']
                    avg_pnl = perf['avg_pnl']

                    strategy = self.model.strategies[strategy_name]

                    # 📈 УСПЕШНАЯ СТРАТЕГИЯ - усиливаем пропорционально прибыли
                    if avg_pnl > 0:
                        # Сила усиления зависит от размера прибыли
                        strength = min(avg_pnl * 10, 0.5)  # максимум +50%

                        # Увеличиваем риск, но не больше 2.0
                        old_risk = strategy['risk_multiplier']
                        strategy['risk_multiplier'] = min(old_risk * (1 + strength), 2.0)

                        # Уменьшаем время удержания (быстрее фиксируем прибыль)
                        old_hold = strategy['target_hold_time_hours']
                        strategy['target_hold_time_hours'] = max(old_hold * (1 - strength * 0.5), 0.5)

                        logger.info(f"📈 Усиление {strategy_name}: "
                                    f"risk {old_risk:.2f}→{strategy['risk_multiplier']:.2f} "
                                    f"(+{strength:.1%}, avg_pnl={avg_pnl:.2%})")

                    # 📉 УБЫТОЧНАЯ СТРАТЕГИЯ - ослабляем пропорционально убытку
                    elif avg_pnl < 0:
                        # Сила ослабления зависит от размера убытка
                        weakness = min(abs(avg_pnl) * 10, 0.5)  # максимум -50%

                        # Уменьшаем риск, но не меньше 0.5
                        old_risk = strategy['risk_multiplier']
                        strategy['risk_multiplier'] = max(old_risk * (1 - weakness), 0.5)

                        # Увеличиваем время удержания (даём больше времени на разворот)
                        old_hold = strategy['target_hold_time_hours']
                        strategy['target_hold_time_hours'] = min(old_hold * (1 + weakness), 24)

                        logger.info(f"📉 Ослабление {strategy_name}: "
                                    f"risk {old_risk:.2f}→{strategy['risk_multiplier']:.2f} "
                                    f"(-{weakness:.1%}, avg_pnl={avg_pnl:.2%})")

                    # Адаптация стоп-лосса под волатильность (объективный фактор)
                    if hasattr(self.model, 'volatility_index'):
                        old_stop = strategy.get('stop_loss_percent', 2.5)
                        old_take = strategy.get('take_profit_percent', 5.0)

                        # Базовая настройка: 2% на стоп, 4% на тейк при нормальной волатильности
                        base_stop = 2.0
                        base_take = 4.0

                        # Корректировка под текущую волатильность
                        vol_factor = 1.0 + self.model.volatility_index
                        strategy['stop_loss_percent'] = base_stop * vol_factor
                        strategy['take_profit_percent'] = base_take * vol_factor

                        logger.debug(f"📊 Адаптация под волатильность {strategy_name}: "
                                     f"vol={self.model.volatility_index:.2f}, "
                                     f"stop {old_stop:.1f}→{strategy['stop_loss_percent']:.1f}%, "
                                     f"take {old_take:.1f}→{strategy['take_profit_percent']:.1f}%")

            # Сохраняем обновленные стратегии
            self.model.save_model()
            logger.info("✅ Стратегии адаптированы на основе реальных результатов")

        except Exception as e:
            logger.error(f"Ошибка адаптации стратегий: {e}")

    def get_portfolio_summary(self) -> Dict:
        """Получение сводки портфеля"""
        prices = self._get_current_prices()
        total_value = self.portfolio.get_total_value(prices)

        """ Расчет pending комиссий"""
        pending_total = 0
        pending_count = 0
        if hasattr(self.portfolio, 'pending_commissions'):
            for comm in self.portfolio.pending_commissions:
                if not comm.get('processed', False):
                    pending_total += comm['amount']
                    pending_count += 1

        summary = {
            'total_value': total_value,
            'cash': self.portfolio.cash,
            'reserved_cash': self.portfolio.reserved_cash,
            'available_cash': self.portfolio.cash - self.portfolio.reserved_cash,
            'pending_commissions': {
                'count': pending_count,
                'total': pending_total,
                'next_settlement': self._get_next_settlement_date()
            },
            'positions_count': len(self.portfolio.positions),
            'initial_capital': self.settings["initial_capital_rub"],
            'total_pnl': total_value - self.settings["initial_capital_rub"],
            'pnl_percent': ((total_value / self.settings["initial_capital_rub"]) - 1) * 100,
            'current_signals': self.signals_cache[:5],
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'session_info': self.scheduler.get_session_info(),
            'last_update': datetime.now().isoformat(),
            'market_sentiment': self._get_market_sentiment(),
            'news_stats': self.news_fetcher.stats if hasattr(self.news_fetcher, 'stats') else {}
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

    def _get_next_settlement_date(self) -> Optional[str]:
        """Получение ближайшей даты списания"""
        if not hasattr(self.portfolio, 'pending_commissions'):
            return None

        upcoming = [
            c for c in self.portfolio.pending_commissions
            if not c.get('processed', False)
        ]

        if not upcoming:
            return None

        upcoming.sort(key=lambda x: x['settlement_date'])
        return upcoming[0]['settlement_date']

    def update_settings(self, new_settings: Dict):
        """Обновление настроек системы"""
        try:
            self.settings.update(new_settings)

            updates_to_apply = []

            if 'initial_capital_rub' in new_settings:
                self.portfolio.initial_capital = new_settings['initial_capital_rub']
                updates_to_apply.append(f"initial_capital_rub={new_settings['initial_capital_rub']}")

            if 'max_positions' in new_settings:
                self.portfolio.max_positions = new_settings['max_positions']
                updates_to_apply.append(f"max_positions={new_settings['max_positions']}")

            if 'risk_per_trade_percent' in new_settings:
                if hasattr(self.risk_manager, 'config'):
                    self.risk_manager.config['risk_per_trade_percent'] = new_settings['risk_per_trade_percent']
                    updates_to_apply.append(f"risk_per_trade_percent={new_settings['risk_per_trade_percent']}%")

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

    # ============================================
    # БЭКОФИС ФУНКЦИИ
    # ============================================

    def check_clearing_liquidity(self):
        """16:00 - Проверка ликвидности"""
        logger.info("=" * 60)
        logger.info("🔰 БЭКОФИС: ПРОВЕРКА ЛИКВИДНОСТИ (16:00)")
        logger.info("=" * 60)

        try:
            prices = self._get_current_prices()

            total_exposure = 0
            positions_detail = []

            for ticker, pos in self.portfolio.positions.items():
                current_price = prices.get(ticker, pos['avg_price'])
                position_value = pos['qty'] * current_price
                total_exposure += position_value

                positions_detail.append({
                    'ticker': ticker,
                    'value': position_value,
                    'qty': pos['qty'],
                    'price': current_price
                })

            liquidity_ratio = self.portfolio.cash / total_exposure if total_exposure > 0 else 1.0
            min_ratio = self.settings.get('back_office', {}).get('min_liquidity_ratio', 0.2)

            report = {
                'timestamp': datetime.now().isoformat(),
                'cash': self.portfolio.cash,
                'total_exposure': total_exposure,
                'liquidity_ratio': liquidity_ratio,
                'min_required_ratio': min_ratio,
                'is_sufficient': liquidity_ratio >= min_ratio,
                'positions_count': len(self.portfolio.positions),
                'positions': positions_detail
            }

            logger.info(f"💰 Кэш: {self.portfolio.cash:,.0f}₽")
            logger.info(f"📊 Обязательства: {total_exposure:,.0f}₽")
            logger.info(f"⚖ Коэффициент ликвидности: {liquidity_ratio:.2%} (мин: {min_ratio:.2%})")

            if liquidity_ratio < min_ratio:
                logger.warning(f"⚠ КРИТИЧЕСКИ НИЗКАЯ ЛИКВИДНОСТЬ!")
                self._send_liquidity_alert(report)
            else:
                logger.info(f"✅ Ликвидность достаточная")

            self._save_liquidity_report(report)
            return report

        except Exception as e:
            logger.error(f"❌ Ошибка проверки ликвидности: {e}")
            return {'error': str(e)}

    def execute_z0_deadline(self):
        """16:50 - Дедлайн сделок Z0"""
        logger.info("=" * 60)
        logger.info("🔰 БЭКОФИС: ДЕДЛАЙН Z0 (16:50)")
        logger.info("=" * 60)

        try:
            back_config = self.settings.get('back_office', {})
            auto_close = back_config.get('auto_close_positions_at_deadline', False)

            if not auto_close:
                logger.info("⏸ Автоматическое закрытие позиций отключено")
                return {'action': 'skipped', 'reason': 'auto_close_disabled'}

            liquidity = self.check_clearing_liquidity()

            if liquidity.get('is_sufficient', True):
                logger.info("✅ Ликвидность достаточная, закрытие не требуется")
                return {'action': 'none', 'liquidity': liquidity}

            max_close_percent = back_config.get('max_position_close_percent', 50)
            closed_positions = []

            prices = self._get_current_prices()

            for ticker, pos in list(self.portfolio.positions.items()):
                if ticker not in prices:
                    continue

                qty_to_sell = int(pos['qty'] * max_close_percent / 100)
                lot_size = pos.get('lot_size', 1)

                if lot_size > 1:
                    qty_to_sell = (qty_to_sell // lot_size) * lot_size

                if qty_to_sell >= lot_size:
                    if self.portfolio.sell(ticker, qty_to_sell, prices[ticker]):
                        closed_positions.append({
                            'ticker': ticker,
                            'qty': qty_to_sell,
                            'price': prices[ticker]
                        })
                        logger.info(f"📉 Частично закрыта {ticker}: {qty_to_sell} @ {prices[ticker]:.2f}")

            result = {
                'action': 'partial_close',
                'closed_positions': closed_positions,
                'cash_after': self.portfolio.cash
            }

            logger.info(f"✅ Дедлайн Z0 выполнен, закрыто {len(closed_positions)} позиций")
            return result

        except Exception as e:
            logger.error(f"❌ Ошибка выполнения дедлайна Z0: {e}")
            return {'error': str(e)}

    def process_clearing_17(self):
        """17:00 - Первый клиринг"""
        logger.info("=" * 60)
        logger.info("🔰 БЭКОФИС: КЛИРИНГ 17:00")
        logger.info("=" * 60)

        try:
            prices = self._get_current_prices()
            portfolio_value = self.portfolio.get_total_value(prices)

            state = {
                'timestamp': datetime.now().isoformat(),
                'clearing': '17:00',
                'portfolio_value': portfolio_value,
                'cash': self.portfolio.cash,
                'positions': len(self.portfolio.positions),
                'open_pnl': portfolio_value - self.portfolio.cash - sum(
                    p['qty'] * p['avg_price'] for p in self.portfolio.positions.values()
                )
            }

            self._save_clearing_state(state)

            logger.info(f"📊 Портфель на 17:00: {portfolio_value:,.0f}₽")
            logger.info(f"💰 Кэш: {self.portfolio.cash:,.0f}₽")

            return state

        except Exception as e:
            logger.error(f"❌ Ошибка клиринга 17:00: {e}")
            return {'error': str(e)}

    def process_clearing_19(self):
        """19:00 - Расчет комиссии и постановка в очередь"""
        logger.info("=" * 60)
        logger.info("🔰 БЭКОФИС: КЛИРИНГ 19:00")
        logger.info("=" * 60)

        try:
            daily_trades = getattr(self.portfolio, 'daily_trades', [])
            total_turnover = sum(t.get('value', 0) for t in daily_trades)
            total_reserved = sum(t.get('commission_reserved', 0) for t in daily_trades)

            if total_reserved == 0:
                logger.info("💰 Комиссия не начислена (нет сделок)")
                return {'commission': 0, 'turnover': 0}

            today = datetime.now()

            weekday_map = self.tbank_config.get('weekday_settlement_map', {
                "4": 3,
                "5": 2,
                "6": 1
            })

            days_to_add = self.tbank_config.get('default_settlement_days', 1)

            weekday_str = str(today.weekday())
            if weekday_str in weekday_map:
                days_to_add = weekday_map[weekday_str]

            settlement_date = today + timedelta(days=days_to_add)

            commission_record = {
                'date': today.strftime('%Y-%m-%d'),
                'amount': total_reserved,
                'turnover': total_turnover,
                'settlement_date': settlement_date.strftime('%Y-%m-%d'),
                'settlement_time': self.tbank_settlement_time,
                'processed': False,
                'created_at': datetime.now().isoformat()
            }

            if not hasattr(self.portfolio, 'pending_commissions'):
                self.portfolio.pending_commissions = []

            self.portfolio.pending_commissions.append(commission_record)

            logger.info(f"📅 Комиссия за {today.strftime('%d.%m.%Y')}: {total_reserved:,.2f}₽")
            logger.info(f"📅 Резерв заморожен: {self.portfolio.reserved_cash:,.2f}₽")
            logger.info(f"📅 Будет списана {settlement_date.strftime('%d.%m.%Y')} до {self.tbank_settlement_time}")

            if hasattr(self.portfolio, 'reset_daily_trades'):
                self.portfolio.reset_daily_trades()

            self._generate_daily_report()

            return {
                'commission': total_reserved,
                'turnover': total_turnover,
                'settlement_date': settlement_date.strftime('%Y-%m-%d'),
                'settlement_time': self.tbank_settlement_time,
                'reserved_cash': self.portfolio.reserved_cash
            }

        except Exception as e:
            logger.error(f"❌ Ошибка клиринга 19:00: {e}")
            return {'error': str(e)}

    def process_pending_commissions(self):
        """Проверка и списание pending комиссий"""
        if not hasattr(self.portfolio, 'pending_commissions'):
            return

        today = datetime.now().date()
        current_hour = datetime.now().hour

        if current_hour >= self.tbank_check_end:
            return

        processed_count = 0
        total_commission = 0

        for comm in list(self.portfolio.pending_commissions):
            if comm.get('processed', False):
                continue

            settlement_date = datetime.strptime(comm['settlement_date'], '%Y-%m-%d').date()

            if settlement_date <= today:
                if self.portfolio.reserved_cash >= comm['amount']:
                    self.portfolio.reserved_cash -= comm['amount']
                    comm['processed'] = True
                    comm['processed_date'] = today.strftime('%Y-%m-%d')
                    comm['processed_time'] = datetime.now().strftime('%H:%M')

                    processed_count += 1
                    total_commission += comm['amount']

                    logger.info(f"💸 СПИСАНО: комиссия {comm['amount']:,.2f}₽ за {comm['date']}")
                else:
                    logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА: недостаточно резерва для комиссии {comm['amount']:,.2f}₽!")
                    logger.error(f"💰 Резерв: {self.portfolio.reserved_cash:,.2f}₽")
                    self.portfolio.cash -= (comm['amount'] - self.portfolio.reserved_cash)
                    self.portfolio.reserved_cash = 0
                    comm['processed'] = True
                    comm['forced'] = True

        if processed_count > 0:
            logger.info(f"✅ Обработано {processed_count} комиссий на сумму {total_commission:,.2f}₽")
            logger.info(f"💰 Остаток резерва: {self.portfolio.reserved_cash:,.2f}₽")

        self.portfolio.pending_commissions = [
            c for c in self.portfolio.pending_commissions
            if not c.get('processed', False)
        ]

    def _send_liquidity_alert(self, report: Dict):
        """Отправка alert о низкой ликвидности"""
        try:
            back_config = self.settings.get('back_office', {})
            if not back_config.get('alert_on_low_liquidity', True):
                return

            email = back_config.get('alert_email', 'backoffice@example.com')

            logger.warning(f"📧 ALERT: Низкая ликвидность! Требуется довнесение средств")
            logger.warning(f"📧 Кому: {email}")
            logger.warning(f"📧 Детали: cash={report['cash']:.0f}₽, exposure={report['total_exposure']:.0f}₽")

        except Exception as e:
            logger.error(f"Ошибка отправки alert: {e}")

    def _save_liquidity_report(self, report: Dict):
        """Сохранение отчета о ликвидности"""
        try:
            import os
            os.makedirs("data/backoffice", exist_ok=True)

            filename = f"data/backoffice/liquidity_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"💾 Отчет ликвидности сохранен: {filename}")

        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")

    def _save_clearing_state(self, state: Dict):
        """Сохранение состояния клиринга"""
        try:
            import os
            os.makedirs("data/backoffice", exist_ok=True)

            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"data/backoffice/clearing_{date_str}.json"

            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except:
                history = []

            history.append(state)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, default=str)

            logger.info(f"💾 Состояние клиринга сохранено")

        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def get_sentiment_history(self, limit: int = 100) -> List[Dict]:
        """Получение истории сентимента для веб-дашборда"""
        try:
            # Получаем новости с сентиментом
            news_items = self.news_fetcher.get_last_news(limit=limit)

            if not news_items:
                return []

            # Анализируем сентимент
            news_with_sentiment = self.news_fetcher.analyze_sentiment_batch(news_items)

            # Преобразуем в формат для дашборда
            sentiment_history = []
            for news in news_with_sentiment:
                # Извлекаем тикеры из новости
                ticker = self._extract_ticker_from_news(news)

                sentiment_history.append({
                    'timestamp': news.get('published', datetime.now().isoformat()),
                    'sentiment': news.get('sentiment', 0.0),
                    'source': news.get('source', 'unknown'),
                    'ticker': ticker or 'MARKET',
                    'title': news.get('title', '')[:50]  # Для подсказки
                })

            # Сортируем по времени (новые сначала)
            sentiment_history.sort(key=lambda x: x['timestamp'], reverse=True)

            return sentiment_history[:limit]

        except Exception as e:
            logger.error(f"Ошибка получения истории сентимента: {e}")
            return []

    def _extract_ticker_from_news(self, news: Dict) -> Optional[str]:
        """Извлечение тикера из новости с использованием всех доступных тикеров"""
        try:
            title = news.get('title', '').upper()

            # 🔥 ПОЛУЧАЕМ ВСЕ ТИКЕРЫ ИЗ ПОРТФЕЛЯ И ТЕКУЩЕГО СПИСКА
            all_tickers = set()

            # Добавляем тикеры из портфеля
            if hasattr(self, 'portfolio') and self.portfolio.positions:
                all_tickers.update(self.portfolio.positions.keys())

            # Добавляем тикеры из текущего списка
            if hasattr(self, 'current_tickers') and self.current_tickers:
                all_tickers.update(self.current_tickers)

            # Добавляем популярные тикеры как fallback
            popular_tickers = ['SBER', 'GAZP', 'LKOH', 'ROSN', 'VTBR',
                               'GMKN', 'NVTK', 'YNDX', 'TATN', 'PLZL',
                               'MTSS', 'MOEX', 'AFLT', 'MGNT', 'NLMK']
            all_tickers.update(popular_tickers)

            # Ищем любой тикер в заголовке
            for ticker in all_tickers:
                # Ищем как отдельное слово (с границами слова)
                if f" {ticker} " in f" {title} ":
                    return ticker
                # Ищем в начале строки
                if title.startswith(ticker + " "):
                    return ticker
                # Ищем в конце строки
                if title.endswith(" " + ticker):
                    return ticker
                # Простое вхождение (для коротких заголовков)
                if ticker in title and len(ticker) > 3:  # Избегаем ложных срабатываний
                    return ticker

            # Проверяем расшифровки компаний
            company_names = {
                'СБЕР': 'SBER',
                'ГАЗПРОМ': 'GAZP',
                'ЛУКОЙЛ': 'LKOH',
                'РОСНЕФТЬ': 'ROSN',
                'ВТБ': 'VTBR',
                'НОРНИКЕЛЬ': 'GMKN',
                'НОВАТЭК': 'NVTK',
                'ЯНДЕКС': 'YNDX',
                'ТАТНЕФТЬ': 'TATN',
                'ПОЛЮС': 'PLZL'
            }

            for name, ticker in company_names.items():
                if name in title:
                    return ticker

            return None

        except Exception as e:
            logger.error(f"Ошибка извлечения тикера: {e}")
            return None

    def shutdown(self):
        """Корректное завершение работы с ПОЛНЫМ СОХРАНЕНИЕМ ОПЫТА"""
        logger.info("=" * 60)
        logger.info("🛑 ЗАВЕРШЕНИЕ РАБОТЫ SMART BROKER")
        logger.info("=" * 60)

        try:
            # 1. Сохраняем все незавершенные опыты
            pending_count = len(self.pending_experiences)
            if pending_count > 0:
                logger.info(f"Завершаем {pending_count} незавершенных опытов...")
                for exp in list(self.pending_experiences):
                    if not exp.get('completed', False):
                        current_price = self.moex.get_price(exp['ticker'])
                        if current_price:
                            self._complete_rl_experience(exp['ticker'], current_price)
                        else:
                            # Если не можем получить цену, завершаем принудительно
                            exp['completed'] = True
                            logger.warning(f"Принудительно завершен опыт {exp['ticker']}")

            # 2. Сохраняем приоритетную память
            if hasattr(self.model, 'prioritized_buffer') and self.model.prioritized_buffer.size > 0:
                logger.info(f"Сохраняем приоритетную память: {self.model.prioritized_buffer.size} опытов")
                # Приоритетная память сохраняется через обычный save_memory

            # 3. Сохраняем обычную память
            if len(self.model.memory) > 0:
                self.model.save_memory()
                logger.info(f"✅ Память модели сохранена: {len(self.model.memory)} опытов")

            # 4. Сохраняем модель
            self.model.save_model()
            logger.info("✅ Модель сохранена")

            # 5. Сохраняем портфель
            self._save_portfolio_state()
            logger.info("✅ Портфель сохранен")

            # 6. Останавливаем фоновые процессы
            if hasattr(self, 'trainer') and self.trainer:
                self.trainer.save_memory()
                logger.info("✅ Память тренера сохранена")

            logger.info("=" * 60)
            logger.info("✅ SmartBroker успешно завершил работу")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Ошибка при завершении: {e}")
            import traceback
            logger.error(traceback.format_exc())