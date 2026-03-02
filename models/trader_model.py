"""
Модернизированная модель трейдера с улучшенной архитектурой
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
from utils.logger import get_logger
logger = get_logger("TRADER_MODEL")

class PrioritizedReplayBuffer:
    """Буфер с приоритетами для важных опытов (TD-error)"""

    def __init__(self, max_size=5000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = []
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.max_size = max_size
        self.alpha = alpha  # Степень приоритизации (0 = равномерно, 1 = только приоритетные)
        self.beta = beta  # Степень коррекции смещения (растет со временем)
        self.beta_increment = beta_increment
        self.position = 0
        self.size = 0

    def add(self, experience, td_error=None):
        """Добавление опыта с приоритетом"""
        if td_error is None:
            # По умолчанию - максимальный приоритет для новых
            priority = self.priorities.max() if self.size > 0 else 1.0
        else:
            priority = (abs(td_error) + 1e-6) ** self.alpha

        if self.size < self.max_size:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        """Выборка с учетом приоритетов"""
        if self.size < batch_size:
            return None, None, None

        # Нормализуем приоритеты в вероятности
        priorities = self.priorities[:self.size]
        probs = priorities / priorities.sum()

        # Выбираем индексы по вероятностям
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Корректируем веса для обучения (importance sampling)
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # Увеличиваем beta для уменьшения смещения
        self.beta = min(1.0, self.beta + self.beta_increment)

        samples = [self.buffer[idx] for idx in indices]

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        """Обновление приоритетов после обучения"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha

warnings.filterwarnings('ignore')

# ========== КОНСТАНТЫ МОДЕЛИ ==========
# Архитектурные параметры
# NEWS_EMBEDDING_DIM = 768
NEWS_EMBEDDING_DIM = 312  # (rubert-tiny2 выдает 312)

NEWS_ENCODED_DIM = 128
BASE_STATE_DIM = 150
STRATEGY_PARAMS_DIM = 6
TOTAL_STATE_DIM = BASE_STATE_DIM + STRATEGY_PARAMS_DIM  # 156

# Слои нейросетей
NEWS_ENCODER_HIDDEN_DIM = 256
NEWS_ENCODER_INTERMEDIATE_DIM = 512
POLICY_HIDDEN_DIM_1 = 256
POLICY_HIDDEN_DIM_2 = 128
POLICY_HIDDEN_DIM_3 = 64
POLICY_HIDDEN_DIM_4 = 32

# Параметры обучения
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_GAMMA = 0.95
DEFAULT_MEMORY_SIZE = 5000
DEFAULT_BATCH_SIZE = 32
MIN_EXPERIENCES_FOR_LEARNING = 100

# Параметры регуляризации
DROPOUT_RATE_1 = 0.2
DROPOUT_RATE_2 = 0.3
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_VALUE = 0.5
ENTROPY_BONUS_COEFF = 0.01

# Параметры стратегии
DEFAULT_EXPLORATION_RATE = 0.3
DEFAULT_CONFIDENCE_BOOST_FACTOR = 0.4
ACTION_EXPLORATION_RATE = 0.2
STRATEGY_MEMORY_SIZE = 2000
MIN_TRADES_FOR_EVALUATION = 10

# Параметры рынка и риска
MAX_SENTIMENT_HISTORY = 200
VOLATILITY_SCALING_FACTOR = 3.0
BASE_VOLATILITY = 0.7
SENTIMENT_SMOOTHING_ALPHA = 0.1
RISK_BASE = 0.4
MAX_RISK = 0.85
MIN_RISK = 0.1

# Параметры торговли
PROFIT_THRESHOLD = 0.05  # 5%
LOSS_THRESHOLD = -0.03  # -3%
SIGNIFICANT_LOSS = -0.02  # -2%
SMALL_PROFIT = 0.01  # 1%
MIN_HOLD_TIME = 0.5  # часов

# Нормализация
PRICE_NORMALIZATION = 10000.0
VOLUME_NORMALIZATION = 1e7
MOMENTUM_SCALING = 20.0
VOLATILITY_SCALING = 10.0
PE_NORMALIZATION = 100.0
RSI_NORMALIZATION = 100.0

# Ранжирование
TOP_CANDIDATES_LIMIT = 30
MAX_TRADES_FOR_EXPERIENCE = 50
MAX_HOLD_TIME_DAYS = 7.0

# Автосохранение
AUTO_SAVE_INTERVAL = 50


# ========== КЛАССЫ МОДЕЛИ ==========

class NewsEncoder(nn.Module):
    """Улучшенный энкодер новостей с поддержкой разных размерностей"""

    def __init__(self, input_dim=NEWS_EMBEDDING_DIM, hidden_dim=NEWS_ENCODER_HIDDEN_DIM):
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, NEWS_ENCODER_INTERMEDIATE_DIM),
            nn.LayerNorm(NEWS_ENCODER_INTERMEDIATE_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_1),

            nn.Linear(NEWS_ENCODER_INTERMEDIATE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_1),

            nn.Linear(hidden_dim, NEWS_ENCODED_DIM),
            nn.Tanh()
        )

        # Адаптивный слой для разных размерностей
        self.adaptor = nn.Linear(NEWS_EMBEDDING_DIM, input_dim) if input_dim != NEWS_EMBEDDING_DIM else nn.Identity()
        self.learn_steps = 0

    def forward(self, x):
        x = self.adaptor(x)
        return self.encoder(x)


class TradingPolicyNetwork(nn.Module):
    """Улучшенная политика с вниманием к новостям"""

    def __init__(self, state_dim=TOTAL_STATE_DIM, action_dim=3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Основная сеть для состояния
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, POLICY_HIDDEN_DIM_1),
            nn.LayerNorm(POLICY_HIDDEN_DIM_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_2),

            nn.Linear(POLICY_HIDDEN_DIM_1, POLICY_HIDDEN_DIM_2),
            nn.LayerNorm(POLICY_HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_2),

            nn.Linear(POLICY_HIDDEN_DIM_2, POLICY_HIDDEN_DIM_3),
            nn.ReLU()
        )

        # Сеть для действий
        self.action_net = nn.Sequential(
            nn.Linear(POLICY_HIDDEN_DIM_3, POLICY_HIDDEN_DIM_4),
            nn.ReLU(),
            nn.Linear(POLICY_HIDDEN_DIM_4, action_dim),
            nn.Softmax(dim=-1)
        )

        # Сеть для оценки состояния
        self.value_net = nn.Sequential(
            nn.Linear(POLICY_HIDDEN_DIM_3, POLICY_HIDDEN_DIM_4),
            nn.ReLU(),
            nn.Linear(POLICY_HIDDEN_DIM_4, 1)
        )

        # Механизм внимания для новостей
        self.news_attention = nn.MultiheadAttention(
            embed_dim=NEWS_ENCODED_DIM,
            num_heads=4,
            dropout=DROPOUT_RATE_1
        )

        # 🔥 НОВАЯ ГОЛОВА ДЛЯ ПРЕДСКАЗАНИЯ ЦЕНЫ
        self.predictor = nn.Sequential(
            nn.Linear(POLICY_HIDDEN_DIM_3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [падение, нейтрально, рост]
        )

    def forward(self, state, news_features=None):
        # 🔥 АКТИВНАЯ КОРРЕКЦИЯ РАЗМЕРНОСТИ
        actual_dim = state.shape[-1]
        expected_dim = self.state_dim

        if actual_dim != expected_dim:
            # Логируем только если это реальное состояние
            if torch.any(state != 0):
                print(f"[TraderModel] ⚠ Размерность состояния: {actual_dim}, ожидалось: {expected_dim}")

            if actual_dim == 150 and expected_dim == 156:
                # 🔥 АКТИВНО добавляем параметры стратегии (balanced)
                strategy_params = torch.tensor(
                    [0.5, 0.5, 1.0, 0.25, 0.025, 0.05],
                    # news_weight, tech_weight, risk_multiplier, hold_time, stop_loss, take_profit
                    device=state.device,
                    dtype=state.dtype
                )

                # Расширяем для batch если нужно
                if state.dim() > 1:
                    strategy_params = strategy_params.expand(state.shape[0], -1)

                state = torch.cat([state, strategy_params], dim=-1)
                print(f"   ⚡ forward добавил стратегию balanced, теперь {state.shape[-1]}")

            elif actual_dim < expected_dim:
                # Стандартное дополнение нулями (для других случаев)
                padding = torch.zeros(
                    *state.shape[:-1],
                    expected_dim - actual_dim,
                    device=state.device,
                    dtype=state.dtype
                )
                state = torch.cat([state, padding], dim=-1)
            else:
                # Обрезаем лишние признаки
                state = state[..., :expected_dim]

        # Обработка состояния
        state_features = self.state_net(state)

        # Если есть новости, применяем внимание
        if news_features is not None and news_features.shape[-1] == NEWS_ENCODED_DIM:
            # Подготовка для multihead attention
            if news_features.dim() == 2:
                news_features = news_features.unsqueeze(0)
            if state_features.dim() == 2:
                state_features_expanded = state_features.unsqueeze(0)
            else:
                state_features_expanded = state_features

            # Применяем внимание
            attended, _ = self.news_attention(
                state_features_expanded,
                news_features,
                news_features
            )

            # Объединяем с оригинальными признаками
            if attended.dim() == 3:
                state_features = state_features + attended.squeeze(0)
            else:
                state_features = state_features + attended

        # Генерация действий и оценки
        action_probs = self.action_net(state_features)
        state_value = self.value_net(state_features)

        # 🔥 ПРЕДСКАЗАНИЕ ДВИЖЕНИЯ ЦЕНЫ
        price_pred = self.predictor(state_features)

        return action_probs, state_value, price_pred

class AdvancedTraderModel:
    """Продвинутая модель трейдера с улучшенной архитектурой"""

    def __init__(self,
                 model_dir: str = "models/saved_trader",
                 learning_rate: float = DEFAULT_LEARNING_RATE,
                 gamma: float = DEFAULT_GAMMA,
                 memory_size: int = DEFAULT_MEMORY_SIZE):

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ✅Загружаем конфигурацию стратегий
        self.strategy_config = self._load_strategy_config()
        self.strategies = self.strategy_config['strategies']

        # Загрузка конфига сериализации памяти
        self.memory_serialization_config = self._load_memory_config()
        print(f"[TraderModel] Конфиг памяти: автосохранение={self.memory_serialization_config['enable_autosave']}")

        # ✅Параметры из конфига
        self.exploration_rate = self.strategy_config['strategy_selection'].get(
            'exploration_rate', DEFAULT_EXPLORATION_RATE
        )
        self.confidence_boost_factor = self.strategy_config['strategy_selection'].get(
            'confidence_boost_factor', DEFAULT_CONFIDENCE_BOOST_FACTOR
        )

        self.strategy_performance = defaultdict(lambda: {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'avg_pnl': 0.0,
            'win_rate': 0.5
        })

        # Используем memory_size из конфига
        self.strategy_memory = deque(
            maxlen=self.strategy_config['strategy_selection'].get(
                'memory_size', STRATEGY_MEMORY_SIZE
            )
        )

        # Загрузка BERT для русского языка
        self.bert_model, self.bert_tokenizer = self._load_bert_model()

        # ✅ АВТОКОРРЕКЦИЯ РАЗМЕРНОСТИ
        global NEWS_EMBEDDING_DIM
        if self.bert_model is not None:
            try:
                # Проверяем реальную размерность BERT
                test_text = ["тест"]
                inputs = self.bert_tokenizer(test_text, return_tensors="pt", padding=True)
                with torch.no_grad():
                    test_output = self.bert_model(**inputs)
                    actual_bert_dim = test_output.last_hidden_state.shape[-1]

                if actual_bert_dim != NEWS_EMBEDDING_DIM:
                    print(f"[TraderModel] ⚠ Автокоррекция: BERT выдает {actual_bert_dim}, "
                          f"меняем NEWS_EMBEDDING_DIM с {NEWS_EMBEDDING_DIM} на {actual_bert_dim}")

                    # Меняем глобальную константу
                    NEWS_EMBEDDING_DIM = actual_bert_dim

                    # Пересоздаем энкодер с правильной размерностью
                    self.news_encoder = NewsEncoder(
                        input_dim=NEWS_EMBEDDING_DIM,
                        hidden_dim=NEWS_ENCODER_HIDDEN_DIM
                    ).to(self.device)

                    print(f"[TraderModel] ✓ Энкодер пересоздан с размерностью {NEWS_EMBEDDING_DIM}")
            except Exception as e:
                print(f"[TraderModel] Ошибка автокоррекции: {e}")
        else:
            print("[TraderModel] BERT не загружен, использую simple_encode_news")

        # ✅ Инициализация сетей (ТОЛЬКО ЕСЛИ ЕЩЁ НЕ СОЗДАНЫ)
        if not hasattr(self, 'news_encoder') or self.news_encoder is None:
            self.news_encoder = NewsEncoder(
                input_dim=NEWS_EMBEDDING_DIM,
                hidden_dim=NEWS_ENCODER_HIDDEN_DIM
            ).to(self.device)  # 👈 to(self.device) сразу при создании

        self.policy_net = TradingPolicyNetwork(
            state_dim=TOTAL_STATE_DIM,
            action_dim=3
        ).to(self.device)  # 👈 to(self.device) сразу при создании

        # ✅ Оптимизаторы
        self.news_optimizer = optim.AdamW(
            self.news_encoder.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )
        self.policy_optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )

        # Память и состояние
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.prioritized_buffer = PrioritizedReplayBuffer(
            max_size=memory_size,
            alpha=0.6,
            beta=0.4
        )
        self.use_prioritized = True  # Флаг использования приоритетного буфера

        # Статистика
        self.error_memory = defaultdict(lambda: {
            'failed_trades': [],
            'avg_loss': 0.0,
            'last_failure': None,
            'failure_count': 0,
            'success_rate': 0.5,
            'total_trades': 0
        })

        self.ticker_stats = defaultdict(lambda: {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'avg_hold_time': 0.0,
            'success_rate': 0.5,
            'last_trade': None
        })

        # Рыночное состояние
        self.market_sentiment = 0.0
        self.sentiment_history = deque(maxlen=MAX_SENTIMENT_HISTORY)
        self.volatility_index = 1.0

        # Загрузка сохраненной модели
        self.load_model()
        self.load_memory()

        # ✅ Загрузка RL конфига
        self.rl_config = self._load_rl_config()

        self.learn_steps = 0
        self.recent_losses = 0
        self.reward_scaling = self.rl_config.get('reward_scaling', 2.0)
        self.price_change_threshold = self.rl_config.get('price_change_threshold', 0.01)
        self.reward_clip_min = self.rl_config.get('reward_clip_min', -100)
        self.reward_clip_max = self.rl_config.get('reward_clip_max', 100)
        self.hold_time_deviation_penalty = self.rl_config.get('hold_time_deviation_penalty', 0.3)
        self.max_hold_time_deviation = self.rl_config.get('max_hold_time_deviation', 0.5)
        self.quick_trade_penalty = self.rl_config.get('quick_trade_penalty', 0.5)
        self.good_profit_bonus = self.rl_config.get('good_profit_bonus', 1.0)
        self.big_loss_penalty = self.rl_config.get('big_loss_penalty', 1.5)
        self.small_profit_bonus = self.rl_config.get('small_profit_bonus', 0.2)
        self.sentiment_positive_threshold = self.rl_config.get('sentiment_positive_threshold', 0.1)
        self.sentiment_negative_threshold = self.rl_config.get('sentiment_negative_threshold', -0.1)
        self.sentiment_positive_bonus_multiplier = self.rl_config.get('sentiment_positive_bonus_multiplier', 0.5)
        self.sentiment_negative_bonus_multiplier = self.rl_config.get('sentiment_negative_bonus_multiplier', 0.3)


        print(f"[TraderModel] Инициализирована на {self.device}")
        print(f"[TraderModel] Статистика: {len(self.error_memory)} тикеров, "
              f"{len(self.memory)} опытов, sentiment={self.market_sentiment:.3f}")
        print(f"[TraderModel] Размерность состояния: {self.policy_net.state_dim}")

    def _load_strategy_config(self, config_path: str = "config/strategies.json") -> Dict:

        """Загрузка конфигурации стратегий"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[TraderModel] Ошибка загрузки конфига стратегий: {e}")
            # Возвращаем дефолтные значения
            return {
                'strategies': {
                    'news_aggressive': {
                        'news_weight': 0.8,
                        'tech_weight': 0.2,
                        'risk_multiplier': 1.5,
                        'target_hold_time_hours': 1,
                        'stop_loss_percent': 2.0,
                        'take_profit_percent': 4.0
                    },
                    'tech_conservative': {
                        'news_weight': 0.2,
                        'tech_weight': 0.8,
                        'risk_multiplier': 0.7,
                        'target_hold_time_hours': 24,
                        'stop_loss_percent': 1.5,
                        'take_profit_percent': 3.0
                    },
                    'balanced': {
                        'news_weight': 0.5,
                        'tech_weight': 0.5,
                        'risk_multiplier': 1.0,
                        'target_hold_time_hours': 6,
                        'stop_loss_percent': 2.5,
                        'take_profit_percent': 5.0
                    },
                    'momentum': {
                        'news_weight': 0.3,
                        'tech_weight': 0.7,
                        'risk_multiplier': 1.2,
                        'target_hold_time_hours': 0.5,
                        'stop_loss_percent': 1.0,
                        'take_profit_percent': 2.0
                    }
                },
                'strategy_selection': {
                    'exploration_rate': DEFAULT_EXPLORATION_RATE,
                    'confidence_boost_factor': DEFAULT_CONFIDENCE_BOOST_FACTOR,
                    'memory_size': STRATEGY_MEMORY_SIZE,
                    'min_trades_for_evaluation': MIN_TRADES_FOR_EVALUATION,
                    'adaptation_rate': 0.1
                }
            }

    def _load_rl_config(self) -> Dict:
        """Загрузка RL конфига"""
        try:
            with open("config/rl_config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[TraderModel] ⚠ Ошибка загрузки RL конфига: {e}")
            return {}


    def _load_memory_config(self) -> Dict:
        """Загрузка конфигурации сериализации памяти"""
        try:
            with open('config/rl_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                memory_config = config.get('memory_serialization', {})

                # Дефолтные значения
                defaults = {
                    'enable_autosave': True,
                    'autosave_interval': 100,
                    'memory_file': 'models/saved_trader/memory_buffer.pkl',
                    'max_memory_to_save': 5000,
                    'compression': True
                }

                # Объединяем с дефолтами
                for key, value in defaults.items():
                    if key not in memory_config:
                        memory_config[key] = value

                return memory_config

        except Exception as e:
            print(f"[TraderModel] Ошибка загрузки конфига памяти: {e}")
            # Возвращаем дефолты
            return {
                'enable_autosave': True,
                'autosave_interval': 100,
                'memory_file': 'models/saved_trader/memory_buffer.pkl',
                'max_memory_to_save': 5000,
                'compression': True
            }

    def load_memory(self):
        """Загрузка сериализованной памяти в оба буфера"""
        config = self.memory_serialization_config
        file_path = config['memory_file']

        if not os.path.exists(file_path):
            print(f"[TraderModel] Файл памяти не найден: {file_path}")
            return

        try:
            import pickle
            import gzip

            if config.get('compression', True):
                with gzip.open(file_path, 'rb') as f:
                    loaded_memory = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    loaded_memory = pickle.load(f)

            # ✅ Загружаем в обычную память
            self.memory.clear()
            self.memory.extend(loaded_memory)
            print(f"[TraderModel] Загружено {len(loaded_memory)} опытов в memory")

            # ✅ ПЕРЕСОЗДАЁМ prioritized_buffer
            if hasattr(self, 'prioritized_buffer'):
                # Создаём НОВЫЙ буфер
                self.prioritized_buffer = PrioritizedReplayBuffer(
                    max_size=self.memory.maxlen,
                    alpha=0.6,
                    beta=0.4
                )

                # Добавляем опыты по одному
                for exp in loaded_memory:
                    self.prioritized_buffer.add(exp, td_error=None)

                print(f"[TraderModel] Загружено {self.prioritized_buffer.size} опытов в prioritized_buffer")

        except Exception as e:
            print(f"[TraderModel] Ошибка загрузки памяти: {e}")
            import traceback
            traceback.print_exc()

    def choose_action_with_strategy(self, state: torch.Tensor, ticker: str,
                                    price: float, market_context: Dict) -> Tuple[int, str, float]:
        """
        Выбор действия с учетом стратегии - модель САМА выбирает стратегию
        """
        strategy_scores = {}

        # ✅ Если состояние уже 156, используем как есть
        if state.shape[-1] == 156:
            base_state = state
        else:
            # Если 150, будем добавлять стратегию для каждой
            base_state = state

        for strategy_name, params in self.strategies.items():
            if state.shape[-1] == 150:
                # Добавляем стратегию к базовому состоянию
                strategy_state = self._create_strategy_state(base_state, params)
            else:
                # Уже полное состояние - используем как есть
                strategy_state = base_state

            with torch.no_grad():
                action_probs, state_value, price_pred = self.policy_net(strategy_state)

            # Модель учится на win_rate стратегий - это обратная связь от реальных результатов!
            perf = self.strategy_performance[strategy_name]
            confidence_boost = perf['win_rate'] * self.confidence_boost_factor

            expected_value = state_value.item() + confidence_boost
            strategy_scores[strategy_name] = {
                'expected_value': expected_value,
                'action_probs': action_probs.cpu().numpy().flatten(),
                'params': params
            }

        # Выбор стратегии - чистая exploitation/exploration, без ручных правил!
        if np.random.random() < self.exploration_rate:
            chosen_strategy = np.random.choice(list(self.strategies.keys()))
        else:
            chosen_strategy = max(strategy_scores.items(),
                                  key=lambda x: x[1]['expected_value'])[0]

        action_probs = strategy_scores[chosen_strategy]['action_probs']

        if np.random.random() < ACTION_EXPLORATION_RATE:
            action = np.random.choice(len(action_probs))
        else:
            action = np.argmax(action_probs)

        confidence = action_probs[action]

        return action, chosen_strategy, confidence

    def _create_strategy_state(self, base_state: torch.Tensor,
                               strategy_params: Dict) -> torch.Tensor:

        # Проверяем размерность
        if base_state.shape[-1] == 156:
            # Уже полное состояние - возвращаем как есть
            return base_state

        # Только если 150 - добавляем параметры
        strategy_state = base_state.clone()

        strategy_params_tensor = torch.tensor([
            float(strategy_params.get('news_weight', 0.5)),
            float(strategy_params.get('tech_weight', 0.5)),
            float(strategy_params.get('risk_multiplier', 1.0)),
            float(strategy_params.get('target_hold_time_hours', 6)) / 24.0,
            float(strategy_params.get('stop_loss_percent', 2.5)) / 100.0,
            float(strategy_params.get('take_profit_percent', 5.0)) / 100.0
        ], dtype=torch.float32, device=self.device)

        strategy_state = torch.cat([strategy_state, strategy_params_tensor])
        return strategy_state

    def record_strategy_outcome(self, strategy_name: str, action: str,
                                pnl: float, hold_time: float):
        """Запись результата стратегии - модель видит, что сработало, а что нет"""

        perf = self.strategy_performance[strategy_name]
        perf['total_trades'] += 1
        perf['total_pnl'] += pnl

        if pnl > 0:
            perf['profitable_trades'] += 1

        # Обновляем статистику - это и есть "обучение" стратегий!
        perf['avg_pnl'] = perf['total_pnl'] / perf['total_trades']
        perf['win_rate'] = perf['profitable_trades'] / perf['total_trades']

        # Сохраняем в memory для дальнейшего обучения
        self.strategy_memory.append({
            'strategy': strategy_name,
            'action': action,
            'pnl': pnl,
            'hold_time': hold_time,
            'win_rate': perf['win_rate'],
            'timestamp': datetime.now().isoformat()
        })

        # Диагностика - модель ВИДИТ, какие стратегии работают
        logger.debug(f"📊 Стратегия {strategy_name}: win_rate={perf['win_rate']:.2%}, "
                     f"avg_pnl={perf['avg_pnl']:.2f}, trades={perf['total_trades']}")

    def _load_bert_model(self):
        """Загрузка BERT модели с обработкой ошибок"""
        try:
            from transformers import AutoTokenizer, AutoModel

            tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
            model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
            model.to(self.device)
            model.eval()

            print("[TraderModel] ✓ Загружена модель RuBERT-tiny2")
            return model, tokenizer

        except Exception as e:
            print(f"[TraderModel] ⚠ Не удалось загрузить BERT: {e}")
            print("[TraderModel] ⚠ Используется упрощенный анализ новостей")
            return None, None

    def save_model(self):
        """Улучшенное сохранение модели"""
        try:
            # Сохраняем веса
            torch.save({
                'news_encoder': self.news_encoder.state_dict(),
                'policy_net': self.policy_net.state_dict(),
                'news_optimizer': self.news_optimizer.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'model_config': {
                    'bert_model': 'cointegrated/rubert-tiny2',
                    'news_input_dim': NEWS_EMBEDDING_DIM,
                    'state_dim': TOTAL_STATE_DIM
                }
            }, os.path.join(self.model_dir, 'model_weights.pth'))

            # Сохраняем состояние
            state = {
                'error_memory': dict(self.error_memory),
                'ticker_stats': dict(self.ticker_stats),
                'market_sentiment': self.market_sentiment,
                'sentiment_history': list(self.sentiment_history),
                'volatility_index': self.volatility_index,
                'memory_size': len(self.memory),
                'total_experiences': sum(len(v['failed_trades']) for v in self.error_memory.values()),
                'strategy_performance': dict(self.strategy_performance),
                'strategy_memory': list(self.strategy_memory),
                'strategies': self.strategies,
                'save_time': datetime.now().isoformat()
            }

            with open(os.path.join(self.model_dir, 'model_state.json'), 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)

            print(f"[TraderModel] Модель сохранена: {len(self.memory)} опытов, "
                  f"{len(self.error_memory)} тикеров, {len(self.strategies)} стратегий")

        except Exception as e:
            print(f"[TraderModel] Ошибка сохранения: {e}")

    def load_model(self):
        """Улучшенная загрузка модели"""
        weights_path = os.path.join(self.model_dir, 'model_weights.pth')
        state_path = os.path.join(self.model_dir, 'model_state.json')

        # Загрузка весов
        if os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location=self.device)

                # Загружаем веса
                self.news_encoder.load_state_dict(checkpoint['news_encoder'])
                self.policy_net.load_state_dict(checkpoint['policy_net'])

                # Загружаем оптимизаторы
                if 'news_optimizer' in checkpoint:
                    self.news_optimizer.load_state_dict(checkpoint['news_optimizer'])
                if 'policy_optimizer' in checkpoint:
                    self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])

                print(f"[TraderModel] ✓ Загружены веса нейросетей")

            except Exception as e:
                print(f"[TraderModel] Ошибка загрузки весов: {e}")

        # Загрузка состояния
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                # Восстанавливаем память ошибок
                self.error_memory.clear()
                for ticker, data in state.get('error_memory', {}).items():
                    self.error_memory[ticker] = data

                # Восстанавливаем статистику
                self.ticker_stats.clear()
                for ticker, stats in state.get('ticker_stats', {}).items():
                    self.ticker_stats[ticker] = stats

                # Восстанавливаем рыночное состояние
                self.market_sentiment = state.get('market_sentiment', 0.0)
                self.sentiment_history = deque(
                    state.get('sentiment_history', []),
                    maxlen=MAX_SENTIMENT_HISTORY
                )
                self.volatility_index = state.get('volatility_index', 1.0)

                # Восстанавливаем стратегии
                if 'strategies' in state:
                    self.strategies = state['strategies']

                # Восстанавливаем статистику стратегий
                if 'strategy_performance' in state:
                    self.strategy_performance.clear()
                    for strategy_name, perf in state.get('strategy_performance', {}).items():
                        self.strategy_performance[strategy_name] = perf

                # Восстанавливаем память стратегий
                if 'strategy_memory' in state:
                    self.strategy_memory = deque(
                        state.get('strategy_memory', []),
                        maxlen=self.strategy_config['strategy_selection'].get(
                            'memory_size', STRATEGY_MEMORY_SIZE
                        )
                    )

                print(f"[TraderModel] ✓ Загружено состояние: {len(self.error_memory)} тикеров, "
                      f"senti={self.market_sentiment:.3f}, vol={self.volatility_index:.2f}, "
                      f"{len(self.strategies)} стратегий")

            except Exception as e:
                print(f"[TraderModel] Ошибка загрузки состояния: {e}")

    def encode_news(self, news_texts: List[str]) -> torch.Tensor:
        """Кодирование новостей через BERT с fallback"""
        if not news_texts:
            return torch.zeros(1, NEWS_ENCODED_DIM).to(self.device)

        # Пробуем BERT
        if self.bert_model is not None and self.bert_tokenizer is not None:
            try:
                # Токенизация
                inputs = self.bert_tokenizer(
                    news_texts,
                    padding=True,
                    truncation=True,
                    max_length=NEWS_ENCODED_DIM,
                    return_tensors="pt"
                ).to(self.device)

                # Получение эмбеддингов
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

                # Кодирование через энкодер
                self.news_encoder.eval()
                with torch.no_grad():
                    news_features = self.news_encoder(embeddings)

                return news_features

            except Exception as e:
                print(f"[TraderModel] Ошибка BERT кодирования: {e}")

        # Fallback: упрощенное кодирование
        return self.simple_encode_news(news_texts)

    def simple_encode_news(self, news_texts: List[str]) -> torch.Tensor:
        """Упрощенное кодирование новостей"""
        batch_size = len(news_texts)
        embeddings = torch.zeros(batch_size, NEWS_EMBEDDING_DIM).to(self.device)

        sentiment_dict = {
            'positive': ['рост', 'прибыль', 'увеличение', 'дивиденд', 'выше', 'улучшение', 'рекомендуют', 'покупать'],
            'negative': ['падение', 'убыток', 'снижение', 'проблемы', 'ниже', 'сокращение', 'продавать', 'снижают'],
            'market': ['рынок', 'акци', 'бирж', 'фондов', 'инвест', 'торг', 'ликвид', 'волатиль']
        }

        for i, text in enumerate(news_texts):
            text_lower = text.lower()

            # Базовые признаки
            MAX_TEXT_LENGTH = 1000
            MAX_WORDS = 200
            length = min(len(text) / MAX_TEXT_LENGTH, 1.0)
            words = text_lower.split()
            word_count = min(len(words) / MAX_WORDS, 1.0)
            unique_ratio = len(set(words)) / max(len(words), 1)

            # Сентимент признаки
            SENTIMENT_DIVISOR = 10
            pos_score = sum(1 for w in sentiment_dict['positive'] if w in text_lower) / SENTIMENT_DIVISOR
            neg_score = sum(1 for w in sentiment_dict['negative'] if w in text_lower) / SENTIMENT_DIVISOR
            market_score = sum(1 for w in sentiment_dict['market'] if w in text_lower) / SENTIMENT_DIVISOR

            # Нормализация
            pos_score = min(pos_score, 1.0)
            neg_score = min(neg_score, 1.0)
            market_score = min(market_score, 1.0)

            # Создание эмбеддинга
            embedding = torch.zeros(NEWS_EMBEDDING_DIM).to(self.device)

            # Заполняем первые признаки
            embedding[0] = length
            embedding[1] = word_count
            embedding[2] = unique_ratio
            embedding[3] = pos_score
            embedding[4] = neg_score
            embedding[5] = market_score
            embedding[6] = pos_score - neg_score  # чистый сентимент
            embedding[7] = (pos_score + neg_score) / 2  # интенсивность
            embedding[8] = market_score
            embedding[9] = length * word_count  # сложность текста

            # Добавляем немного случайности для разнообразия
            RANDOM_FEATURES_START = 10
            RANDOM_FEATURES_COUNT = 10
            RANDOM_NOISE_SCALE = 0.1
            embedding[RANDOM_FEATURES_START:RANDOM_FEATURES_START + RANDOM_FEATURES_COUNT] = \
                torch.randn(RANDOM_FEATURES_COUNT).to(self.device) * RANDOM_NOISE_SCALE

            embeddings[i] = embedding

        # Пропускаем через энкодер
        self.news_encoder.eval()
        with torch.no_grad():
            news_features = self.news_encoder(embeddings)

        return news_features

    def calculate_risk_score(self, ticker: str, price: float, sentiment: float) -> float:
        """Расчет риск-скора с учетом волатильности"""
        error_data = self.error_memory[ticker]
        stats = self.ticker_stats[ticker]

        # Параметры расчета риска
        FAILURE_PENALTY_RATE = 0.08
        MAX_FAILURE_PENALTY = 0.8
        LOSS_PENALTY_MULTIPLIER = 1.5
        MAX_LOSS_PENALTY = 0.6
        SUCCESS_BONUS_RATE = 0.3
        BASE_SENTIMENT_FACTOR = 1.2
        VOLATILITY_MULTIPLIER = 0.4
        BASE_VOLATILITY_FACTOR = 0.8

        # Базовый риск
        if error_data['failure_count'] == 0:
            base_risk = RISK_BASE
        else:
            failure_penalty = min(error_data['failure_count'] * FAILURE_PENALTY_RATE, MAX_FAILURE_PENALTY)
            loss_penalty = min(abs(error_data['avg_loss']) * LOSS_PENALTY_MULTIPLIER, MAX_LOSS_PENALTY)
            success_bonus = max(stats['success_rate'] - 0.5, 0) * SUCCESS_BONUS_RATE

            base_risk = RISK_BASE + failure_penalty + loss_penalty - success_bonus

        # Корректировка на сентимент
        sentiment_factor = BASE_SENTIMENT_FACTOR - abs(sentiment)

        # Корректировка на волатильность рынка
        volatility_factor = BASE_VOLATILITY_FACTOR + (self.volatility_index * VOLATILITY_MULTIPLIER)

        final_risk = base_risk * sentiment_factor * volatility_factor

        # Ограничение диапазона
        return max(MIN_RISK, min(MAX_RISK, final_risk))

    def build_state_vector(self,
                           ticker: str,
                           price: float,
                           momentum: float,
                           sentiment: float,
                           news_features: torch.Tensor,
                           market_data: Dict,
                           market_sentiment: float = 0.0) -> torch.Tensor:

        global logger

        # 🔥 КОРРЕКЦИЯ В САМОМ НАЧАЛЕ
        # Сохраняем оригинальные данные для диагностики
        original_ticker = ticker
        original_price = price

        """Построение вектора состояния с признаками саморегуляции"""

        # Признаки из новостей
        if news_features.numel() > 0 and news_features.shape[1] == NEWS_ENCODED_DIM:
            news_vec = news_features.mean(dim=0).cpu().numpy()
        else:
            news_vec = np.zeros(NEWS_ENCODED_DIM)

        # Статистика тикера
        stats = self.ticker_stats[ticker]
        success_rate = stats['success_rate']
        total_trades = stats['total_trades']
        avg_hold_time = stats['avg_hold_time']

        # Расчет риска
        risk_score = self.calculate_risk_score(ticker, price, sentiment)

        # Базовые признаки
        features = [
            # Ценовые признаки
            price / PRICE_NORMALIZATION,
            momentum * MOMENTUM_SCALING,
            sentiment,

            # Риск и статистика
            risk_score,
            success_rate,
            min(total_trades / MAX_TRADES_FOR_EXPERIENCE, 2.0),
            min(avg_hold_time / 24.0, MAX_HOLD_TIME_DAYS),

            # Данные рынка
            market_data.get('volume', 0) / VOLUME_NORMALIZATION,
            market_data.get('spread', 0.01) * 100,
            market_data.get('liquidity', 0.5),
            market_data.get('rsi', 50) / RSI_NORMALIZATION,
            market_data.get('volatility', 0.1) * VOLATILITY_SCALING,

            # Технические признаки
            market_data.get('sma_10_ratio', 1.0),
            market_data.get('sma_20_ratio', 1.0),
            market_data.get('bb_position', 0.5),
            market_data.get('volume_ratio', 1.0),

            # Временные признаки
            datetime.now().hour / 24.0,
            datetime.now().weekday() / 7.0,

            # Дополнительные
            market_data.get('market_cap', 0) / 1e12,
            market_data.get('pe_ratio', 15) / PE_NORMALIZATION,

            # Рыночные признаки
            self.market_sentiment,
            market_sentiment,
            self.volatility_index,
        ]

        # Новостные признаки
        features.extend(news_vec.tolist())

        # ПРИЗНАКИ САМОРЕГУЛЯЦИИ
        positions_norm = 0.0
        if hasattr(self, 'portfolio') and hasattr(self.portfolio, 'positions'):
            total_positions = len(self.portfolio.positions)
            positions_norm = min(total_positions / 10.0, 1.0)

        trades_per_hour_norm = 0.0
        if hasattr(self, 'portfolio') and hasattr(self.portfolio, 'trade_history'):
            last_hour = time.time() - 3600
            recent_trades = 0
            for t in self.portfolio.trade_history[-100:]:
                if isinstance(t, dict) and 'timestamp' in t:
                    try:
                        t_time = datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')).timestamp()
                        if t_time > last_hour:
                            recent_trades += 1
                    except:
                        pass
            trades_per_hour_norm = min(recent_trades / 20.0, 1.0)

        exposure_norm = 0.0
        if hasattr(self, 'portfolio') and hasattr(self.portfolio, 'initial_capital'):
            positions_value = sum(p['qty'] * p.get('avg_price', 0) for p in self.portfolio.positions.values())
            exposure_ratio = positions_value / self.portfolio.initial_capital
            exposure_norm = min(exposure_ratio, 1.0)

        features.extend([positions_norm, trades_per_hour_norm, exposure_norm])

        time_pressure = 0.0
        if hasattr(self, 'scheduler'):
            time_to_close = self.scheduler.get_time_to_next_session()
            if time_to_close:
                hours_left = time_to_close[0] + time_to_close[1] / 60
                time_pressure = max(0, 1.0 - hours_left / 6.0)

        features.append(time_pressure)


        # Преобразование в тензор
        state_vector = torch.FloatTensor(features).to(self.device)

        # Проверка размерности - должно быть BASE_STATE_DIM (150)
        if state_vector.shape[0] != BASE_STATE_DIM:
            if state_vector.shape[0] < BASE_STATE_DIM:
                padding = torch.zeros(BASE_STATE_DIM - state_vector.shape[0]).to(self.device)
                state_vector = torch.cat([state_vector, padding])
            else:
                state_vector = state_vector[:BASE_STATE_DIM]

        return state_vector

    def choose_action(self,
                      state: torch.Tensor,
                      ticker: str,
                      current_price: float,
                      market_context: Dict = None) -> Tuple[int, float, float]:
        """
        Выбор действия с динамической exploration rate
        Автоматически добавляет параметры стратегии если нужно (150 → 156)
        """
        global logger

        # Режим с выбором стратегии (передан market_context)
        if market_context is not None:
            # Если состояние 150 - добавляем стратегию из контекста
            if state.shape[-1] == 150:
                strategy_name = market_context.get('current_strategy', 'balanced')
                strategy_params = self.strategies.get(strategy_name, self.strategies['balanced'])
                state = self._create_strategy_state(state, strategy_params)
                print(f"   ⚡ choose_action добавил стратегию {strategy_name}, теперь {state.shape[-1]}")

            # Вызываем выбор действия со стратегией
            action, strategy, confidence = self.choose_action_with_strategy(
                state, ticker, current_price, market_context
            )

            # Получаем оценку состояния для TD-error
            with torch.no_grad():
                _, state_value, _ = self.policy_net(state.unsqueeze(0))

            return action, confidence, state_value.item()

        # Режим без стратегии (простой выбор действия)
        self.policy_net.eval()

        with torch.no_grad():
            action_probs, state_value, price_pred = self.policy_net(state.unsqueeze(0))

        probs = action_probs.cpu().numpy().flatten()

        # Корректировка на основе истории ошибок
        error_data = self.error_memory[ticker]
        stats = self.ticker_stats[ticker]

        FAILURE_FACTOR_RATE = 0.2
        MAX_FAILURE_FACTOR = 0.6
        SUCCESS_FACTOR_RATE = 0.4
        POOR_PERFORMANCE_PENALTY = 0.3

        if error_data['failure_count'] > 1:
            failure_factor = min(error_data['failure_count'] * FAILURE_FACTOR_RATE, MAX_FAILURE_FACTOR)
            probs[0] *= (1.0 - failure_factor)  # BUY
            probs[2] *= (1.0 + failure_factor)  # SELL
            probs = probs / probs.sum()

        MIN_TRADES_FOR_ADJUSTMENT = 5
        SUCCESS_RATE_THRESHOLD_HIGH = 0.6
        SUCCESS_RATE_THRESHOLD_LOW = 0.4

        if stats['total_trades'] > MIN_TRADES_FOR_ADJUSTMENT:
            success_factor = max(stats['success_rate'] - 0.5, 0) * SUCCESS_FACTOR_RATE

            if stats['success_rate'] > SUCCESS_RATE_THRESHOLD_HIGH:
                probs[0] *= (1.0 + success_factor)  # BUY
            elif stats['success_rate'] < SUCCESS_RATE_THRESHOLD_LOW:
                probs[2] *= (1.0 + POOR_PERFORMANCE_PENALTY)  # SELL

            probs = probs / probs.sum()

        # 🔥 ДИНАМИЧЕСКАЯ EXPLORATION RATE
        base_rate = 0.3

        # Фактор количества позиций
        position_factor = 1.0
        if hasattr(self, 'portfolio') and hasattr(self.portfolio, 'positions'):
            position_factor = max(0, 1.0 - len(self.portfolio.positions) / 10.0)

        # Фактор недавних убытков
        loss_factor = 1.0
        if hasattr(self, 'recent_losses') and self.recent_losses > 2:
            loss_factor = 0.5

        # Фактор времени до закрытия
        time_factor = 1.0
        if hasattr(self, 'scheduler'):
            time_to_close = self.scheduler.get_time_to_next_session()
            if time_to_close and time_to_close[0] < 1:  # меньше часа до закрытия
                time_factor = 0.3

        # Decay на основе накопленного опыта
        memory_factor = max(0.1, 1.0 - len(self.memory) / 5000)

        exploration_rate = base_rate * position_factor * loss_factor * time_factor * memory_factor
        exploration_rate = max(0.05, min(0.5, exploration_rate))

        # Выбор действия
        if np.random.random() < exploration_rate:
            action = np.random.choice(len(probs))
        else:
            action = np.argmax(probs)

        confidence = probs[action]

        return action, confidence, state_value.item()

    def get_state_value(self, state: torch.Tensor) -> float:
        try:
            # ✅ Если состояние 150, добавляем стратегию по умолчанию
            if state.shape[-1] == 150:
                strategy_params = self.strategies['balanced']
                state = self._create_strategy_state(state, strategy_params)
                print(f"   ⚡ get_state_value добавил стратегию, теперь {state.shape[-1]}")

            self.policy_net.eval()
            with torch.no_grad():
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                _, value, _ = self.policy_net(state)
                return value.item()
        except Exception as e:
            print(f"[TraderModel] Ошибка get_state_value: {e}")
            return 0.0

    def remember_experience(self,
                            state: torch.Tensor,
                            action: int,
                            reward: float,
                            next_state: torch.Tensor,
                            done: bool,
                            news_features: Optional[torch.Tensor] = None,
                            td_error: Optional[float] = None,
                            sentiment_data=None,
                            pnl_rub: float = 0.0):

        print(f"\n🔥🔥🔥 remember_experience FIRED! 🔥🔥🔥")
        print(f"   state.shape: {state.shape}")

        if reward < self.reward_clip_min or reward > self.reward_clip_max:
            logger.debug(f"Clipping reward from {reward} to [{self.reward_clip_min}, {self.reward_clip_max}]")
            reward = max(self.reward_clip_min, min(self.reward_clip_max, reward))

        experience = {
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done,
            'news_features': news_features.cpu() if news_features is not None else None,
            'sentiment_data': sentiment_data,  # ✅ СОХРАНЯЕМ НОВОСТНОЙ САНТИМЕНТ
            'timestamp': datetime.now().isoformat()
        }

        # ✅ 1. ВСЕГДА добавляем в обычную память
        self.memory.append(experience)
        print(f"   ✅ memory size: {len(self.memory)}")

        # ✅ 2. ВСЕГДА добавляем в prioritized_buffer (если он существует)
        if hasattr(self, 'prioritized_buffer'):
            self.prioritized_buffer.add(experience, td_error)
            logger.debug(f"   ✅ prioritized_buffer size: {self.prioritized_buffer.size}")
            print(f"   ✅ prioritized_buffer size: {self.prioritized_buffer.size}")
        else:
            logger.debug(f"   ⚠️ prioritized_buffer не найден")
            print(f"   ⚠️ prioritized_buffer не найден")

        # Автосохранение
        if (self.memory_serialization_config['enable_autosave'] and
                len(self.memory) % self.memory_serialization_config['autosave_interval'] == 0):
            self.save_memory()

    def learn_from_experience(self, batch_size: int = DEFAULT_BATCH_SIZE):
        """Обучение на опыте с предсказанием цены"""
        if len(self.memory) < max(batch_size * 2, MIN_EXPERIENCES_FOR_LEARNING):
            return None

        try:
            # Выбор батча
            actual_batch_size = min(batch_size, len(self.memory) // 2)
            if actual_batch_size < 4:
                return None

            indices = np.random.choice(len(self.memory), actual_batch_size, replace=False)
            batch = [self.memory[i] for i in indices]

            # Подготовка данных
            try:
                states = torch.stack([exp['state'] for exp in batch]).to(self.device)
                actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
                rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
                next_states = torch.stack([exp['next_state'] for exp in batch]).to(self.device)
                dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)
            except Exception as e:
                print(f"[TraderModel] Ошибка подготовки батча: {e}")
                return None

            # Подготовка новостей
            news_features = None
            has_valid_news = False
            if batch[0].get('news_features') is not None:
                try:
                    news_features = torch.stack([exp['news_features'] for exp in batch]).to(self.device)
                    if news_features.shape[-1] == NEWS_ENCODED_DIM:
                        has_valid_news = True
                except:
                    pass

            # 🔥 ПОЛУЧАЕМ РЕАЛЬНЫЕ ИЗМЕНЕНИЯ ЦЕН
            price_changes = []
            for i, exp in enumerate(batch):
                if 'next_price' in exp:
                    price_change = (exp['next_price'] - exp['price']) / exp['price']
                else:
                    # Если нет сохранённой цены, используем reward как прокси
                    price_change = (exp['reward'] / self.reward_scaling)  # приблизительно

                # Преобразуем в класс: 0=падение, 1=нейтрально, 2=рост
                if price_change < -self.price_change_threshold:
                    price_class = 0  # падение
                elif price_change > self.price_change_threshold:
                    price_class = 2  # рост
                else:
                    price_class = 1  # нейтрально
                price_changes.append(price_class)
                # pnl = exp['reward'] / self.reward_scaling
                pnl_value = exp['reward'] / self.reward_scaling
                logger.debug(f"PnL in batch: {pnl_value:.4f}")

            price_targets = torch.LongTensor(price_changes).to(self.device)

            # Обучение
            self.policy_net.train()

            # Прямой проход
            try:
                if has_valid_news:
                    current_probs, current_values, price_pred = self.policy_net(states, news_features)
                else:
                    current_probs, current_values, price_pred = self.policy_net(states)
            except Exception as e:
                print(f"[TraderModel] Ошибка прямого прохода: {e}")
                self.policy_net.eval()
                return None

            # Следующие оценки
            with torch.no_grad():
                if has_valid_news:
                    _, next_values, _ = self.policy_net(next_states, news_features)
                else:
                    _, next_values, _ = self.policy_net(next_states)

            # Целевые значения
            target_values = rewards + (1 - dones) * self.gamma * next_values

            # Value loss
            value_loss = nn.SmoothL1Loss()(current_values, target_values.detach())

            # Policy loss
            dist = torch.distributions.Categorical(current_probs)
            log_probs = dist.log_prob(actions)
            advantages = (target_values - current_values).detach()
            policy_loss = -(log_probs * advantages).mean()

            # Entropy
            entropy = dist.entropy().mean()
            entropy_bonus = ENTROPY_BONUS_COEFF * entropy

            # 🔥 PRICE PREDICTION LOSS
            price_loss = nn.CrossEntropyLoss()(price_pred, price_targets)

            # Коэффициент для предсказания (можно настраивать)
            PRICE_PREDICTION_WEIGHT = 0.1

            # Общий loss
            total_loss = value_loss + policy_loss - entropy_bonus + PRICE_PREDICTION_WEIGHT * price_loss

            # Оптимизация
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP_VALUE)
            self.policy_optimizer.step()

            self.policy_net.eval()

            # Логирование accuracy предсказаний
            with torch.no_grad():
                predictions = price_pred.argmax(dim=1)
                accuracy = (predictions == price_targets).float().mean().item()
                if self.learn_steps % 10 == 0:
                    print(f"[TraderModel] Price prediction accuracy: {accuracy:.2%}")

            return total_loss.item()

        except Exception as e:
            print(f"[TraderModel] Критическая ошибка обучения: {e}")
            import traceback
            print(traceback.format_exc())
            self.policy_net.eval()
            return None

    def learn_from_prioritized(self, batch_size: int = DEFAULT_BATCH_SIZE):
        """Обучение с приоритетной выборкой"""
        if not hasattr(self, 'prioritized_buffer') or self.prioritized_buffer is None:
            return self.learn_from_experience(batch_size)

        # ✅ Проверяем размер prioritized_buffer
        if self.prioritized_buffer.size < batch_size:
            print(
                f"[TraderModel] Недостаточно опытов в prioritized_buffer: {self.prioritized_buffer.size} < {batch_size}")
            return None

        # Приоритетная выборка
        batch, indices, weights = self.prioritized_buffer.sample(batch_size)

        try:
            # Подготовка данных
            states = torch.stack([exp['state'] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
            next_states = torch.stack([exp['next_state'] for exp in batch]).to(self.device)
            dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)
            weights_tensor = torch.FloatTensor(weights).to(self.device)

            # Новости (если есть)
            news_features = None
            if batch[0].get('news_features') is not None:
                try:
                    news_features = torch.stack([exp['news_features'] for exp in batch]).to(self.device)
                except:
                    pass

            # Обучение
            self.policy_net.train()

            # Прямые проходы
            if news_features is not None:
                current_probs, current_values, price_pred = self.policy_net(states, news_features)
                with torch.no_grad():
                    _, next_values = self.policy_net(next_states, news_features)
            else:
                current_probs, current_values, price_pred = self.policy_net(states)
                with torch.no_grad():
                    _, next_values, _ = self.policy_net(next_states)

            # Целевые значения
            target_values = rewards + (1 - dones) * self.gamma * next_values

            # Value loss с весами
            value_loss = (weights_tensor * nn.SmoothL1Loss(reduction='none')(
                current_values, target_values.detach())).mean()

            # Policy loss
            dist = torch.distributions.Categorical(current_probs)
            log_probs = dist.log_prob(actions)
            advantages = (target_values - current_values).detach()
            policy_loss = -(weights_tensor * log_probs * advantages).mean()

            # Entropy
            entropy = dist.entropy().mean()
            entropy_bonus = ENTROPY_BONUS_COEFF * entropy

            total_loss = value_loss + policy_loss - entropy_bonus

            # Оптимизация
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP_VALUE)
            self.policy_optimizer.step()

            self.policy_net.eval()

            # ✅ ИСПРАВЛЕНО: добавляем .squeeze()
            with torch.no_grad():
                # Вычисляем разницу
                td_errors = (target_values - current_values).detach().cpu().numpy()

                # ГАРАНТИРОВАННО превращаем в одномерный массив чисел
                td_errors = np.asarray(td_errors, dtype=np.float32).flatten()

                # Проверка на всякий случай
                print(f"td_errors shape: {td_errors.shape}, dtype: {td_errors.dtype}")
                print(f"First few values: {td_errors[:5]}")

            # Обновляем приоритеты
            self.prioritized_buffer.update_priorities(indices, td_errors)

            return total_loss.item()

        except Exception as e:
            print(f"[TraderModel] Ошибка приоритетного обучения: {e}")
            import traceback
            traceback.print_exc()
            self.policy_net.eval()
            return None



    def learn_from_experience_custom(self, states, actions, rewards, next_states, dones, news_features=None):
        """Обучение на готовом батче с ЗАЩИТОЙ ОТ ОШИБОК"""
        try:
            # ✅ ПРОВЕРКА РАЗМЕРНОСТЕЙ
            batch_size = states.shape[0]
            if batch_size != actions.shape[0] or batch_size != rewards.shape[0]:
                print(f"[TraderModel] Несоответствие размеров батча: "
                      f"states={states.shape[0]}, actions={actions.shape[0]}, rewards={rewards.shape[0]}")
                return None

            # ✅ ПРОВЕРКА НОВОСТЕЙ
            has_news = False
            if news_features is not None:
                if news_features.shape[0] == batch_size:
                    has_news = True
                else:
                    print(f"[TraderModel] Несоответствие новостей: "
                          f"states={batch_size}, news={news_features.shape[0]}")
                    news_features = None

            # Переключаем в режим обучения
            self.policy_net.train()

            # Текущие оценки
            if has_news:
                current_probs, current_values, price_pred = self.policy_net(states, news_features)
            else:
                current_probs, current_values, price_pred = self.policy_net(states)

            # Следующие оценки (без градиентов)
            with torch.no_grad():
                if has_news:
                    _, next_values, _ = self.policy_net(next_states, news_features)
                else:
                    _, next_values, _ = self.policy_net(next_states)

            # Целевые значения
            target_values = rewards + (1 - dones) * self.gamma * next_values

            # Value loss
            value_loss = nn.SmoothL1Loss()(current_values, target_values.detach())

            # Policy loss
            dist = torch.distributions.Categorical(current_probs)
            log_probs = dist.log_prob(actions)

            advantages = (target_values - current_values).detach()
            policy_loss = -(log_probs * advantages).mean()

            # Entropy regularization
            entropy = dist.entropy().mean()
            entropy_bonus = ENTROPY_BONUS_COEFF * entropy

            # Общий loss
            total_loss = value_loss + policy_loss - entropy_bonus

            # Оптимизация
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP_VALUE)
            self.policy_optimizer.step()

            # Возвращаем в eval режим
            self.policy_net.eval()

            return total_loss.item()

        except Exception as e:
            print(f"[TraderModel] Ошибка кастомного обучения: {e}")
            import traceback
            print(traceback.format_exc())
            # ✅ ВАЖНО: ВСЕГДА ВОЗВРАЩАЕМСЯ В EVAL РЕЖИМ ПРИ ОШИБКЕ
            self.policy_net.eval()
            return None


    def _train_news_encoder(self, batch):
        """Обучение энкодера новостей"""
        try:
            # Простая задача реконструкции
            news_features = torch.stack([exp['news_features'] for exp in batch
                                         if exp['news_features'] is not None]).to(self.device)

            MIN_BATCH_FOR_ENCODER_TRAINING = 4
            if len(news_features) < MIN_BATCH_FOR_ENCODER_TRAINING:
                return

            self.news_encoder.train()

            # Прямой проход
            encoded = self.news_encoder(news_features)

            # Декодер (простой линейный слой)
            decoder = nn.Linear(NEWS_ENCODED_DIM, news_features.shape[1]).to(self.device)

            # Реконструкция
            reconstructed = decoder(encoded)

            # Loss реконструкции
            recon_loss = nn.MSELoss()(reconstructed, news_features)

            # Оптимизация
            self.news_optimizer.zero_grad()
            recon_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.news_encoder.parameters(), 1.0)
            self.news_optimizer.step()

            self.news_encoder.eval()

        except Exception as e:
            print(f"[TraderModel] Ошибка обучения энкодера: {e}")

    def update_market_sentiment(self, sentiment_score: float):
        """Обновление рыночного настроения"""
        # Экспоненциальное сглаживание
        self.market_sentiment = (1 - SENTIMENT_SMOOTHING_ALPHA) * self.market_sentiment + \
                                SENTIMENT_SMOOTHING_ALPHA * sentiment_score

        # Сохранение в историю
        self.sentiment_history.append(self.market_sentiment)

        # Обновление индекса волатильности на основе колебаний сентимента
        MIN_HISTORY_FOR_VOLATILITY = 10
        if len(self.sentiment_history) > MIN_HISTORY_FOR_VOLATILITY:
            recent_sentiments = list(self.sentiment_history)[-MIN_HISTORY_FOR_VOLATILITY:]
            volatility = np.std(recent_sentiments)
            self.volatility_index = BASE_VOLATILITY + volatility * VOLATILITY_SCALING_FACTOR

    def record_trade_outcome(self,
                             ticker: str,
                             action: str,
                             entry_price: float,
                             exit_price: float,
                             hold_time: float,
                             news_sentiment: float,
                             market_conditions: Dict,
                             strategy: str = None,
                             market_sentiment: float = 0.0) -> Tuple[float, float]:

        """Запись результата сделки с УНИВЕРСАЛЬНЫМ PnL РАСЧЕТОМ"""

        # ✅ УНИВЕРСАЛЬНЫЙ PnL РАСЧЕТ ДЛЯ ЛЮБОЙ ТОРГОВОЙ ЛОГИКИ
        if entry_price > 0 and exit_price > 0:
            price_change_ratio = (exit_price - entry_price) / entry_price

            if action == 'BUY':
                pnl = 0.0
                trade_return = 0.0
            elif action == 'SELL':
                pnl = price_change_ratio
                trade_return = price_change_ratio
            else:
                pnl = 0.0
                trade_return = 0.0
        else:
            pnl = 0.0
            trade_return = 0.0

        # ✅ ОБНОВЛЕНИЕ СТАТИСТИКИ ТИКЕРА
        stats = self.ticker_stats[ticker]
        stats['total_trades'] += 1

        if action in ['SELL', 'CLOSE']:
            stats['total_pnl'] += pnl
            if pnl > 0:
                stats['profitable_trades'] += 1
            if stats['total_trades'] == 1:
                stats['avg_hold_time'] = hold_time
            else:
                stats['avg_hold_time'] = (stats['avg_hold_time'] * (stats['total_trades'] - 1) + hold_time) / stats[
                    'total_trades']

        # ✅ РАСЧЕТ УСПЕШНОСТИ
        if stats['total_trades'] > 0:
            stats['success_rate'] = stats['profitable_trades'] / stats['total_trades']
        else:
            stats['success_rate'] = 0.5

        stats['last_trade'] = datetime.now().isoformat()

        # ✅ ОБНОВЛЕНИЕ СТАТИСТИКИ СТРАТЕГИИ
        if strategy and action in ['SELL', 'CLOSE']:
            self.record_strategy_outcome(strategy, action, pnl, hold_time)

        # ✅ ЗАПИСЬ ОШИБОК
        if pnl < LOSS_THRESHOLD and action in ['SELL', 'CLOSE']:
            error_data = self.error_memory[ticker]
            trade_record = {
                'date': datetime.now().isoformat(),
                'action': action,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'hold_time': hold_time,
                'sentiment': news_sentiment,
                'market_conditions': market_conditions,
                'strategy': strategy
            }
            error_data['failed_trades'].append(trade_record)
            error_data['failure_count'] += 1
            error_data['last_failure'] = datetime.now().isoformat()
            if error_data['failed_trades']:
                losses = [t['pnl'] for t in error_data['failed_trades']]
                error_data['avg_loss'] = sum(losses) / len(losses)
            error_data['success_rate'] = stats['success_rate']
            error_data['total_trades'] = stats['total_trades']
            print(f"[TraderModel] ⚠ Запомнена ошибка: {ticker} {action}, убыток {pnl:.2%}, стратегия: {strategy}")

        # ✅ РАСЧЕТ НАГРАДЫ - ИСПОЛЬЗУЕМ ПЕРЕМЕННЫЕ ИЗ КОНФИГА
        reward = pnl

        if action in ['SELL', 'CLOSE']:
            # Бонус за правильную реакцию на рыночное настроение
            sentiment_alignment_bonus = 0.0
            if pnl > 0 and market_sentiment > self.sentiment_positive_threshold:
                sentiment_alignment_bonus = market_sentiment * self.sentiment_positive_bonus_multiplier
            elif pnl < 0 and market_sentiment < self.sentiment_negative_threshold:
                sentiment_alignment_bonus = abs(market_sentiment) * self.sentiment_negative_bonus_multiplier
            reward += sentiment_alignment_bonus

            # Учитываем стратегию в награде
            if strategy:
                target_hold_time = self.strategies.get(strategy, {}).get('target_hold_time_hours', 6)
                hold_time_diff = abs(hold_time - target_hold_time)
                if hold_time_diff > target_hold_time * self.max_hold_time_deviation:
                    reward -= self.hold_time_deviation_penalty * hold_time_diff / target_hold_time

            # Бонусы/штрафы
            if hold_time < MIN_HOLD_TIME and abs(pnl) < SMALL_PROFIT:
                reward -= self.quick_trade_penalty
            elif pnl > PROFIT_THRESHOLD:
                reward += self.good_profit_bonus
            elif pnl < LOSS_THRESHOLD:
                reward -= self.big_loss_penalty
            elif pnl > 0:
                reward += self.small_profit_bonus
        else:
            reward = 0.0

        return reward, price_change_ratio

    def rank_candidates(self,
                        prices: Dict[str, float],
                        securities: Dict[str, Dict],
                        ticker_sentiment: Dict[str, float],
                        news_by_ticker: Dict[str, List[str]]) -> List[Tuple[str, float]]:
        """Ранжирование кандидатов с улучшенной логикой"""

        candidates = []

        # Параметры для расчета скора
        MAX_NEWS_PER_TICKER = 3
        CONFIDENCE_WEIGHT_BUY = 0.5
        SENTIMENT_WEIGHT_BUY = 0.3
        SUCCESS_BONUS_RATE = 0.4
        RISK_PENALTY_RATE = 0.3
        VOLATILITY_PENALTY_RATE = 0.5
        MARKET_SENTIMENT_WEIGHT = 0.2
        MOMENTUM_WEIGHT_BUY = 0.2

        CONFIDENCE_WEIGHT_SELL = 0.4
        SENTIMENT_WEIGHT_SELL = 0.3
        FAILURE_PENALTY_RATE = 0.15
        MAX_FAILURE_PENALTY = 0.6
        LOSS_BONUS_RATE = 0.3
        MOMENTUM_WEIGHT_SELL = 0.3

        for ticker, price in prices.items():
            if ticker not in securities:
                continue

            market_data = securities[ticker]
            momentum = market_data.get('momentum', 0.0)
            sentiment = ticker_sentiment.get(ticker, self.market_sentiment)

            # Новости
            ticker_news = news_by_ticker.get(ticker, [])
            if ticker_news:
                news_features = self.encode_news(ticker_news[:MAX_NEWS_PER_TICKER])
            else:
                news_features = torch.zeros(1, NEWS_ENCODED_DIM).to(self.device)

            # Построение состояния
            state = self.build_state_vector(
                ticker=ticker,
                price=price,
                momentum=momentum,
                sentiment=sentiment,
                news_features=news_features,
                market_data=market_data,
            )

            # Выбор действия
            try:
                # Передаём пустой market_context, чтобы choose_action не ждал стратегию
                action, confidence, _ = self.choose_action(state, ticker, price, market_context={})
            except Exception as e:
                print(f"[TraderModel] Ошибка choose_action для {ticker}: {e}")
                action, confidence = 1, 0.5

            # Оценка кандидата
            stats = self.ticker_stats[ticker]
            error_data = self.error_memory[ticker]

            if action == 0:  # BUY
                # Факторы для покупки
                success_bonus = max(stats['success_rate'] - 0.5, 0) * SUCCESS_BONUS_RATE
                risk_score = self.calculate_risk_score(ticker, price, sentiment)
                risk_penalty = risk_score * RISK_PENALTY_RATE

                # Награда за низкую волатильность
                volatility_penalty = market_data.get('volatility', 0.1) * VOLATILITY_PENALTY_RATE

                score = (confidence * CONFIDENCE_WEIGHT_BUY +
                         sentiment * SENTIMENT_WEIGHT_BUY +
                         success_bonus -
                         risk_penalty -
                         volatility_penalty +
                         self.market_sentiment * MARKET_SENTIMENT_WEIGHT +
                         momentum * MOMENTUM_WEIGHT_BUY)

            elif action == 2:  # SELL
                # Факторы для продажи
                failure_penalty = min(error_data['failure_count'] * FAILURE_PENALTY_RATE, MAX_FAILURE_PENALTY)

                # Если позиция в убытке - выше приоритет продажи
                current_return = stats.get('current_return', 0.0)
                loss_bonus = max(-current_return * LOSS_BONUS_RATE, 0) if current_return < 0 else 0

                score = (confidence * CONFIDENCE_WEIGHT_SELL -
                         sentiment * SENTIMENT_WEIGHT_SELL +
                         failure_penalty +
                         loss_bonus +
                         (-momentum * MOMENTUM_WEIGHT_SELL))

            else:  # HOLD
                score = 0.0

            candidates.append((ticker, score))

        # Сортировка
        candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"[TraderModel] Отранжировано {len(candidates)} кандидатов")

        return candidates[:TOP_CANDIDATES_LIMIT]

    def get_worst_position(self,
                           positions: Dict[str, Dict],
                           prices: Dict[str, float]) -> Tuple[Optional[str], int]:
        """Определение худшей позиции"""
        if not positions:
            return None, 0

        # Параметры для расчета
        FAILURE_PENALTY_RATE = 0.2
        MAX_HOLD_DAYS = 60
        HOLD_PENALTY_RATE = 0.3
        MIN_SELL_PERCENTAGE = 0.25
        MAX_SELL_PERCENTAGE = 0.50
        MIN_QUANTITY = 1

        worst_score = float('inf')
        worst_ticker = None

        for ticker, pos_data in positions.items():
            if ticker not in prices:
                continue

            current_price = prices[ticker]
            avg_price = pos_data['avg_price']

            if avg_price > 0:
                pnl_ratio = (current_price - avg_price) / avg_price
            else:
                pnl_ratio = 0.0

            # История ошибок
            error_data = self.error_memory[ticker]
            failure_penalty = error_data['failure_count'] * FAILURE_PENALTY_RATE

            # Время удержания
            hold_time_days = pos_data.get('hold_time_days', 0)
            hold_penalty = min(hold_time_days / MAX_HOLD_DAYS, 1.0) * HOLD_PENALTY_RATE

            # Общий score (чем ниже, тем хуже)
            score = pnl_ratio - failure_penalty - hold_penalty

            if score < worst_score:
                worst_score = score
                worst_ticker = ticker

        if worst_ticker:
            pos = positions[worst_ticker]
            # Продаем от 25% до 50% позиции в зависимости от "плохости"
            severity = min((0.5 - worst_score) / 0.5, 1.0)  # Нормализация
            sell_percentage = MIN_SELL_PERCENTAGE + severity * (MAX_SELL_PERCENTAGE - MIN_SELL_PERCENTAGE)

            qty = int(pos['qty'] * sell_percentage)
            qty = max(qty, MIN_QUANTITY)

            return worst_ticker, qty

        return None, 0


    def save_memory(self):
        """Сериализация памяти"""
        config = self.memory_serialization_config

        if not config['enable_autosave'] or len(self.memory) == 0:
            return

        try:
            import pickle
            import gzip

            # Определяем сколько опытов сохранять
            max_to_save = min(config['max_memory_to_save'], len(self.memory))
            memory_to_save = list(self.memory)[-max_to_save:]

            file_path = config['memory_file']

            # Создаем директорию если нет
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            if config.get('compression', True):
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(memory_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(memory_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"[TraderModel] Память сохранена: {len(memory_to_save)} опытов в {file_path}")

        except Exception as e:
            print(f"[TraderModel] Ошибка сохранения памяти: {e}")

    def periodic_learning(self):
        """Периодическое обучение"""
        if len(self.memory) > MIN_EXPERIENCES_FOR_LEARNING:
            loss = self.learn_from_experience(batch_size=DEFAULT_BATCH_SIZE)

            # Сохранение через интервалы
            if not hasattr(self, '_learn_steps'):
                self._learn_steps = 0

            self._learn_steps += 1

            if self._learn_steps % AUTO_SAVE_INTERVAL == 0:
                self.save_model()
                print(f"[TraderModel] Автосохранение после {self._learn_steps} шагов обучения")

            # Сериализация памяти
            if self.memory_serialization_config['enable_autosave']:
                self.save_memory()

            return loss

        return None

    def get_model_stats(self) -> Dict:
        """Получение статистики модели"""
        return {
            'memory_size': len(self.memory),
            'ticker_stats_count': len(self.ticker_stats),
            'error_memory_count': len(self.error_memory),
            'market_sentiment': self.market_sentiment,
            'volatility_index': self.volatility_index,
            'device': str(self.device),
            'last_learn_steps': getattr(self, '_learn_steps', 0)
        }


# Глобальный экземпляр
trader_model_instance = AdvancedTraderModel()