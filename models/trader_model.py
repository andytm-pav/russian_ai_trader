"""
Модернизированная модель трейдера с улучшенной архитектурой
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# ========== КОНСТАНТЫ МОДЕЛИ ==========
# Архитектурные параметры
NEWS_EMBEDDING_DIM = 768
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

    def forward(self, state, news_features=None):
        # Проверяем размерность состояния
        if state.shape[-1] != self.state_dim:
            # Автоматическая корректировка размерности
            if state.shape[-1] < self.state_dim:
                padding = torch.zeros(
                    *state.shape[:-1],
                    self.state_dim - state.shape[-1]
                ).to(state.device)
                state = torch.cat([state, padding], dim=-1)
            else:
                state = state[..., :self.state_dim]

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

        return action_probs, state_value


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

        # Инициализация сетей с правильными размерностями
        self.news_encoder = NewsEncoder(
            input_dim=NEWS_EMBEDDING_DIM,
            hidden_dim=NEWS_ENCODER_HIDDEN_DIM
        )
        self.news_encoder.to(self.device)

        self.policy_net = TradingPolicyNetwork(
            state_dim=TOTAL_STATE_DIM,
            action_dim=3
        )
        self.policy_net.to(self.device)

        # Загрузка BERT для русского языка
        self.bert_model, self.bert_tokenizer = self._load_bert_model()

        # Оптимизаторы
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

    def choose_action_with_strategy(self, state: torch.Tensor, ticker: str,
                                    price: float, market_context: Dict) -> Tuple[int, str, float]:
        """
        Выбор действия с учетом стратегии
        """
        # 1. Оцениваем пригодность каждой стратегии для текущего состояния
        strategy_scores = {}

        for strategy_name, params in self.strategies.items():
            # Модифицируем состояние под стратегию
            strategy_state = self._create_strategy_state(state, params)

            # Получаем предсказание
            with torch.no_grad():
                action_probs, state_value = self.policy_net(strategy_state)

            # ✅ ИСПРАВЛЯЕМ: Используем конфигурируемый confidence_boost_factor
            perf = self.strategy_performance[strategy_name]
            confidence_boost = perf['win_rate'] * self.confidence_boost_factor

            # Ожидаемая ценность стратегии
            expected_value = state_value.item() + confidence_boost

            strategy_scores[strategy_name] = {
                'expected_value': expected_value,
                'action_probs': action_probs.cpu().numpy().flatten(),
                'params': params
            }

        # 2. Выбор стратегии (epsilon-greedy)
        if np.random.random() < self.exploration_rate:
            # Exploration: случайная стратегия
            chosen_strategy = np.random.choice(list(self.strategies.keys()))
        else:
            # Exploitation: лучшая стратегия
            chosen_strategy = max(strategy_scores.items(),
                                  key=lambda x: x[1]['expected_value'])[0]

        # 3. Выбор действия для выбранной стратегии
        action_probs = strategy_scores[chosen_strategy]['action_probs']

        if np.random.random() < ACTION_EXPLORATION_RATE and hasattr(self, 'memory') and len(self.memory) < 2000:
            action = np.random.choice(len(action_probs))
        else:
            action = np.argmax(action_probs)

        confidence = action_probs[action]

        return action, chosen_strategy, confidence

    def _create_strategy_state(self, base_state: torch.Tensor,
                               strategy_params: Dict) -> torch.Tensor:
        """Создание состояния для конкретной стратегии"""
        # Клонируем базовое состояние
        strategy_state = base_state.clone()

        # Параметры стратегии
        strategy_params_tensor = torch.tensor([
            strategy_params['news_weight'],
            strategy_params['tech_weight'],
            strategy_params['risk_multiplier'],
            strategy_params.get('target_hold_time_hours', 6) / 24.0,  # Нормализованное время
            strategy_params.get('stop_loss_percent', 2.5) / 100.0,  # Нормализованный стоп-лосс
            strategy_params.get('take_profit_percent', 5.0) / 100.0  # Нормализованный тейк-профит
        ]).to(self.device)

        # Объединяем с основным состоянием
        strategy_state = torch.cat([strategy_state, strategy_params_tensor])

        return strategy_state

    def record_strategy_outcome(self, strategy_name: str, action: str,
                                pnl: float, hold_time: float):
        """Запись результата стратегии"""
        perf = self.strategy_performance[strategy_name]
        perf['total_trades'] += 1
        perf['total_pnl'] += pnl

        if pnl > 0:
            perf['profitable_trades'] += 1

        perf['avg_pnl'] = perf['total_pnl'] / perf['total_trades']
        perf['win_rate'] = perf['profitable_trades'] / perf['total_trades']

        # Сохраняем в memory для обучения
        self.strategy_memory.append({
            'strategy': strategy_name,
            'action': action,
            'pnl': pnl,
            'hold_time': hold_time,
            'timestamp': datetime.now().isoformat()
        })

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
                           market_data: Dict) -> torch.Tensor:
        """Построение вектора состояния"""

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

            # Рыночные признаки
            self.market_sentiment,
            self.volatility_index,

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
        ]

        # Новостные признаки
        features.extend(news_vec.tolist())

        # Преобразование в тензор
        state_vector = torch.FloatTensor(features).to(self.device)

        # Проверка размерности - должно быть BASE_STATE_DIM
        if state_vector.shape[0] != BASE_STATE_DIM:
            # Автоматическая корректировка
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
        """Выбор действия с улучшенной логикой"""
        if market_context is not None:
            action, strategy, confidence = self.choose_action_with_strategy(
                state, ticker, current_price, market_context
            )
            return action, confidence, 0.0  # Возвращаем совместимый формат

        self.policy_net.eval()

        with torch.no_grad():
            # Получаем вероятности действий
            action_probs, state_value = self.policy_net(state.unsqueeze(0))

        probs = action_probs.cpu().numpy().flatten()

        # Корректировка на основе истории
        error_data = self.error_memory[ticker]
        stats = self.ticker_stats[ticker]

        # Параметры корректировки
        FAILURE_FACTOR_RATE = 0.2
        MAX_FAILURE_FACTOR = 0.6
        SUCCESS_FACTOR_RATE = 0.4
        POOR_PERFORMANCE_PENALTY = 0.3

        if error_data['failure_count'] > 1:
            # Корректировка для проблемных тикеров
            failure_factor = min(error_data['failure_count'] * FAILURE_FACTOR_RATE, MAX_FAILURE_FACTOR)

            # Снижаем вероятность покупки, повышаем продажи
            probs[0] *= (1.0 - failure_factor)  # BUY
            probs[2] *= (1.0 + failure_factor)  # SELL

            # Нормализация
            probs = probs / probs.sum()

        # Корректировка на основе успешности
        MIN_TRADES_FOR_ADJUSTMENT = 5
        SUCCESS_RATE_THRESHOLD_HIGH = 0.6
        SUCCESS_RATE_THRESHOLD_LOW = 0.4

        if stats['total_trades'] > MIN_TRADES_FOR_ADJUSTMENT:
            success_factor = max(stats['success_rate'] - 0.5, 0) * SUCCESS_FACTOR_RATE

            if stats['success_rate'] > SUCCESS_RATE_THRESHOLD_HIGH:
                # Успешные тикеры - больше покупок
                probs[0] *= (1.0 + success_factor)  # BUY
            elif stats['success_rate'] < SUCCESS_RATE_THRESHOLD_LOW:
                # Неуспешные тикеры - больше продаж
                probs[2] *= (1.0 + POOR_PERFORMANCE_PENALTY)  # SELL

            probs = probs / probs.sum()

        # Выбор действия (с эксплорацией на ранних этапах)
        BASE_EXPLORATION_RATE = 0.3
        MIN_EXPLORATION_RATE = 0.1
        EXPLORATION_DECAY_EXPERIENCES = 10000
        EXPLORATION_DECAY_RATE = 0.2

        exploration_rate = max(
            MIN_EXPLORATION_RATE,
            BASE_EXPLORATION_RATE - (len(self.memory) / EXPLORATION_DECAY_EXPERIENCES) * EXPLORATION_DECAY_RATE
        )

        MIN_EXPLORATION_MEMORY = 3000

        if np.random.random() < exploration_rate and len(self.memory) < MIN_EXPLORATION_MEMORY:
            # Случайное действие для исследования
            action = np.random.choice(len(probs))
        else:
            # Жадное действие
            action = np.argmax(probs)

        confidence = probs[action]

        return action, confidence, state_value.item()

    def remember_experience(self,
                            state: torch.Tensor,
                            action: int,
                            reward: float,
                            next_state: torch.Tensor,
                            done: bool,
                            news_features: Optional[torch.Tensor] = None):
        """Сохранение опыта с новостями"""
        self.memory.append({
            'state': state.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu(),
            'done': done,
            'news_features': news_features.cpu() if news_features is not None else None,
            'timestamp': datetime.now().isoformat()
        })

    def learn_from_experience(self, batch_size: int = DEFAULT_BATCH_SIZE):
        """Обучение на опыте с улучшенной стабильностью"""
        if len(self.memory) < batch_size * 2:
            return None

        try:
            # Выборка с приоритетом (простейшая реализация)
            recent_size = min(len(self.memory) // 4, batch_size // 2)
            recent_indices = list(range(len(self.memory) - recent_size, len(self.memory)))
            random_indices = np.random.choice(len(self.memory) - recent_size,
                                              batch_size - recent_size,
                                              replace=False)
            indices = list(random_indices) + recent_indices

            batch = [self.memory[i] for i in indices]

            # Подготовка данных
            states = torch.stack([exp['state'] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
            next_states = torch.stack([exp['next_state'] for exp in batch]).to(self.device)
            dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)

            # Новостные признаки (если есть)
            news_features_list = []
            for exp in batch:
                if exp['news_features'] is not None:
                    news_features_list.append(exp['news_features'])

            if news_features_list:
                news_features = torch.stack(news_features_list).to(self.device)
            else:
                news_features = None

            # Переключаем в режим обучения
            self.policy_net.train()

            # Текущие оценки
            if news_features is not None:
                current_probs, current_values = self.policy_net(states, news_features)
            else:
                current_probs, current_values = self.policy_net(states)

            # Следующие оценки (без градиентов)
            with torch.no_grad():
                if news_features is not None:
                    _, next_values = self.policy_net(next_states, news_features)
                else:
                    _, next_values = self.policy_net(next_states)

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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), GRADIENT_CLIP_VALUE)

            self.policy_optimizer.step()

            # Периодическое обучение энкодера новостей
            ENCODER_TRAIN_INTERVAL = 100
            if len(self.memory) % ENCODER_TRAIN_INTERVAL == 0 and news_features is not None:
                self._train_news_encoder(batch)

            return total_loss.item()

        except Exception as e:
            print(f"[TraderModel] Ошибка обучения: {e}")
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
                             strategy: str = None) -> Tuple[float, float]:
        """Запись результата сделки"""

        # Расчет PnL
        if action == 'BUY':
            # Для покупки PnL рассчитывается при продаже
            pnl = 0.0
        elif action == 'SELL':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = 0.0

        # Обновление статистики тикера
        stats = self.ticker_stats[ticker]
        stats['total_trades'] += 1

        if action == 'SELL':
            stats['total_pnl'] += pnl

            if pnl > 0:
                stats['profitable_trades'] += 1

            # Обновление среднего времени удержания
            if stats['total_trades'] == 1:
                stats['avg_hold_time'] = hold_time
            else:
                stats['avg_hold_time'] = (stats['avg_hold_time'] * (stats['total_trades'] - 1) + hold_time) / stats[
                    'total_trades']

        stats['success_rate'] = stats['profitable_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0.5
        stats['last_trade'] = datetime.now().isoformat()

        # Обновление статистики стратегии если указана
        if strategy and action == 'SELL':
            self.record_strategy_outcome(strategy, action, pnl, hold_time)

        # Параметры для определения значительных убытков
        SIGNIFICANT_LOSS_THRESHOLD = -0.02  # -2%

        # Запись ошибок (значительные убытки)
        if pnl < SIGNIFICANT_LOSS_THRESHOLD:
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

            # Обновление среднего убытка
            if error_data['failed_trades']:
                losses = [t['pnl'] for t in error_data['failed_trades']]
                error_data['avg_loss'] = sum(losses) / len(losses)

            # Обновление success rate в error memory
            error_data['success_rate'] = stats['success_rate']
            error_data['total_trades'] = stats['total_trades']

            print(f"[TraderModel] Запомнена ошибка: {ticker} {action}, убыток {pnl:.2%}, стратегия: {strategy}")

        # Параметры наград
        REWARD_SCALING = 20.0
        HOLD_TIME_DEVIATION_PENALTY = 0.3
        MAX_HOLD_TIME_DEVIATION = 0.5
        QUICK_TRADE_PENALTY = 0.5
        GOOD_PROFIT_BONUS = 1.0
        BIG_LOSS_PENALTY = 1.5
        SMALL_PROFIT_BONUS = 0.2

        # Награда для обучения
        if action == 'SELL':
            reward = pnl * REWARD_SCALING

            # Учитываем стратегию в награде
            if strategy:
                # Проверяем соответствует ли hold_time целевой стратегии
                target_hold_time = self.strategies.get(strategy, {}).get('target_hold_time_hours', 6)
                hold_time_diff = abs(hold_time - target_hold_time)

                if hold_time_diff > target_hold_time * MAX_HOLD_TIME_DEVIATION:
                    reward -= HOLD_TIME_DEVIATION_PENALTY * hold_time_diff / target_hold_time

            # Бонусы/штрафы
            if hold_time < MIN_HOLD_TIME and abs(pnl) < SMALL_PROFIT:
                reward -= QUICK_TRADE_PENALTY
            elif pnl > PROFIT_THRESHOLD:
                reward += GOOD_PROFIT_BONUS
            elif pnl < LOSS_THRESHOLD:
                reward -= BIG_LOSS_PENALTY
            elif pnl > 0:
                reward += SMALL_PROFIT_BONUS
        else:
            reward = 0.0

        return reward, pnl

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
                market_data=market_data
            )

            # Выбор действия
            try:
                action, confidence, _ = self.choose_action(state, ticker, price)
            except Exception as e:
                print(f"[TraderModel] Ошибка choose_action для {ticker}: {e}")
                action, confidence = 1, 0.5  # HOLD по умолчанию

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