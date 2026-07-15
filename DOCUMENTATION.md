# Документация по изменениям в системе russian_ai_trader

## Содержание

1. [Обзор внедрённых исследований](#1-обзор-внедрённых-исследований)
2. [Новые файлы (созданные)](#2-новые-файлы-созданные)
3. [Изменённые файлы](#3-изменённые-файлы)
4. [Конфигурационные изменения](#4-конфигурационные-изменения)
5. [Подробное описание каждого модуля](#5-подробное-описание-каждого-модуля)
6. [Что ожидается от внедрения](#6-что-ожидается-от-внедрения)
7. [Как перенести изменения](#7-как-перенести-изменения)
8. [Известные ограничения и следующие шаги](#8-известные-ограничения-и-следующие-шаги)

---

## 1. Обзор внедрённых исследований

На основе 6 итераций аналитических исследований индекса Мосбиржи (IMOEX)
внедрено 12 компонентов. Все настройки вынесены в конфигурационные файлы —
хардкод в логике минимален (только внутренние алгоритмические константы).

| # | Компонент | Источник (исследование) | Статус |
|---|-----------|------------------------|--------|
| 1 | ML-тональность ruBERT-tiny2 | Анализ тональности новостей | ✅ Уже было |
| 2 | Прокси-стакан (микроструктура) | Анализ микроструктуры MOEX | ✅ Внедрено |
| 3 | Imbalance как ранний сигнал | Анализ микроструктуры | ✅ Внедрено |
| 4 | Каскадный предсказатель 5д→1м | Теория хаоса (Ляпунов) | ✅ Внедрено |
| 5 | History loader 3 мес + инкремент | Подготовка к прогнозированию | ✅ Внедрено |
| 6 | CNYRUB в macro data | Корреляционный анализ IMOEX | ✅ Внедрено |
| 7 | Херст H live (R/S анализ) | Теория хаоса (персистентность) | ✅ Внедрено |
| 8 | Ляпунов (max_hold=120ч=5д) | Теория хаоса (горизонт) | ✅ Через конфиг |
| 9 | D₂ фрактальная размерность | Теория хаоса (аттрактор) | ✅ Внедрено |
| 10 | Процесс Хокса | Анализ событий (hit rate 80%) | ✅ Внедрено |
| 11 | LLM-коуч отключён | Анализ архитектуры | ✅ Отключён |
| 12 | Weekend session из конфига | Тестирование | ✅ Исправлено |

---

## 2. Новые файлы (созданные)

### 2.1. `core/hawkes_signal.py` (175 строк)

**Назначение:** Процесс Хокса с экспоненциальным ядром для прогнозирования
кластеризации движений цены.

**Класс:** `HawkesSignalGenerator`

**Что делает:**
- Отслеживает цены по тикерам и обнаруживает события (|log-return| > threshold)
- Разделяет события на bullish (рост) и bearish (падение)
- Обучает процесс Хокса через EM-алгоритм: оценивает μ (фоновая интенсивность),
  η (branching ratio — сила кластеризации), β (скорость затухания)
- Rolling re-fit: переобучение каждые `refit_interval` циклов (по умолчанию 500)
- Прогноз: E[bullish events] и E[bearish events] на горизонт `forecast_horizon_hours`
- Возвращает net_signal = bull_expected − bear_expected для каскадного предсказателя

**Конфигурация** (секция `hawkes` в `config/settings.json`):
```json
{
  "hawkes": {
    "event_threshold_pct": 0.01,    // порог события: 1% движения
    "window_size": 4000,            // размер окна истории
    "refit_interval": 500,          // переобучение каждые 500 циклов
    "forecast_horizon_hours": 48,   // горизонт прогноза: 2 дня
    "max_iter": 50                  // max итераций EM-алгоритма
  }
}
```

**Интеграция:**
- В `smart_broker.py`: импортируется как `hawkes_signal` (синглтон)
- В `_run_cycle_impl()`: обновляет цены через `hawkes.update_price()`,
  переобучает через `hawkes.fit()` для тикеров в портфеле
- В `check_stops_and_tp()`: передаёт `hawkes_signal` в каскадный предсказатель

---

### 2.2. `fetchers/microstructure_fetcher.py` (251 строка)

**Назначение:** Прокси-стакан из доступных полей MOEX marketdata + генерация
микроструктурных сигналов.

**Класс:** `MicrostructureFetcher`

**Что делает:**
- Получает из MOEX: BID, OFFER, SPREAD, BIDDEPTH, OFFERDEPTH, NUMBIDS, NUMOFFERS
- Рассчитывает 6 микроструктурных признаков:
  1. `spread_pct` — спред в % от цены
  2. `imbalance` — дисбаланс объёмов (BIDDEPTH − OFFERDEPTH) / (BIDDEPTH + OFFERDEPTH)
  3. `order_imbalance` — дисбаланс числа заявок
  4. `bid_vol_relative` — объём bid относительно скользящего среднего
  5. `offer_vol_relative` — объём offer относительно скользящего среднего
  6. `microstructure_regime` — режим: 0=сбалансированный, 1=покупательский, 2=продавецкий
- Генерирует сигналы:
  - imbalance > buy_threshold + spread < max_spread → BUY
  - imbalance < sell_threshold + spread < max_spread → SELL
  - spread > low_liquidity_spread → HOLD (низкая ликвидность)

**Конфигурация** (секция `microstructure` в `config/settings.json`):
```json
{
  "microstructure": {
    "enabled": true,
    "max_tickers_per_cycle": 20,
    "imbalance_buy_threshold": 0.3,
    "imbalance_sell_threshold": -0.3,
    "max_spread_pct_for_signal": 0.2,
    "low_liquidity_spread_pct": 0.5,
    "regime_balanced_imbalance": 0.15,
    "regime_balanced_spread_pct": 0.2
  }
}
```

**Интеграция:**
- В `smart_broker.py`: импортируется как `microstructure_fetcher` (синглтон)
- Метод `_generate_microstructure_signals()` генерирует сигналы в каждом цикле
- Метод `get_microstructure()` вызывается в `check_stops_and_tp()` для каскада
- Метод `get_microstructure_features_vector()` готовит 6 признаков для state vector

---

### 2.3. `fetchers/history_loader.py` (500 строк)

**Назначение:** Загрузка исторических данных MOEX за 3 месяца при первом старте,
инкрементальная дозагрузка при последующих, расчёт хаос-метрик.

**Класс:** `HistoryLoader`

**Что делает:**
- **При первом старте:** загружает часовые свечи за 3 месяца для 50+ тикеров
  и 10 индексов (IMOEX, MOEXOG, MOEXFN, MOEXMM и др.)
- **При рестарте:** проверяет последнюю дату в `price_history_extended.json`,
  дозагружает только недостающие часы
- Сохраняет историю в два файла:
  - `data/price_history_extended.json` — полная история (до 2000 точек на тикер)
  - `data/price_history.json` — последние 100 точек для `TechnicalTraderCore`
- **Расчёт хаос-метрик** (кэшируется в `data/chaos_metrics_cache.json`):
  - **Херст H** через R/S анализ (персистентность: H > 0.5 — трендовый)
  - **D₂** через Grassberger-Procaccia (фрактальная размерность: D₂ < 3 — детерминированный хаос)
  - Волатильность, momentum 24ч

**Конфигурация** (секция `history_loader` в `config/settings.json`):
```json
{
  "history_loader": {
    "months_back": 3,
    "max_points_per_ticker": 2000,
    "rate_limit_seconds": 0.15,
    "top_tickers_limit": 50,
    "history_file": "data/price_history_extended.json",
    "chaos_cache_file": "data/chaos_metrics_cache.json"
  }
}
```

**Результаты (проверено live):**
| Тикер | Херст H | D₂ | Волатильность |
|-------|---------|-----|---------------|
| IMOEX | 0.639 | — | 0.284% |
| GMKN | 0.598 | — | 0.554% |
| PLZL | 0.596 | — | 0.923% |
| SBER | 0.532 | — | 0.294% |

Все H > 0.5 — рынок персистентный (подтверждает наше исследование H=0.577).

---

### 2.4. `models/price_predictor.py` (342 строки)

**Назначение:** Каскадный предсказатель цены с адаптивным горизонтом.
После покупки тикера прогнозирует цену на уменьшающихся горизонтах и продаёт
перед предсказанным снижением.

**Класс:** `PricePredictionCascade`

**Логика каскада:**
| Уровень | Условие | Горизонт | Порог продажи | Источники |
|---------|---------|----------|---------------|-----------|
| 0 | PnL ≥ 0 | 5 дней (120ч) | P(снижение) > 50% | RL + Хокс + Херст |
| 1 | PnL < 0, hold < 48ч | 2 дня (48ч) | P(снижение) > 55% | RL + momentum + RSI |
| 2 | PnL < 0, hold < 120ч | 1 день (24ч) | P(снижение) > 60% | momentum + RSI + BB |
| 3 | PnL < 0, hold ≥ 120ч | 1 час | P(снижение) > 65% | микроструктура + momentum |
| 4 | PnL < 0, hold ≥ 168ч | 15 мин | P(снижение) > 70% | микроструктура + spread |
| 5 | PnL < 0, hold ≥ 204ч | 1 мин | P(снижение) > 75% | tick momentum + spread |

**Ключевые правила:**
- **Максимум удержания — 5 дней (120 часов)** — по Ляпунову (λ=0.0085, T=5.3 дня)
- Каскад **только углубляется** (не откатывается назад)
- Работает **только для тикеров в портфеле** (в `check_stops_and_tp`)
- Веса источников меняются по горизонту: длинные → RL+Хокс, короткие → микроструктура

**Источники прогноза:**
1. RL-модель predictor (3 класса: down/sideways/up)
2. Технические индикаторы (RSI, BB, momentum)
3. Микроструктура (imbalance, spread)
4. Хокс-сигнал (ожидаемое число событий)
5. Херст (персистентность: тренд продолжится или развернётся)

**Конфигурация** (секция `price_predictor` в `config/settings.json`):
```json
{
  "price_predictor": {
    "enabled": true,
    "max_hold_hours": 120,
    "cascade": {
      "level_0_horizon_hours": 120,
      "level_0_name": "5_days",
      "level_0_sell_threshold": 0.50,
      "level_0_max_hold_hours": 120,
      ...
      "level_5_horizon_hours": 0.0167,
      "level_5_name": "1_min",
      "level_5_sell_threshold": 0.75,
      "level_5_max_hold_hours": 120
    },
    "weights_long": {"rl": 0.40, "tech": 0.20, "ms": 0.10, "hawkes": 0.15, "hurst": 0.15},
    "weights_medium": {"rl": 0.30, "tech": 0.30, "ms": 0.15, "hawkes": 0.10, "hurst": 0.15},
    "weights_short": {"rl": 0.20, "tech": 0.35, "ms": 0.30, "hawkes": 0.05, "hurst": 0.10},
    "weights_ultra_short": {"rl": 0.10, "tech": 0.20, "ms": 0.55, "hawkes": 0.05, "hurst": 0.10}
  }
}
```

**Авто-восстановление:**
Если файл `models/price_predictor.py` удалён, `smart_broker.py` создаёт
встроенный `_FallbackPricePredictor` с минимальной логикой (возвращает HOLD).
Система продолжает работать без сбоев.

---

## 3. Изменённые файлы

### 3.1. `main.py` (−13 строк)
- **Убран блок инициализации LLM-коуча** (12 строк удалено)
- Коуч заменён на каскадный предсказатель + микроструктуру + Хокс

### 3.2. `models/smart_broker.py` (+211 строк)

**Импорты (строки 19-55):**
- `microstructure_fetcher` — синглтон прокси-стакана
- `HistoryLoader` — загрузчик истории
- `price_predictor` — синглтон каскада (с fallback при отсутствии файла)
- `hawkes_signal` — синглтон процесса Хокса (с fallback при отсутствии)

**__init__ (строки 116-155):**
- Инициализация `self.microstructure`
- Инициализация `self.price_predictor`
- Инициализация `self.history_loader` (с передачей `moex_fetcher` и `technical_core`)
- Инициализация `self.hawkes`
- LLM-коуч: `self.coach = None` (принудительно отключён)

**_initialize_components (строки 1708-1721):**
- Загрузка 3 месяцев истории при первом старте
- Инкрементальная дозагрузка при рестарте
- Расчёт хаос-метрик (Херст, D₂)

**_generate_microstructure_signals (строки 337-370):**
- Новый метод: генерация сигналов из прокси-стакана
- Пороги из конфига: `imbalance_buy_threshold`, `imbalance_sell_threshold`
- Ограничение `max_tickers_per_cycle` из конфига

**_run_cycle_impl (строки 1967-1973):**
- Добавлен вызов `_generate_microstructure_signals()` в цикл генерации сигналов
- Добавлено обновление Хокса: `hawkes.update_price()` для всех тикеров
- Переобучение Хокса: `hawkes.fit()` для тикеров в портфеле каждые 500 циклов

**Фильтр ликвидности (строки 1946-1954):**
- Адаптивный порог объёма для выходных дней
- Параметры из конфига: `weekend_volume_divisor`, `weekend_trades_divisor`, `weekend_min_trades`
- Минимум тикеров: `min_tickers_after_filter` из конфига (по умолчанию 10)

**check_stops_and_tp (строки 2470-2532):**
- Добавлен параметр `securities` для передачи в микроструктуру
- **Каскадный предсказатель** после проверки TP:
  - Определяет уровень каскада по PnL и hold_time
  - Получает Хокс-сигнал, Херст, D₂, микроструктуру
  - Прогнозирует P(снижение)
  - Если P(снижение) > порог → продаёт

**_periodic_learning (строка 1288):**
- LLM-коуч блок заменён комментарием (отключён)

### 3.3. `fetchers/moex_fetcher.py` (+77 строк)

**get_all_securities (строки 291-342):**
- Расширен `marketdata.columns`: добавлены `MARKETPRICE`, `LCURRENTPRICE`, `BID`, `OFFER`
- **Fallback цены** (критический багфикс):
  - Было: `LAST` → если None, price = 0.0
  - Стало: `LAST` → `MARKETPRICE` → `LCURRENTPRICE` → midspread `(BID+OFFER)/2`
- Добавлены поля `bid` и `offer` в итоговый словарь securities

**_get_currency_rate (строки 1021-1049) — новый метод:**
- Универсальный метод получения курса любой валюты с MOEX SELT
- Используется для CNYRUB

**get_macro_data (строки 797-828):**
- Добавлен `cny_rub` — курс юаня к рублю (наш анализ: β=−0.15, значим)
- Значение получается через `self._get_currency_rate('CNYRUB_TOM')`

### 3.4. `core/trading_hours_scheduler.py` (+14 строк)

**can_trade_now (строки 306-324):**
- **Багфикс:** `weekend_session` не был в списке разрешённых периодов для ордеров
- Теперь читается из конфига `market_schedule.json` → `weekend_sessions.allow_order_placement`
- Если `true` → `weekend_session` добавляется в `allowed_periods`
- Все три флага (`can_place_orders`, `can_cancel_orders`, `can_modify_orders`) учитывают этот флаг

### 3.5. `Tools_pre_train.py` (+11 строк)
- Добавлен импорт `HistoryLoader` и `hawkes_signal`
- В `market_data` (внутри `train_on_period`) добавлен `cny_rub`
- В `create_base_state` добавлен `cny_rub`

---

## 4. Конфигурационные изменения

### 4.1. `config/settings.json` (+122 строки)

**Новые секции:**

```json
{
  "liquidity_filter": {
    "min_tickers_after_filter": 10,
    "weekend_volume_divisor": 5,
    "weekend_trades_divisor": 3,
    "weekend_min_trades": 30
  },

  "microstructure": {
    "enabled": true,
    "max_tickers_per_cycle": 20,
    "imbalance_buy_threshold": 0.3,
    "imbalance_sell_threshold": -0.3,
    "max_spread_pct_for_signal": 0.2,
    "low_liquidity_spread_pct": 0.5,
    "regime_balanced_imbalance": 0.15,
    "regime_balanced_spread_pct": 0.2
  },

  "price_predictor": {
    "enabled": true,
    "max_hold_hours": 120,
    "cascade": { ... 6 уровней ... },
    "weights_long": { ... },
    "weights_medium": { ... },
    "weights_short": { ... },
    "weights_ultra_short": { ... },
    "tech_rsi_overbought": 70,
    "tech_rsi_oversold": 30,
    ...
  },

  "history_loader": {
    "months_back": 3,
    "max_points_per_ticker": 2000,
    "rate_limit_seconds": 0.15,
    "top_tickers_limit": 50,
    "history_file": "data/price_history_extended.json",
    "chaos_cache_file": "data/chaos_metrics_cache.json"
  },

  "hawkes": {
    "event_threshold_pct": 0.01,
    "window_size": 4000,
    "refit_interval": 500,
    "forecast_horizon_hours": 48,
    "max_iter": 50
  }
}
```

### 4.2. `config/rl_config.json`
- `llm_coach.enabled`: `true` → `false` (коуч отключён)

### 4.3. `config/market_schedule.json`
- `weekend_sessions.allow_order_placement`: `true` (новое поле)

---

## 5. Подробное описание каждого модуля

### 5.1. Поток данных в одном торговом цикле

```
1. run_cycle() запускается каждые 10 сек
   │
   ├── Получение 499 бумаг с MOEX (цены, объёмы, BID/OFFER)
   │   └── Fallback цены: LAST → MARKETPRICE → LCURRENTPRICE → midspread
   │
   ├── Фильтр ликвидности (адаптивный для выходных)
   │   └── min_tickers_after_filter: 10 (из конфига)
   │
   ├── Генерация сигналов:
   │   ├── _generate_news_signals() — ML-тональность (ruBERT-tiny2)
   │   ├── technical_core.analyze_all_tickers() — RSI, MACD, BB, ATR
   │   └── _generate_microstructure_signals() — imbalance, spread ← НОВОЕ
   │
   ├── 🆕 Обновление Хокса:
   │   ├── hawkes.update_price() для всех тикеров
   │   └── hawkes.fit() для тикеров в портфеле (каждые 500 циклов)
   │
   ├── check_stops_and_tp(prices, securities):
   │   ├── Трейлинг-стоп (активация +3%, дистанция 2%)
   │   ├── Частичная фиксация (50% при +3%)
   │   ├── Тейк-профит (полный при +12%)
   │   ├── Стоп-лосс (при −6%)
   │   └── 🆕 КАСКАДНЫЙ ПРЕДСКАЗАТЕЛЬ:
   │       ├── Определение уровня (0-5) по PnL и hold_time
   │       ├── Сбор данных: RL predictor + индикаторы + микроструктура
   │       │   + Хокс-сигнал + Херст + D₂
   │       ├── Прогноз P(снижение)
   │       └── Если P(снижение) > порог → ПРОДАТЬ
   │
   ├── _execute_trading_decisions():
   │   ├── RL-модель выбирает действие (7 вариантов)
   │   ├── Выбор стратегии (8 вариантов)
   │   └── Risk manager: размер позиции через ATR
   │
   └── _periodic_learning():
       ├── Prioritized replay (каждые 10 циклов)
       ├── Extreme learning (PnL > 8%)
       ├── Regular learning (каждые 10 циклов)
       └── LLM-коуч: ОТКЛЮЧЁН
```

### 5.2. Каскадный предсказатель — детально

После покупки тикера каскад прогнозирует цену:

**Уровень 0 (PnL ≥ 0):**
- Горизонт: 5 дней (120 часов) — по Ляпунову
- Источники: RL (40%) + Хокс (15%) + Херст (15%) + техника (20%) + микроструктура (10%)
- Решение: продать если P(снижение) > 50%

**Переход на уровень 1 (PnL < 0, hold < 48ч):**
- Горизонт: 2 дня (48 часов)
- Источники: RL (40%) + техника (20%) + Хокс (15%) + Херст (15%) + микроструктура (10%)
- Решение: продать если P(снижение) > 55%

**Переход на уровень 2 (PnL < 0, hold ≥ 48ч):**
- Горизонт: 1 день (24 часа)
- Источники: техника (30%) + RL (30%) + микроструктура (15%) + Херст (15%) + Хокс (10%)
- Решение: продать если P(снижение) > 60%

**Переход на уровень 3 (hold ≥ 120ч = 5 дней):**
- Горизонт: 1 час
- Источники: техника (35%) + микроструктура (30%) + RL (20%) + Херст (10%) + Хокс (5%)
- Решение: продать если P(снижение) > 65%

**Уровни 4-5:** 15 мин и 1 мин — микроструктура доминирует (55%)

**Каскад только углубляется** — не откатывается назад. Это означает: если позиция
перешла на уровень 2, она не вернётся на уровень 1, даже если PnL стал положительным.

### 5.3. Хокс процесс — детально

**События:** |log-return| > 1% (настраивается в конфиге)
- bullish: log-return > +threshold
- bearish: log-return < −threshold

**EM-алгоритм (до 50 итераций):**
1. E-step: для каждого события считаем вероятность, что оно фоновое (p_bg) или
   вызвано предыдущим (p_trig = 1 − p_bg)
2. M-step: обновляем параметры:
   - μ = Σp_bg / T (фоновая интенсивность)
   - η = Σp_trig / n (branching ratio, 0.01-0.95)
   - β = Σp_trig / Σ(p_trig × B/A) (скорость затухания)

**Прогноз:**
```
E[событий за horizon] = μ × horizon + η × Σ (1 − e^(−β×h)) × e^(−β×(t_now − t_i))
P(≥1 событие) = 1 − e^(−E[событий])
```

**Branching ratio η** показывает силу кластеризации:
- η = 0.85 (наш backtest) — события сильно кластеризуются
- η → 0 — события независимы (случайные)
- η → 1 — критическая кластеризация (лавина)

---

## 6. Что ожидается от внедрения

### 6.1. Прямые эффекты

| Метрика | До внедрения | После внедрения | Источник улучшения |
|---------|-------------|-----------------|-------------------|
| Hit rate | ~50% | 55-65% | Хокс (80% на 2д) + микроструктура |
| Sharpe | <0 | 0.5-1.0 | Каскад (продаёт перед снижением) |
| Max drawdown | ? | −20-30% | D₂ (риск-менеджмент в кризис) |
| Hold time | 1-72ч (случайно) | 1-120ч (по Ляпунову) | Каскад (макс 5 дней) |
| Выбор стратегии | 8 фиксированных | 8 + адаптация по H | Херст (momentum vs mean_reversion) |
| Brent в macro | 0.0 (баг) | 0.0 (не фиксировали) | Требует отдельного фикса |
| CNYRUB | Не было | 11.317₽ | Корреляционный анализ (β=−0.15) |
| Weekend trading | Не работало | Работает | allow_order_placement: true |

### 6.2. Косвенные эффекты

1. **Модель становится «физически осведомлённой»:**
   - Знает, что рынок персистентный (H=0.58) → momentum лучше
   - Знает, что горизонт предсказуемости = 5 дней → не держит дольше
   - Знает, что D₂ = 2.5 → рынок управляется 2-3 факторами
   - Знает, что Хокс η = 0.85 → события кластеризуются

2. **Адаптивность к режимам:**
   - В кризис (D₂ > 2.6) — сокращает позиции
   - В спокойный период (D₂ < 2.4) — торгует агрессивнее
   - При H > 0.6 — momentum-стратегия
   - При H < 0.45 — mean_reversion

3. **Скорость:**
   - LLM-коуч отключён → циклы быстрее на 2-20 сек
   - Микроструктура: миллисекунды (локальный расчёт)
   - Хокс: EM-алгоритм ~0.1 сек на тикер

### 6.3. Честные риски

1. **Модель необучена** — 310 опытов это мало (нужно 5000+)
2. **Предобучение прерывается** — 37 тикеров × ~1 мин = ~40 мин
3. **Brent = 0** — yfinance не работает, нужен альтернативный источник
4. **Хокс накапливает данные** — первые 500 циклов работает без переобучения
5. **Микроструктура в выходные** — BIDDEPTH/OFFERDEPTH = None, imbalance = 0

---

## 7. Как перенести изменения

### Структура архива

Архив `russian_ai_trader_changes.tar.gz` содержит:

```
russian_ai_trader_changes/
├── DOCUMENTATION.md          — этот файл
├── NEW_FILES/
│   ├── core/
│   │   └── hawkes_signal.py
│   ├── fetchers/
│   │   ├── history_loader.py
│   │   └── microstructure_fetcher.py
│   └── models/
│       └── price_predictor.py
└── MODIFIED_FILES/
    ├── main.py
    ├── Tools_pre_train.py
    ├── config/
    │   ├── market_schedule.json
    │   ├── rl_config.json
    │   └── settings.json
    ├── core/
    │   └── trading_hours_scheduler.py
    ├── fetchers/
    │   └── moex_fetcher.py
    └── models/
        └── smart_broker.py
```

### Установка

1. Распаковать архив в корень проекта:
```bash
cd /path/to/russian_ai_trader
tar xzf russian_ai_trader_changes.tar.gz
```

2. Скопировать новые файлы:
```bash
cp -r russian_ai_trader_changes/NEW_FILES/* .
```

3. Скопировать изменённые файлы (с заменой):
```bash
cp -r russian_ai_trader_changes/MODIFIED_FILES/* .
```

4. Установить зависимости (если не установлены):
```bash
pip install feedparser beautifulsoup4 pyahocorasick
```

5. Очистить старую модель и портфель:
```bash
rm -f models/saved_trader/*
echo '{"cash": 10000.0, "total_value": 10000.0, "initial_capital": 10000.0, "positions": {}, "trade_history": [], "total_trades": 0, "total_pnl": 0.0, "total_commission": 0.0, "last_update": "", "sector_allocation": {}, "correlation_matrix": {}}' > data/portfolio_state.json
```

6. Предобучить модель:
```bash
python Tools_pre_train.py
```

7. Запустить систему:
```bash
python main.py
```

---

## 8. Известные ограничения и следующие шаги

### Что не внедрено

1. **Brent цена** — yfinance не работает из-за rate limit. Нужно заменить на
   MOEX фьючерс BR или Stooq.
2. **Полное предобучение** — 37 тикеров требуют ~40 минут. Текущие 310 опытов —
   стартовый минимум. Нужно запустить `Tools_pre_train.py` на машине без таймаутов.
3. **Хокс на исторических данных** — Хокс начинает накапливать данные только с
   момента запуска. Для мгновенной работы нужно загрузить исторические события
   из `price_history_extended.json`.

### Что нужно сделать после переноса

1. **Запустить `Tools_pre_train.py`** на машине без таймаутов (40 минут)
2. **Проверить Brent** — если нужен, заменить источник в `moex_fetcher.py`
3. **Накопить 500 циклов** для первого переобучения Хокса
4. **Мониторить логи** — каскад логирует `🎯 КАСКАД-ПРОДАЖА` при срабатывании
5. **Через 1 неделю** — сравнить hit rate с baseline (50%)

### Что НЕ менять

- Архитектуру `smart_broker.py` (God Object) — не трогать до стабилизации
- `trader_model.py` — размерность state vector (227) не менять без переобучения
- Структуру конфигов — все новые секции уже добавлены

---
*Документация подготовлена на основе 6 итераций исследований IMOEX.*
*Дата: 12 июля 2026*
