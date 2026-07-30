"""
Веб-сервер на Dash для торгового интерфейса
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import threading
import os
from typing import Dict, List, Optional, Any

from web.web_dashboard import dashboard_viz

from utils.logger import get_logger
from models.smart_broker import SmartPortfolioBroker
from models.trainer import model_trainer_instance

logger = get_logger("WEB_APP")


def get_portfolio_tickers():
    """Получаем тикеры из текущего портфеля"""
    try:
        portfolio_path = 'data/portfolio_state.json'
        if os.path.exists(portfolio_path):
            with open(portfolio_path, 'r', encoding='utf-8') as f:
                portfolio = json.load(f)

            positions = portfolio.get('positions', {})
            portfolio_tickers = list(positions.keys())

            if 'MOEX' not in portfolio_tickers:
                portfolio_tickers.append('MOEX')

            return portfolio_tickers
    except Exception as e:
        logger.error(f"Ошибка загрузки портфеля: {e}")

    return ['SBER', 'GAZP', 'LKOH', 'MOEX']


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)

app.title = "AI Trader - Российский рынок"

# 🆕 v16.4: JavaScript для обработки кликов по кнопкам коррекции сентимента
app.index_string = '''
<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
<script>
// v16.4: Обработка кликов по кнопкам коррекции сентимента
document.addEventListener('click', function(e) {
    var btn = e.target.closest('button[data-label]');
    if (!btn) return;
    e.preventDefault();
    var label = btn.getAttribute('data-label');
    var newsData = btn.getAttribute('data-news');
    if (!newsData) return;
    try {
        var news = JSON.parse(newsData);
        fetch('/api/sentiment/correct', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                title: news.title || '',
                summary: news.summary || '',
                source: news.source || '',
                original_label: news.original_label || 'NEUTRAL',
                original_sentiment: news.original_sentiment || 0.0,
                corrected_label: label
            })
        }).then(function(r) { return r.json(); })
          .then(function(d) {
            if (d.status === 'ok') {
                btn.style.opacity = '0.5';
                btn.title = 'Сохранено: ' + label + ' (всего: ' + d.total_corrections + ')';
            } else {
                alert('Ошибка: ' + (d.error || 'неизвестно'));
            }
        }).catch(function(err) { alert('Ошибка сети: ' + err); });
    } catch(ex) { alert('Ошибка: ' + ex); }
});
</script>
</footer>
</body>
</html>
'''

broker_instance = None
update_interval = 5000
web_stop_event = threading.Event()


def run_web_server(broker: SmartPortfolioBroker, stop_event: threading.Event):
    """Запуск веб-сервера"""
    global broker_instance, web_stop_event
    broker_instance = broker
    web_stop_event = stop_event

    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            port = config.get('web_port', 8050)
    except:
        port = 8050

    logger.info(f"Запуск веб-сервера на порту {port}")

    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False
        )
    except Exception as e:
        logger.error(f"Ошибка запуска веб-сервера: {e}")


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🤖 AI Trader - Российский рынок",
                    className="text-center mb-4",
                    style={'color': '#00ff88'}),
            html.P("Профессиональная система алгоритмической торговли с AI",
                   className="text-center text-muted mb-4")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Nav([
                dbc.NavLink("📊 Дашборд", href="/", active="exact", id="nav-dashboard"),
                dbc.NavLink("💼 Портфель", href="/portfolio", active="exact", id="nav-portfolio"),
                dbc.NavLink("📈 Графики", href="/charts", active="exact", id="nav-charts"),
                dbc.NavLink("📰 Новости", href="/news", active="exact", id="nav-news"),
                dbc.NavLink("⚙️ Настройки", href="/settings", active="exact", id="nav-settings"),
                dbc.NavLink("📋 Логи", href="/logs", active="exact", id="nav-logs")
            ], pills=True, vertical=False, className="mb-4")
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id="page-content")
        ])
    ]),

    dcc.Interval(
        id='interval-component',
        interval=update_interval,
        n_intervals=0
    ),

    dcc.Store(id='portfolio-store'),
    dcc.Store(id='signals-store'),
    dcc.Store(id='session-store')
], fluid=True, className="p-4")


dashboard_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🟢 Статус системы", className="bg-success text-white"),
                dbc.CardBody([
                    html.H4("АКТИВНА", id="system-status", className="text-success"),
                    html.P("Торговая система работает", className="text-muted"),
                    html.Div([
                        dbc.Badge("Рынок открыт", color="success", id="market-status"),
                        dbc.Badge("AI онлайн", color="info", className="ms-2"),
                        dbc.Badge("Новости активны", color="warning", className="ms-2")
                    ], className="mt-2")
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("💰 Капитал", className="bg-primary text-white"),
                dbc.CardBody([
                    html.H2("10,000 ₽", id="total-capital"),
                    html.P("Общая стоимость", className="text-muted"),
                    html.Div([
                        html.Span("Кэш: ", className="text-muted"),
                        html.Span("10,000 ₽", id="cash-amount", className="fw-bold"),
                        html.Span(" | Позиции: ", className="text-muted ms-2"),
                        html.Span("0 ₽", id="positions-value", className="fw-bold")
                    ])
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 PnL", className="bg-info text-white"),
                dbc.CardBody([
                    html.H2("+0.00%", id="pnl-percent", className="text-success"),
                    html.P("Доходность", className="text-muted"),
                    html.Div([
                        html.Span("Сегодня: ", className="text-muted"),
                        html.Span("+0 ₽", id="daily-pnl", className="fw-bold"),
                        html.Span(" | Всего: ", className="text-muted ms-2"),
                        html.Span("+0 ₽", id="total-pnl", className="fw-bold")
                    ]),
                    html.Div([
                        html.Span("💎 Накоплено: ", className="text-muted"),
                        html.Span("0 ₽", id="fixated-profit", className="fw-bold text-info")
                    ], className="mt-2")
                ])
            ])
        ], width=4)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Активные позиции"),
                dbc.CardBody([
                    html.Div(id="positions-table",
                             className="table-responsive",
                             style={'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🚦 Последние сигналы"),
                dbc.CardBody([
                    html.Div(id="signals-list",
                             className="table-responsive",
                             style={'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎮 Управление системой"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("▶️ Старт торговли", id="start-trading-btn", color="success", className="w-100 mb-2", disabled=False),
                            dbc.Button("⏸️ Пауза", id="pause-trading-btn", color="warning", className="w-100 mb-2", disabled=True),
                            dbc.Button("⏹️ Стоп", id="stop-trading-btn", color="danger", className="w-100", disabled=False)
                        ], width=4),
                        dbc.Col([
                            dbc.Button("📊 Обновить данные",
                                       id="refresh-data-btn",
                                       color="info",
                                       className="w-100 mb-2"),
                            dbc.Button("💾 Сохранить состояние",
                                       id="save-state-btn",
                                       color="primary",
                                       className="w-100 mb-2"),
                            dbc.Button("🔄 Ребалансировка",
                                       id="rebalance-btn",
                                       color="secondary",
                                       className="w-100 mb-2"),

                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.Label("Скорость обновления:", className="form-label"),
                                dcc.Slider(id="update-speed-slider", min=1, max=10, step=1, value=5,
                                           marks={i: f"{i}s" for i in range(1, 11)})
                            ]),
                            html.Div([
                                html.Label("Уровень риска:", className="form-label mt-3"),
                                dcc.Slider(id="risk-level-slider", min=1, max=10, step=1, value=5,
                                           marks={1: 'Консерв.', 5: 'Умерен.', 10: 'Агресс.'})
                            ])
                        ], width=4)
                    ])
                ])
            ])
        ])
    ]),

    # 🆕 Карточки новых модулей (Вариант F + D + Hawkes + Chaos)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎯 Вариант F: Entry Cascading Confirmer", className="bg-info text-white"),
                dbc.CardBody([
                    html.Div(id="entry-confirmer-status",
                             style={'maxHeight': '200px', 'overflowY': 'auto'})
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔄 Вариант D: Rolling Exit Manager", className="bg-warning text-white"),
                dbc.CardBody([
                    html.Div(id="rolling-exit-status",
                             style={'maxHeight': '200px', 'overflowY': 'auto'})
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⚡ Hawkes Process (per-ticker thresholds)", className="bg-danger text-white"),
                dbc.CardBody([
                    html.Div(id="hawkes-status",
                             style={'maxHeight': '250px', 'overflowY': 'auto'})
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🌀 Chaos Metrics (Hurst, D₂, RQA, kurtosis)", className="bg-success text-white"),
                dbc.CardBody([
                    html.Div(id="chaos-metrics-status",
                             style={'maxHeight': '250px', 'overflowY': 'auto'})
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    # 🆕 Таблица rolling exit позиций (детально)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔍 Rolling Exit: активные позиции (детально)"),
                dbc.CardBody([
                    html.Div(id="rolling-exit-positions-table",
                             className="table-responsive",
                             style={'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ])
        ])
    ], className="mb-4"),

    # 🆕 Таблица хаос-метрик топ-тикеров
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Хаос-метрики: топ-тикеры по волатильности"),
                dbc.CardBody([
                    html.Div(id="chaos-metrics-table",
                             className="table-responsive",
                             style={'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ])
        ])
    ])
])


portfolio_layout = dbc.Container([
    dbc.Row([dbc.Col([html.H2("💼 Детали портфеля", className="mb-4")])]),
    dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("📋 Все позиции"), dbc.CardBody([html.Div(id="detailed-positions-table")])])])], className="mb-4"),
    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardHeader("📊 Распределение по секторам"), dbc.CardBody([dcc.Graph(id="sector-pie-chart")])])], width=6),
        dbc.Col([dbc.Card([dbc.CardHeader("📈 Динамика стоимости"), dbc.CardBody([dcc.Graph(id="portfolio-value-chart")])])], width=6)
    ], className="mb-4"),
    dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("📝 История сделок"), dbc.CardBody([html.Div(id="trade-history-table")])])])])
])


charts_layout = dbc.Container([
    dbc.Row([dbc.Col([html.H2("📈 Аналитические графики", className="mb-4")])]),
    dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("🔍 Выбор инструмента"), dbc.CardBody([dbc.Row([
        dbc.Col([html.Label("Тикер:", className="form-label"), dcc.Dropdown(id="ticker-selector", options=[], value='MOEX', clearable=False)], width=4),
        dbc.Col([html.Label("Период:", className="form-label"), dcc.Dropdown(id="period-selector", options=[
            {'label': '1 день', 'value': '1d'}, {'label': '1 неделя', 'value': '1w'}, {'label': '1 месяц', 'value': '1m'},
            {'label': '3 месяца', 'value': '3m'}, {'label': '6 месяцев', 'value': '6m'}], value='1m', clearable=False)], width=4),
        dbc.Col([html.Label("Интервал:", className="form-label"), dcc.Dropdown(id="interval-selector", options=[
            {'label': '1 минута', 'value': '1min'}, {'label': '5 минут', 'value': '5min'}, {'label': '15 минут', 'value': '15min'},
            {'label': '1 час', 'value': '1h'}, {'label': '1 день', 'value': '1d'}], value='1h', clearable=False)], width=4)
    ])])])])], className="mb-4"),
    dbc.Row([dbc.Col([dbc.Card([dbc.CardHeader("📊 Цена и объем"), dbc.CardBody([dcc.Graph(id="price-volume-chart")])])])], className="mb-4"),
    dbc.Row([
        dbc.Col([dbc.Card([dbc.CardHeader("📉 Технические индикаторы"), dbc.CardBody([dcc.Graph(id="indicators-chart")])])], width=6),
        dbc.Col([dbc.Card([dbc.CardHeader("📰 Новостной сентимент"), dbc.CardBody([dcc.Graph(id="sentiment-chart")])])], width=6)
    ])
])


# ============================================================
# 📰 Страница "Новости и сентимент"
# ============================================================
news_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("📰 Новости, вызвавшие сентимент", className="mb-3"),
            html.P("Лента новостей с ML-анализом тональности. Только новости с ненулевым сентиментом "
                   "влияют на торговые решения. Кэш обновляется каждые 30 секунд.",
                   className="text-muted mb-4")
        ])
    ]),

    # —— Сводная статистика ——
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Сводка по сентименту", className="bg-info text-white"),
                dbc.CardBody([
                    html.Div(id="news-sentiment-stats",
                             className="d-flex flex-wrap gap-3 justify-content-around")
                ])
            ])
        ])
    ], className="mb-4"),

    # —— Топ-3 позитивных и негативных ——
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🟢 Топ-3 позитивных новости", className="bg-success text-white"),
                dbc.CardBody([
                    html.Div(id="news-top-positive",
                             style={'maxHeight': '220px', 'overflowY': 'auto'})
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔴 Топ-3 негативных новости", className="bg-danger text-white"),
                dbc.CardBody([
                    html.Div(id="news-top-negative",
                             style={'maxHeight': '220px', 'overflowY': 'auto'})
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    # —— Фильтры ——
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔍 Фильтры"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Тикер:", className="form-label"),
                            dcc.Dropdown(
                                id='news-ticker-filter',
                                options=[{'label': 'Все тикеры', 'value': 'ALL'}],
                                value='ALL',
                                clearable=False
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Сентимент:", className="form-label"),
                            dcc.Dropdown(
                                id='news-label-filter',
                                options=[
                                    {'label': 'Все', 'value': 'ALL'},
                                    {'label': '🟢 Позитивный', 'value': 'POSITIVE'},
                                    {'label': '🔴 Негативный', 'value': 'NEGATIVE'},
                                    {'label': '⚪ Нейтральный', 'value': 'NEUTRAL'},
                                ],
                                value='ALL',
                                clearable=False
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Мин. |sentiment|:", className="form-label"),
                            dcc.Slider(
                                id='news-min-sentiment-slider',
                                min=0, max=0.9, step=0.05, value=0.05,
                                marks={0: '0', 0.3: '0.3', 0.6: '0.6', 0.9: '0.9'},
                                tooltip={'placement': 'bottom', 'always_visible': False}
                            )
                        ], width=4)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),

    # —— Таблица новостей ——
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    "📋 Лента новостей с сентиментом",
                    html.Span(id="news-count-badge", className="badge bg-secondary ms-2")
                ]),
                dbc.CardBody([
                    html.Div(id="news-feed-table",
                             className="table-responsive",
                             style={'maxHeight': '500px', 'overflowY': 'auto'})
                ])
            ])
        ])
    ]),

    # 🆕 v16.4: Блок дообучения модели сентимента
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎓 Дообучение модели сентимента", className="bg-primary text-white"),
                dbc.CardBody([
                    html.P("Нажмите 🟢/🔴/⚪ в колонке «Коррекция» у каждой новости. "
                           "После накопления 5+ коррекций — нажмите «Дообучить».",
                           className="text-muted small mb-3"),
                    html.Div(id="sentiment-ft-stats", className="mb-2"),
                    dbc.Button("🎓 Дообучить модель", id="sentiment-ft-button",
                               color="primary", className="w-100 mb-2", disabled=True),
                    html.Div(id="sentiment-ft-status", className="text-center small text-muted"),
                    html.Div(id="sentiment-ft-result", className="mt-2"),
                    html.Hr(className="my-2"),
                    # 🆕 v16.7: Кнопка перезапуска NewsAnalyzer
                    dbc.Button("🔄 Перезапустить анализатор новостей", id="reload-news-analyzer-btn",
                               color="success", size="sm", className="w-100 mb-2"),
                    html.Div(id="reload-analyzer-status", className="text-center small"),
                    html.Hr(className="my-2"),
                    html.Details([
                        html.Summary("📋 Список коррекций", className="small text-muted"),
                        html.Div(id="sentiment-corrections-list",
                                 style={'maxHeight': '200px', 'overflowY': 'auto'},
                                 className="mt-2 small")
                    ])
                ])
            ])
        ])
    ], className="mt-3"),

    # 🆕 v16.4: Hidden store для передачи данных коррекции
    dcc.Store(id='sentiment-correction-store'),
    # Hidden div для приёма кликов коррекции (через clientside)
    html.Div(id="sentiment-correction-trigger", style={'display': 'none'}),
    # Кнопка для обновления списка коррекций
    dbc.Button("🔄 Обновить список коррекций", id="refresh-corrections-btn",
               color="secondary", size="sm", className="mt-2 w-100"),
])
# Часть 2 из 4: settings_layout, logs_layout

settings_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("⚙️ Настройки системы", className="mb-4")
        ])
    ]),

    # Торговые настройки
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Торговые параметры"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Начальный капитал (₽):", className="form-label"),
                            dbc.Input(id="initial-capital-input", type="number", value=10000, min=1000, step=1000)
                        ], width=4),
                        dbc.Col([
                            html.Label("Макс. позиций:", className="form-label"),
                            dbc.Input(id="max-positions-input", type="number", value=5, min=1, max=20)
                        ], width=4),
                        dbc.Col([
                            html.Label("Макс. вес позиции (%):", className="form-label"),
                            dbc.Input(id="max-weight-input", type="number", value=20, min=5, max=100, step=5)
                        ], width=4)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Стоп-лосс (%):", className="form-label"),
                            dbc.Input(id="stop-loss-input", type="number", value=3.0, min=0.5, max=10, step=0.5)
                        ], width=4),
                        dbc.Col([
                            html.Label("Тейк-профит (%):", className="form-label"),
                            dbc.Input(id="take-profit-input", type="number", value=6.0, min=1, max=20, step=0.5)
                        ], width=4),
                        dbc.Col([
                            html.Label("Риск на сделку (%):", className="form-label"),
                            dbc.Input(id="risk-per-trade-input", type="number", value=1.5, min=0.5, max=5, step=0.1)
                        ], width=4)
                    ], className="mb-3"),
                    dbc.Button("💾 Сохранить настройки", id="save-settings-btn", color="primary", className="w-100"),
                    html.Hr(),
                    dbc.Button("🔧 Обновить конфиги",
                               id="reload-configs-btn",
                               color="warning",
                               className="w-100 mt-2")
                ])
            ])
        ])
    ], className="mb-4"),

    # Защитные лимиты
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🛡️ Защитные лимиты"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Кулдаун (сек):", className="form-label"),
                            dbc.Input(id="cooldown-input", type="number", value=1800, min=60, step=60)
                        ], width=4),
                        dbc.Col([
                            html.Label("Мин. сумма сделки (₽):", className="form-label"),
                            dbc.Input(id="min-cash-input", type="number", value=1000, min=100, step=100)
                        ], width=4),
                        dbc.Col([
                            html.Label("Мин. риск на акцию (%):", className="form-label"),
                            dbc.Input(id="min-risk-input", type="number", value=0.01, min=0.001, max=1.0, step=0.001)
                        ], width=4)
                    ], className="mb-3"),
                ])
            ])
        ])
    ], className="mb-4"),

    # Фиксация прибыли
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("💰 Фиксация дневной прибыли"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "Включить фиксацию", "value": "enabled"}],
                                value=["enabled"],
                                id="fixation-enabled-check",
                                switch=True
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Мин. прибыль для фиксации (₽):", className="form-label"),
                            dbc.Input(id="fixation-min-profit-input", type="number", value=50, min=10, step=10)
                        ], width=4),
                        dbc.Col([
                            html.Label("Реинвестировать (%):", className="form-label"),
                            dcc.Slider(
                                id="fixation-reinvest-slider",
                                min=0, max=100, step=10, value=0,
                                marks={0: '0%', 50: '50%', 100: '100%'}
                            )
                        ], width=4)
                    ], className="mb-3"),
                ])
            ])
        ])
    ], className="mb-4"),

    # Настройки LLM-коуча
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🧠 LLM-Коуч", id="coach-card-header"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "Включить коуча", "value": "enabled"}],
                                value=[], id="coach-enabled-check", switch=True
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Модель:", className="form-label"),
                            html.Div([
                                dcc.Dropdown(
                                    id="coach-model-dropdown",
                                    options=[],
                                    value="",
                                    placeholder="Выберите модель...",
                                    clearable=False
                                ),
                                dbc.Button("🔄 Обновить список", id="refresh-models-btn", color="secondary", size="sm",
                                           className="mt-1 w-100")
                            ])
                        ], width=5),
                        dbc.Col([
                            html.Div(id="coach-model-status", className="mt-4 text-center")
                        ], width=3)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Частота вызовов (циклов):", className="form-label mt-2"),
                            dbc.Input(id="coach-interval-input", type="number", value=20, min=5, max=100, step=5)
                        ], width=4),
                        dbc.Col([
                            html.Label("Таймаут (сек):", className="form-label mt-2"),
                            dbc.Input(id="coach-timeout-input", type="number", value=30, min=10, max=120, step=5)
                        ], width=4),
                        dbc.Col([
                            html.Label("Вес советов:", className="form-label mt-2"),
                            dcc.Slider(id="coach-weight-slider", min=0.1, max=1.0, step=0.1, value=0.3,
                                       marks={0.1: '0.1', 0.5: '0.5', 1.0: '1.0'})
                        ], width=4)
                    ], className="mt-2"),
                    dbc.Button("💾 Сохранить настройки коуча", id="save-coach-settings-btn", color="primary",
                               className="w-100 mt-2")
                ])
            ])
        ])
    ], className="mb-4"),


    # Настройки торговых сессий
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⏰ Часы торговли"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Основная сессия:", className="form-label"),
                            dbc.InputGroup([
                                dbc.Input(id="session-start-input", type="time", value="06:50"),
                                dbc.InputGroupText("-"),
                                dbc.Input(id="session-end-input", type="time", value="18:50")
                            ])
                        ], width=6),
                        dbc.Col([
                            html.Label("Вечерняя сессия:", className="form-label"),
                            dbc.InputGroup([
                                dbc.Input(id="evening-start-input", type="time", value="19:00"),
                                dbc.InputGroupText("-"),
                                dbc.Input(id="evening-end-input", type="time", value="23:50")
                            ])
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "Торговля в выходные", "value": "weekend"}],
                                value=[], id="weekend-trading-check", switch=True
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "Принудительный режим 24/7", "value": "force"}],
                                value=[], id="force-trading-check", switch=True
                            )
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="session-save-status", className="mt-2 text-center"),
                            dbc.Button("💾 Сохранить часы торговли", id="session-save-btn", color="primary", className="w-100 mt-3")
                        ])
                    ])
                ])
            ])
        ])
    ], className="mb-4"),

    # RSS источники
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📰 RSS источники новостей"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(id="new-rss-url", placeholder="Введите URL RSS-ленты...", type="url", className="mb-2"),
                            dbc.Input(id="new-rss-name", placeholder="Название источника (необязательно)", className="mb-2"),
                            dbc.Button("➕ Добавить источник", id="add-rss-btn", color="success", className="w-100"),
                            html.Div(id="rss-action-status", className="mt-2 text-center"),
                            html.Hr(),
                            html.H5("Текущие источники"),
                            html.Div(id="rss-sources-list")
                        ])
                    ])
                ])
            ])
        ])
    ]),

    # 🆕 Screen Universe (выбор тикеров)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🌍 Screen Universe — отбор тикеров"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Топ-N тикеров:", className="form-label"),
                            dbc.Input(id="su-top-n-input", type="number", value=60, min=10, max=200, step=10)
                        ], width=3),
                        dbc.Col([
                            html.Label("Мин. объём/день (₽):", className="form-label"),
                            dbc.Input(id="su-min-vol-input", type="number", value=50000000, min=1000000, step=1000000)
                        ], width=3),
                        dbc.Col([
                            html.Label("Мин. цена (₽):", className="form-label"),
                            dbc.Input(id="su-min-price-input", type="number", value=5.0, min=1.0, step=1.0)
                        ], width=3),
                        dbc.Col([
                            html.Label("Мин. волатильность (%):", className="form-label"),
                            dbc.Input(id="su-min-volat-input", type="number", value=0.3, min=0.1, step=0.1)
                        ], width=3),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Вес ликвидности:", className="form-label"),
                            dcc.Slider(id="su-liq-weight-slider", min=0, max=1, step=0.1, value=0.5,
                                       marks={0: '0', 0.5: '0.5', 1: '1'})
                        ], width=6),
                        dbc.Col([
                            html.Label("Вес волатильности:", className="form-label"),
                            dcc.Slider(id="su-vol-weight-slider", min=0, max=1, step=0.1, value=0.5,
                                       marks={0: '0', 0.5: '0.5', 1: '1'})
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Button("💾 Сохранить screen_universe",
                               id="save-screen-universe-btn", color="primary", className="w-100"),
                    html.Div(id="screen-universe-status", className="mt-2 text-center")
                ])
            ])
        ])
    ], className="mb-4"),

    # 🆕 Вариант F: Entry Cascading
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎯 Вариант F: Entry Cascading Confirmer"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "Включить", "value": "enabled"}],
                                value=["enabled"], id="ec-enabled-check", switch=True
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Min confidence:", className="form-label"),
                            dcc.Slider(id="ec-min-conf-slider", min=0.2, max=0.8, step=0.05, value=0.45,
                                       marks={0.2: '0.2', 0.45: '0.45', 0.6: '0.6', 0.8: '0.8'})
                        ], width=3),
                        dbc.Col([
                            html.Label("Max positions:", className="form-label"),
                            dbc.Input(id="ec-max-pos-input", type="number", value=5, min=1, max=20)
                        ], width=3),
                        dbc.Col([
                            html.Label("Max weight %:", className="form-label"),
                            dbc.Input(id="ec-max-weight-input", type="number", value=20, min=5, max=50, step=5)
                        ], width=3),
                    ], className="mb-3"),
                    html.Hr(),
                    html.H6("Hawkes-триггер", className="text-info"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Bull/Bear ratio ≥:", className="form-label"),
                            dbc.Input(id="ec-bull-bear-input", type="number", value=1.5, min=1.0, step=0.1)
                        ], width=4),
                        dbc.Col([
                            html.Label("Min bull expected:", className="form-label"),
                            dbc.Input(id="ec-min-bull-input", type="number", value=0.5, min=0.1, step=0.1)
                        ], width=4),
                        dbc.Col([
                            html.Label("Min P(bull):", className="form-label"),
                            dbc.Input(id="ec-min-pbull-input", type="number", value=0.5, min=0.1, max=0.9, step=0.1)
                        ], width=4),
                    ], className="mb-3"),
                    html.Hr(),
                    html.H6("Техническое подтверждение", className="text-info"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Wait (мин):", className="form-label"),
                            dbc.Input(id="ec-tech-wait-input", type="number", value=0, min=0, max=60, step=5)
                        ], width=3),
                        dbc.Col([
                            html.Label("RSI min:", className="form-label"),
                            dbc.Input(id="ec-rsi-min-input", type="number", value=50, min=20, max=70)
                        ], width=3),
                        dbc.Col([
                            html.Label("RSI max:", className="form-label"),
                            dbc.Input(id="ec-rsi-max-input", type="number", value=70, min=50, max=90)
                        ], width=3),
                        dbc.Col([
                            html.Label("Min mom %:", className="form-label"),
                            dbc.Input(id="ec-min-mom-input", type="number", value=0.1, min=0, step=0.1)
                        ], width=3),
                    ], className="mb-3"),
                    html.Hr(),
                    html.H6("Хаос-фильтр", className="text-info"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Min RQA DET:", className="form-label"),
                            dbc.Input(id="ec-min-det-input", type="number", value=0.25, min=0, max=1, step=0.05)
                        ], width=3),
                        dbc.Col([
                            html.Label("Min L_max:", className="form-label"),
                            dbc.Input(id="ec-min-lmax-input", type="number", value=4, min=1, max=50)
                        ], width=3),
                        dbc.Col([
                            html.Label("Max kurtosis:", className="form-label"),
                            dbc.Input(id="ec-max-kurt-input", type="number", value=100, min=10, max=500, step=10)
                        ], width=3),
                        dbc.Col([
                            html.Label("Min η (branching):", className="form-label"),
                            dbc.Input(id="ec-min-eta-input", type="number", value=0.3, min=0, max=0.95, step=0.05)
                        ], width=3),
                    ], className="mb-3"),
                    dbc.Button("💾 Сохранить entry_cascading",
                               id="save-entry-cascading-btn", color="primary", className="w-100"),
                    html.Div(id="entry-cascading-status", className="mt-2 text-center")
                ])
            ])
        ])
    ], className="mb-4"),

    # 🆕 Вариант D: Rolling Exit
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔄 Вариант D: Rolling Exit Manager"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                options=[{"label": "Включить", "value": "enabled"}],
                                value=["enabled"], id="re-enabled-check", switch=True
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Min hold (ч):", className="form-label"),
                            dbc.Input(id="re-min-hold-input", type="number", value=0, min=0, max=24, step=1)
                        ], width=3),
                        dbc.Col([
                            html.Label("Max hold (ч):", className="form-label"),
                            dbc.Input(id="re-max-hold-input", type="number", value=120, min=24, max=240, step=12)
                        ], width=3),
                        dbc.Col([
                            html.Label("Phase exit:", className="form-label"),
                            dbc.Checklist(
                                options=[{"label": "Включить", "value": "enabled"}],
                                value=["enabled"], id="re-phase-check", switch=True
                            )
                        ], width=3),
                    ], className="mb-3"),
                    html.Hr(),
                    html.H6("Пороги по hold_time", className="text-warning"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Early (ч):", className="form-label"),
                            dbc.Input(id="re-early-h-input", type="number", value=4, min=1, max=12)
                        ], width=3),
                        dbc.Col([
                            html.Label("Early threshold:", className="form-label"),
                            dbc.Input(id="re-early-t-input", type="number", value=0.65, min=0.3, max=0.9, step=0.05)
                        ], width=3),
                        dbc.Col([
                            html.Label("Mid (ч):", className="form-label"),
                            dbc.Input(id="re-mid-h-input", type="number", value=24, min=8, max=72)
                        ], width=3),
                        dbc.Col([
                            html.Label("Mid threshold:", className="form-label"),
                            dbc.Input(id="re-mid-t-input", type="number", value=0.55, min=0.3, max=0.9, step=0.05)
                        ], width=3),
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Late (ч):", className="form-label"),
                            dbc.Input(id="re-late-h-input", type="number", value=72, min=24, max=168)
                        ], width=3),
                        dbc.Col([
                            html.Label("Late threshold:", className="form-label"),
                            dbc.Input(id="re-late-t-input", type="number", value=0.45, min=0.2, max=0.8, step=0.05)
                        ], width=3),
                        dbc.Col([
                            html.Label("Force threshold:", className="form-label"),
                            dbc.Input(id="re-force-t-input", type="number", value=0.30, min=0.1, max=0.6, step=0.05)
                        ], width=3),
                        dbc.Col([], width=3),
                    ], className="mb-3"),
                    html.Hr(),
                    html.H6("Hard stops", className="text-danger"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Base SL %:", className="form-label"),
                            dbc.Input(id="re-base-sl-input", type="number", value=2.5, min=1, max=10, step=0.5)
                        ], width=3),
                        dbc.Col([
                            html.Label("Kurt penalty max %:", className="form-label"),
                            dbc.Input(id="re-kurt-pen-input", type="number", value=5.0, min=0, max=20, step=0.5)
                        ], width=3),
                        dbc.Col([
                            html.Label("Profit taking %:", className="form-label"),
                            dbc.Input(id="re-pt-input", type="number", value=5.0, min=2, max=20, step=0.5)
                        ], width=3),
                        dbc.Col([
                            html.Label("Trailing ATR ×:", className="form-label"),
                            dbc.Input(id="re-trailing-input", type="number", value=1.5, min=0.5, max=5, step=0.5)
                        ], width=3),
                    ], className="mb-3"),
                    dbc.Button("💾 Сохранить rolling_exit",
                               id="save-rolling-exit-btn", color="primary", className="w-100"),
                    html.Div(id="rolling-exit-cfg-status", className="mt-2 text-center")
                ])
            ])
        ])
    ], className="mb-4"),

    # 🆕 Hawkes
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⚡ Hawkes Process"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Default threshold %:", className="form-label"),
                            dbc.Input(id="hw-default-thr-input", type="number", value=0.5, min=0.05, max=5, step=0.05)
                        ], width=3),
                        dbc.Col([
                            html.Label("Refit interval (циклов):", className="form-label"),
                            dbc.Input(id="hw-refit-input", type="number", value=100, min=10, max=1000, step=10)
                        ], width=3),
                        dbc.Col([
                            html.Label("Forecast horizon (ч):", className="form-label"),
                            dbc.Input(id="hw-horizon-input", type="number", value=4, min=1, max=48)
                        ], width=3),
                        dbc.Col([
                            html.Label("Max iter EM:", className="form-label"),
                            dbc.Input(id="hw-max-iter-input", type="number", value=30, min=10, max=200, step=10)
                        ], width=3),
                    ], className="mb-3"),
                    html.Hr(),
                    html.H6("Per-ticker thresholds (калибровка по волатильности)",
                            className="text-info"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Split vol %:", className="form-label"),
                            dbc.Input(id="hw-split-input", type="number", value=0.5, min=0.1, max=2, step=0.1)
                        ], width=4),
                        dbc.Col([
                            html.Label("Low-vol multiplier:", className="form-label"),
                            dbc.Input(id="hw-low-mult-input", type="number", value=0.5, min=0.1, max=1, step=0.1)
                        ], width=4),
                        dbc.Col([
                            html.Label("High-vol multiplier:", className="form-label"),
                            dbc.Input(id="hw-high-mult-input", type="number", value=0.8, min=0.3, max=2, step=0.1)
                        ], width=4),
                    ], className="mb-3"),
                    dbc.Button("💾 Сохранить hawkes",
                               id="save-hawkes-btn", color="primary", className="w-100"),
                    html.Div(id="hawkes-cfg-status", className="mt-2 text-center")
                ])
            ])
        ])
    ], className="mb-4"),

    # 🆕 Обновление тикеров
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔄 Обновить список тикеров (MOEX)"),
                dbc.CardBody([
                    html.P("Пересобрать config/tickers.json из 60 самых ликвидных+волатильных акций MOEX TQBR. "
                           "Скрипт: scripts/select_liquid_volatile.py",
                           className="small text-muted"),
                    dbc.Button("🔄 Обновить тикеры",
                               id="refresh-tickers-btn", color="warning", className="w-100"),
                    html.Div(id="refresh-tickers-status", className="mt-2 text-center")
                ])
            ])
        ])
    ])
])


# Страница логов
logs_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("📋 Системные логи", className="mb-4")
        ])
    ]),

    # Табы внутри страницы логов
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="📋 Журнал событий", tab_id="tab-events"),
                dbc.Tab(label="🧠 Работа моделей", tab_id="tab-models"),
            ], id="logs-tabs", active_tab="tab-events")
        ])
    ], className="mb-3"),

    # Контент вкладок
    dbc.Row([
        dbc.Col([
            html.Div(id="logs-tab-content")
        ])
    ])
])

# Вкладка "Работа моделей" (внутри страницы логов)
models_log_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4("🧠 Рекомендации коуча", className="mb-3"),
            html.Div(id="coach-recommendations-table",
                     style={'maxHeight': '400px', 'overflowY': 'auto'})
        ])
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.H4("🤖 Действия модели", id="model-actions-header", className="mb-3"),
            html.Div(id="model-actions-table",
                     style={'maxHeight': '400px', 'overflowY': 'auto'})
        ])
    ])
])

# Часть 3 из 4: коллбэки навигации, дашборда, управления, настроек, логов

# Коллбэк навигации
@app.callback(
    Output("page-content", "children"),
    [Input("nav-dashboard", "n_clicks"),
     Input("nav-portfolio", "n_clicks"),
     Input("nav-charts", "n_clicks"),
     Input("nav-news", "n_clicks"),
     Input("nav-settings", "n_clicks"),
     Input("nav-logs", "n_clicks")],
    prevent_initial_call=True
)
def update_page_content(dash_clicks, port_clicks, charts_clicks, news_clicks, settings_clicks, logs_clicks):
    """Обновление контента страницы"""
    ctx_triggered = ctx.triggered_id

    if ctx_triggered == "nav-dashboard":
        return dashboard_layout
    elif ctx_triggered == "nav-portfolio":
        return portfolio_layout
    elif ctx_triggered == "nav-charts":
        return charts_layout
    elif ctx_triggered == "nav-news":
        return news_layout
    elif ctx_triggered == "nav-settings":
        return settings_layout
    elif ctx_triggered == "nav-logs":
        return logs_layout

    return dashboard_layout


# Коллбэк обновления дашборда
@app.callback(
    [Output("system-status", "children"),
     Output("market-status", "children"),
     Output("total-capital", "children"),
     Output("cash-amount", "children"),
     Output("positions-value", "children"),
     Output("pnl-percent", "children"),
     Output("daily-pnl", "children"),
     Output("total-pnl", "children"),
     Output("fixated-profit", "children"),
     Output("positions-table", "children"),
     Output("signals-list", "children")],
    [Input("interval-component", "n_intervals"),
     Input("refresh-data-btn", "n_clicks")]
)
def update_dashboard(n_intervals, refresh_clicks):
    """Обновление данных на дашборде"""
    if broker_instance is None:
        return ["ОФФЛАЙН", "Рынок закрыт", "0 ₽", "0 ₽", "0 ₽", "0%", "0 ₽",
                "0 ₽", "0 ₽", "Нет данных", "Нет данных"]

    try:
        summary = broker_instance.get_portfolio_summary()
        session_info = summary.get('session_info', {})

        is_trading_time = session_info.get('is_trading_time', False)
        is_trading_day = session_info.get('is_trading_day', False)
        trading_enabled = broker_instance.trading_enabled

        if trading_enabled and is_trading_time:
            system_status = "АКТИВНА"
        elif trading_enabled and is_trading_day:
            system_status = "ОЖИДАНИЕ"
        else:
            system_status = "ПАУЗА"

        if is_trading_time:
            market_status = "Рынок открыт"
        elif is_trading_day:
            market_status = "Рынок скоро откроется"
        else:
            market_status = "Рынок закрыт"

        total_value = summary.get('total_value', 0)
        cash = summary.get('cash', 0)
        positions_value = total_value - cash

        total_capital = f"{total_value:,.0f} ₽"
        cash_amount = f"{cash:,.0f} ₽"
        positions_val = f"{positions_value:,.0f} ₽"

        initial = summary.get('initial_capital', total_value)
        pnl_percent = ((total_value / initial) - 1) * 100 if initial > 0 else 0
        pnl_abs = total_value - initial
        pnl_percent_text = f"{pnl_percent:+.2f}%"
        pnl_percent_class = "text-success" if pnl_percent >= 0 else "text-danger"

        daily_pnl = summary.get('risk_metrics', {}).get('daily_pnl', 0)
        daily_pnl_text = f"{daily_pnl:+,.0f} ₽"
        total_pnl_text = f"{pnl_abs:+,.0f} ₽"

        positions_table = create_positions_table(summary.get('positions', []))
        signals_list = create_signals_list(summary.get('current_signals', []))

        if n_intervals and n_intervals % 5 == 0:
            save_portfolio_history()

        reserved_cash = summary.get('reserved_cash', 0)
        fixated_text = f"{reserved_cash:+,.0f} ₽"

        return [
            system_status, market_status, total_capital, cash_amount, positions_val,
            html.Span(pnl_percent_text, className=pnl_percent_class),
            html.Span(daily_pnl_text, className="text-success" if daily_pnl >= 0 else "text-danger"),
            html.Span(total_pnl_text, className="text-success" if pnl_abs >= 0 else "text-danger"),
            html.Span(fixated_text, className="text-info fw-bold"),
            positions_table, signals_list
        ]

    except Exception as e:
        logger.error(f"Ошибка обновления дашборда: {e}")
        return ["ОШИБКА", "Ошибка", "0 ₽", "0 ₽", "0 ₽", "0%", "0 ₽",
                "0 ₽", "0 ₽", "Ошибка загрузки", "Ошибка загрузки"]


def create_positions_table(positions: List[Dict]) -> html.Table:
    """Создание таблицы позиций"""
    if not positions:
        return html.P("Нет открытых позиций", className="text-muted")

    header = html.Thead(html.Tr([
        html.Th("Тикер"), html.Th("Кол-во"), html.Th("Ср. цена"),
        html.Th("Тек. цена"), html.Th("Стоимость"), html.Th("PnL"), html.Th("Вес"),
        html.Th("Стратегия"), html.Th("Hold")
    ]))

    rows = []
    for pos in positions:
        pnl_class = "text-success" if pos.get('pnl', 0) >= 0 else "text-danger"
        # Расчёт времени удержания
        import time as _time
        hold_hours = 0
        if pos.get('buy_time'):
            hold_hours = (_time.time() - pos['buy_time']) / 3600
        hold_str = f"{hold_hours:.1f}ч" if hold_hours < 24 else f"{hold_hours/24:.1f}д"
        rows.append(html.Tr([
            html.Td(pos['ticker']),
            html.Td(f"{pos['quantity']:,}"),
            html.Td(f"{pos['avg_price']:.2f}"),
            html.Td(f"{pos.get('current_price', 0):.2f}"),
            html.Td(f"{pos.get('position_value', 0):,.0f} ₽"),
            html.Td(html.Span(f"{pos.get('pnl_percent', 0):+.1f}%", className=pnl_class)),
            html.Td(f"{pos.get('weight', 0):.1f}%"),
            html.Td(pos.get('strategy', '—')),
            html.Td(hold_str)
        ]))

    return html.Table([header, html.Tbody(rows)], className="table table-sm table-hover")


def create_signals_list(signals: List[Dict]) -> html.Table:
    """Создание списка сигналов"""
    if not signals:
        return html.P("Нет активных сигналов", className="text-muted")

    header = html.Thead(html.Tr([
        html.Th("Тикер"), html.Th("Сигнал"), html.Th("Уверенность"), html.Th("Причина"), html.Th("Время")
    ]))

    rows = []
    for sig in signals[:10]:
        action = sig.get('action', 'HOLD')
        confidence = min(sig.get('confidence', 0), 1.0)

        if action == 'BUY':
            action_class = "text-success"
        elif action == 'SELL':
            action_class = "text-danger"
        else:
            action_class = "text-warning"

        if confidence > 0.8:
            conf_class = "text-success"
        elif confidence > 0.6:
            conf_class = "text-warning"
        else:
            conf_class = "text-muted"

        timestamp = sig.get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp[:8]
        else:
            time_str = ''

        rows.append(html.Tr([
            html.Td(sig['ticker']),
            html.Td(html.Span(action, className=action_class)),
            html.Td(html.Span(f"{confidence:.1%}", className=conf_class)),
            html.Td(sig.get('reason', 'N/A')),
            html.Td(time_str)
        ]))

    return html.Table([header, html.Tbody(rows)], className="table table-sm table-hover")


# Коллбэки управления
@app.callback(
    [Output("start-trading-btn", "disabled"),
     Output("pause-trading-btn", "disabled"),
     Output("stop-trading-btn", "disabled")],
    [Input("start-trading-btn", "n_clicks"),
     Input("pause-trading-btn", "n_clicks"),
     Input("stop-trading-btn", "n_clicks")]
)
def control_trading(start_clicks, pause_clicks, stop_clicks):
    """Управление торговлей"""
    if broker_instance is None:
        return [True, True, True]

    ctx_triggered = ctx.triggered_id

    if ctx_triggered == "start-trading-btn":
        broker_instance.trading_enabled = True
        logger.info("Торговля запущена")
        return [True, False, False]
    elif ctx_triggered == "pause-trading-btn":
        broker_instance.trading_enabled = False
        logger.info("Торговля приостановлена")
        return [False, True, False]
    elif ctx_triggered == "stop-trading-btn":
        broker_instance.trading_enabled = False
        logger.info("Торговля остановлена")
        return [False, True, True]

    return [not broker_instance.trading_enabled, broker_instance.trading_enabled, False]


@app.callback(
    Output("save-state-btn", "children"),
    Input("save-state-btn", "n_clicks")
)
def save_system_state(n_clicks):
    """Сохранение состояния системы"""
    if n_clicks and broker_instance is not None:
        try:
            broker_instance._save_portfolio_state()
            broker_instance.model.save_model()
            logger.info("Состояние системы сохранено")
            return "✅ Сохранено!"
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
            return "❌ Ошибка!"
    return "💾 Сохранить состояние"


@app.callback(
    Output("rebalance-btn", "children"),
    Input("rebalance-btn", "n_clicks")
)
def rebalance_portfolio(n_clicks):
    """Ребалансировка портфеля"""
    if n_clicks and broker_instance is not None:
        try:
            prices = broker_instance._get_current_prices()
            if prices:
                securities = broker_instance.moex.get_all_securities()
                broker_instance._rebalance_portfolio(prices, securities)
                logger.info("Ребалансировка выполнена")
                return "✅ Ребалансировано!"
        except Exception as e:
            logger.error(f"Ошибка ребалансировки: {e}")
            return "❌ Ошибка!"
    return "🔄 Ребалансировка"


@app.callback(
    Output("interval-component", "interval"),
    Input("update-speed-slider", "value")
)
def update_refresh_speed(value):
    """Обновление скорости обновления"""
    return value * 1000


# Сохранение настроек
@app.callback(
    Output("save-settings-btn", "children"),
    [Input("save-settings-btn", "n_clicks")],
    [State("initial-capital-input", "value"),
     State("max-positions-input", "value"),
     State("max-weight-input", "value"),
     State("stop-loss-input", "value"),
     State("take-profit-input", "value"),
     State("risk-per-trade-input", "value"),
     State("cooldown-input", "value"),
     State("min-cash-input", "value"),
     State("min-risk-input", "value"),
     State("fixation-enabled-check", "value"),
     State("fixation-min-profit-input", "value"),
     State("fixation-reinvest-slider", "value")]
)
def save_trading_settings(n_clicks, initial_capital, max_positions, max_weight,
                          stop_loss, take_profit, risk_per_trade,
                          cooldown, min_cash, min_risk,
                          fixation_enabled, fixation_min_profit, fixation_reinvest):
    """Сохранение торговых настроек"""
    if n_clicks is None or n_clicks == 0:
        return "💾 Сохранить настройки"

    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)

        settings['initial_capital_rub'] = initial_capital
        settings['max_positions'] = max_positions
        settings['max_position_weight_percent'] = max_weight
        settings['stop_loss_percent'] = stop_loss
        settings['take_profit_percent'] = take_profit
        settings['risk_per_trade_percent'] = risk_per_trade
        settings['cooldown_seconds'] = cooldown
        settings['min_cash_per_trade'] = min_cash
        settings['min_risk_per_share_percent'] = min_risk

        if 'daily_profit_fixation' not in settings:
            settings['daily_profit_fixation'] = {}
        settings['daily_profit_fixation']['enabled'] = 'enabled' in fixation_enabled
        settings['daily_profit_fixation']['min_profit_to_fix'] = fixation_min_profit
        settings['daily_profit_fixation']['reinvest_percent'] = fixation_reinvest / 100.0

        with open('config/settings.json', 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)

        if broker_instance is not None:
            broker_instance.settings = settings
            broker_instance.portfolio.initial_capital = initial_capital
            broker_instance.portfolio.max_positions = max_positions
            if hasattr(broker_instance, 'risk_manager'):
                broker_instance.risk_manager.config['cooldown_seconds'] = cooldown
                broker_instance.risk_manager.config['min_cash_per_trade'] = min_cash
                broker_instance.risk_manager.config['min_risk_per_share_percent'] = min_risk

        logger.info(f"Настройки сохранены: капитал={initial_capital}, макс.позиций={max_positions}")
        return "✅ Настройки сохранены!"

    except Exception as e:
        logger.error(f"Ошибка сохранения настроек: {e}")
        return "❌ Ошибка сохранения"


@app.callback(
    Output("session-save-status", "children"),
    [Input("session-save-btn", "n_clicks")],
    [State("session-start-input", "value"),
     State("session-end-input", "value"),
     State("evening-start-input", "value"),
     State("evening-end-input", "value"),
     State("weekend-trading-check", "value"),
     State("force-trading-check", "value")]
)
def save_session_settings(n_clicks, main_start, main_end,
                          evening_start, evening_end, weekend_trading, force_trading):
    """Сохранение настроек торговых сессий"""
    if not n_clicks:
        return ""

    try:
        with open('config/settings.json', 'r', encoding='utf-8') as f:
            settings = json.load(f)

        if 'trading_hours' not in settings:
            settings['trading_hours'] = {}

        settings['trading_hours']['main_session'] = f"{main_start}-{main_end}"
        settings['trading_hours']['evening_session'] = f"{evening_start}-{evening_end}"
        settings['trading_hours']['trade_on_weekend'] = 'weekend' in weekend_trading
        settings['trading_hours']['force_247'] = 'force' in force_trading

        with open('config/settings.json', 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)

        if broker_instance and hasattr(broker_instance, 'scheduler'):
            broker_instance.scheduler.trading_hours = settings['trading_hours']

        logger.info(f"Часы торговли сохранены: {settings['trading_hours']['main_session']}")
        return "✅ Часы торговли сохранены!"

    except Exception as e:
        logger.error(f"Ошибка сохранения часов: {e}")
        return "❌ Ошибка"


@app.callback(
    [Output("rss-action-status", "children"),
     Output("rss-sources-list", "children"),
     Output("new-rss-url", "value"),
     Output("new-rss-name", "value")],
    [Input("add-rss-btn", "n_clicks")],
    [State("new-rss-url", "value"),
     State("new-rss-name", "value")]
)
def add_rss_source(n_clicks, new_url, new_name):
    if n_clicks is None or not new_url:
        return "", "Загружаем список...", "", ""

    config_path = "config/rss_sources.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {'sources': []}

    if any(source.get('url') == new_url for source in config['sources']):
        return f"❌ Источник {new_url} уже добавлен.", "", new_url, new_name

    new_source = {
        'url': new_url,
        'name': new_name or f"Источник {len(config['sources'])+1}",
        'enabled': True,
        'update_interval': 300
    }
    config['sources'].append(new_source)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    sources_list = html.Ul([
        html.Li([
            html.Span(f"📰 {src['name']} ", style={'font-weight': 'bold'}),
            html.Span(f"({src['url'][:50]}...)"),
            dbc.Button("🗑️", id={'type': 'del-rss-btn', 'index': i}, color="danger", size="sm", className="ms-2")
        ]) for i, src in enumerate(config['sources'])
    ])

    return f"✅ Добавлен: {new_source['name']}", sources_list, "", ""


# 🆕 Callback: сохранение screen_universe
@app.callback(
    Output("screen-universe-status", "children"),
    Input("save-screen-universe-btn", "n_clicks"),
    [State("su-top-n-input", "value"),
     State("su-min-vol-input", "value"),
     State("su-min-price-input", "value"),
     State("su-min-volat-input", "value"),
     State("su-liq-weight-slider", "value"),
     State("su-vol-weight-slider", "value")]
)
def save_screen_universe(n_clicks, top_n, min_vol, min_price, min_volat, liq_w, vol_w):
    if n_clicks is None:
        return ""
    if broker_instance is None:
        return "❌ Брокер не инициализирован"
    try:
        config = {
            'top_n': int(top_n or 60),
            'min_avg_daily_volume_rub': int(min_vol or 50000000),
            'min_price': float(min_price or 5.0),
            'min_volatility_pct': float(min_volat or 0.3),
            'history_days': 30,
            'liquidity_weight': float(liq_w or 0.5),
            'volatility_weight': float(vol_w or 0.5),
        }
        result = broker_instance.update_advanced_module_config('screen_universe', config)
        return "✅ Сохранено" if result.get('success') else f"❌ {result.get('error')}"
    except Exception as e:
        return f"❌ Ошибка: {e}"


# 🆕 Callback: сохранение entry_cascading
@app.callback(
    Output("entry-cascading-status", "children"),
    Input("save-entry-cascading-btn", "n_clicks"),
    [State("ec-enabled-check", "value"),
     State("ec-min-conf-slider", "value"),
     State("ec-max-pos-input", "value"),
     State("ec-max-weight-input", "value"),
     State("ec-bull-bear-input", "value"),
     State("ec-min-bull-input", "value"),
     State("ec-min-pbull-input", "value"),
     State("ec-tech-wait-input", "value"),
     State("ec-rsi-min-input", "value"),
     State("ec-rsi-max-input", "value"),
     State("ec-min-mom-input", "value"),
     State("ec-min-det-input", "value"),
     State("ec-min-lmax-input", "value"),
     State("ec-max-kurt-input", "value"),
     State("ec-min-eta-input", "value")]
)
def save_entry_cascading(n_clicks, enabled, min_conf, max_pos, max_weight,
                          bull_bear, min_bull, min_pbull,
                          tech_wait, rsi_min, rsi_max, min_mom,
                          min_det, min_lmax, max_kurt, min_eta):
    if n_clicks is None:
        return ""
    if broker_instance is None:
        return "❌ Брокер не инициализирован"
    try:
        config = {
            'enabled': 'enabled' in (enabled or []),
            'hawkes_trigger': {
                'bull_to_bear_ratio': float(bull_bear or 1.5),
                'min_bull_expected': float(min_bull or 0.5),
                'min_prob_bull': float(min_pbull or 0.5),
            },
            'technical_confirmation': {
                'wait_minutes': int(tech_wait or 0),
                'rsi_min': int(rsi_min or 50),
                'rsi_max': int(rsi_max or 70),
                'bb_position_min': 0.4,
                'bb_position_max': 0.8,
                'min_momentum_pct': float(min_mom or 0.1),
            },
            'chaos_filter': {
                'min_rqa_DET': float(min_det or 0.25),
                'min_rqa_L_max': int(min_lmax or 4),
                'max_kurtosis': float(max_kurt or 100),
                'min_hawkes_branching_ratio': float(min_eta or 0.3),
            },
            'portfolio_constraints': {
                'max_positions': int(max_pos or 5),
                'max_position_weight_pct': int(max_weight or 20),
                'max_per_sector': 2,
                'max_correlation_with_held': 0.7,
                'min_confidence': float(min_conf or 0.45),
            },
        }
        result = broker_instance.update_advanced_module_config('entry_cascading', config)
        return "✅ Сохранено" if result.get('success') else f"❌ {result.get('error')}"
    except Exception as e:
        return f"❌ Ошибка: {e}"


# 🆕 Callback: сохранение rolling_exit
@app.callback(
    Output("rolling-exit-cfg-status", "children"),
    Input("save-rolling-exit-btn", "n_clicks"),
    [State("re-enabled-check", "value"),
     State("re-min-hold-input", "value"),
     State("re-max-hold-input", "value"),
     State("re-phase-check", "value"),
     State("re-early-h-input", "value"),
     State("re-early-t-input", "value"),
     State("re-mid-h-input", "value"),
     State("re-mid-t-input", "value"),
     State("re-late-h-input", "value"),
     State("re-late-t-input", "value"),
     State("re-force-t-input", "value"),
     State("re-base-sl-input", "value"),
     State("re-kurt-pen-input", "value"),
     State("re-pt-input", "value"),
     State("re-trailing-input", "value")]
)
def save_rolling_exit(n_clicks, enabled, min_hold, max_hold, phase_check,
                       early_h, early_t, mid_h, mid_t, late_h, late_t, force_t,
                       base_sl, kurt_pen, pt, trailing):
    if n_clicks is None:
        return ""
    if broker_instance is None:
        return "❌ Брокер не инициализирован"
    try:
        config = {
            'enabled': 'enabled' in (enabled or []),
            'evaluation_interval_cycles': 1,
            'thresholds_by_hold_time': {
                'early_hours': int(early_h or 4),
                'early_threshold': float(early_t or 0.65),
                'mid_hours': int(mid_h or 24),
                'mid_threshold': float(mid_t or 0.55),
                'late_hours': int(late_h or 72),
                'late_threshold': float(late_t or 0.45),
                'force_threshold': float(force_t or 0.30),
                'max_hold_hours_hard_cap': int(max_hold or 120),
            },
            'hard_stops': {
                'base_stop_loss_pct': float(base_sl or 2.5),
                'kurtosis_penalty_factor': 0.02,
                'kurtosis_baseline': 3,
                'kurtosis_penalty_max': float(kurt_pen or 5.0),
                'profit_taking_threshold_pct': float(pt or 5.0),
                'profit_taking_min_sell_score': 0.40,
            },
            'phase_exit': {
                'enabled': 'enabled' in (phase_check or []),
                'phase1_ratio': 0.5,
                'phase2_ratio': 0.3,
                'phase3_ratio': 0.2,
                'phase2_confirm_cycles': 1,
                'phase3_trailing_atr_mult': float(trailing or 1.5),
            },
            'minimum_hold_hours_before_sell': int(min_hold or 0),
            'min_pnl_pct_for_profit_sell': 0.5,
        }
        result = broker_instance.update_advanced_module_config('rolling_exit', config)
        return "✅ Сохранено" if result.get('success') else f"❌ {result.get('error')}"
    except Exception as e:
        return f"❌ Ошибка: {e}"


# 🆕 Callback: сохранение hawkes
@app.callback(
    Output("hawkes-cfg-status", "children"),
    Input("save-hawkes-btn", "n_clicks"),
    [State("hw-default-thr-input", "value"),
     State("hw-refit-input", "value"),
     State("hw-horizon-input", "value"),
     State("hw-max-iter-input", "value"),
     State("hw-split-input", "value"),
     State("hw-low-mult-input", "value"),
     State("hw-high-mult-input", "value")]
)
def save_hawkes(n_clicks, default_thr, refit, horizon, max_iter,
                 split, low_mult, high_mult):
    if n_clicks is None:
        return ""
    if broker_instance is None:
        return "❌ Брокер не инициализирован"
    try:
        # default_thr в %, конвертируем в долях (0.5% → 0.005)
        config = {
            'event_threshold_pct': float(default_thr or 0.5) / 100,
            'window_size': 4000,
            'refit_interval': int(refit or 100),
            'forecast_horizon_hours': int(horizon or 4),
            'max_iter': int(max_iter or 30),
            'per_ticker_thresholds': {
                'vol_threshold_split': float(split or 0.5),
                'low_vol_multiplier': float(low_mult or 0.5),
                'high_vol_multiplier': float(high_mult or 0.8),
            },
        }
        result = broker_instance.update_advanced_module_config('hawkes', config)
        return "✅ Сохранено" if result.get('success') else f"❌ {result.get('error')}"
    except Exception as e:
        return f"❌ Ошибка: {e}"


# 🆕 Callback: обновление тикеров
@app.callback(
    Output("refresh-tickers-status", "children"),
    Input("refresh-tickers-btn", "n_clicks")
)
def refresh_tickers(n_clicks):
    if n_clicks is None:
        return ""
    try:
        import subprocess
        result = subprocess.run(
            ['python3', '/home/z/my-project/scripts/select_liquid_volatile.py'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            # Читаем обновлённый файл
            with open('config/tickers.json', 'r', encoding='utf-8') as f:
                t = json.load(f)
            n = len(t.get('watchlist', []))
            return f"✅ Обновлено: {n} тикеров в config/tickers.json"
        else:
            return f"❌ Ошибка: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return "❌ Таймаут (300с). Попробуйте запустить вручную."
    except Exception as e:
        return f"❌ Ошибка: {e}"


@app.callback(
    Output("logs-tab-content", "children"),
    Input("logs-tabs", "active_tab")
)
def switch_logs_tab(active_tab):
    """Переключение вкладок внутри страницы логов"""
    if active_tab == "tab-events":
        return dbc.Container([
            # Фильтр логов
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Уровень:", className="form-label"),
                                    dcc.Dropdown(
                                        id='log-level-filter',
                                        options=[
                                            {'label': 'Все', 'value': 'ALL'},
                                            {'label': 'INFO', 'value': 'INFO'},
                                            {'label': 'WARNING', 'value': 'WARNING'},
                                            {'label': 'ERROR', 'value': 'ERROR'},
                                            {'label': 'СДЕЛКИ', 'value': 'TRADES'},
                                            {'label': 'TRAINER', 'value': 'TRAINER'}

                                        ],
                                        value='ALL',
                                        clearable=False
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Компонент:", className="form-label"),
                                    dcc.Dropdown(
                                        id='log-component-filter',
                                        options=[
                                            {'label': 'Все', 'value': 'ALL'},
                                            {'label': 'MAIN', 'value': 'MAIN'},
                                            {'label': 'MOEX_FETCHER', 'value': 'MOEX_FETCHER'},
                                            {'label': 'NEWS_FETCHER_OPT', 'value': 'NEWS_FETCHER_OPT'},
                                            {'label': 'PORTFOLIO', 'value': 'PORTFOLIO'},
                                            {'label': 'RISK_MANAGER', 'value': 'RISK_MANAGER'},
                                            {'label': 'SMART_BROKER', 'value': 'SMART_BROKER'},
                                            {'label': 'TECH_CORE', 'value': 'TECH_CORE'},
                                            {'label': 'TRADER_MODEL', 'value': 'TRADER_MODEL'},
                                            {'label': 'TRAINER', 'value': 'TRAINER'},
                                            {'label': 'LLM_COACH', 'value': 'LLM_COACH'}
                                        ],
                                        value='ALL',
                                        clearable=False
                                    )
                                ], width=4),
                            ])
                        ])
                    ], className="mb-2")
                ])
            ]),
            # Журнал событий
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            "Журнал событий",
                            dbc.Button("🗑️ Очистить", id="clear-logs-btn", color="danger", size="sm", className="float-end")
                        ]),
                        dbc.CardBody([
                            html.Div(id="logs-content",
                                     style={
                                         'height': '400px',
                                         'overflowY': 'auto',
                                         'backgroundColor': '#1a1a1a',
                                         'padding': '10px',
                                         'fontFamily': 'monospace',
                                         'fontSize': '12px'
                                     })
                        ])
                    ])
                ])
            ]),
            dbc.Row([
                dbc.Col([dbc.Card([dbc.CardHeader("Статистика системы"), dbc.CardBody([html.Div(id="system-stats")])])], width=6),
                dbc.Col([dbc.Card([dbc.CardHeader("Статус компонентов"), dbc.CardBody([html.Div(id="components-status")])])], width=6)
            ], className="mt-4")
        ])
    elif active_tab == "tab-models":
        return models_log_layout
    return ""

# Коллбэк логов (новый — из буфера памяти)
@app.callback(
    Output("logs-content", "children"),
    [Input("interval-component", "n_intervals"),
     Input("clear-logs-btn", "n_clicks"),
     Input("log-level-filter", "value"),
     Input("log-component-filter", "value")],
    prevent_initial_call=False
)
def update_logs(n_intervals, clear_clicks, level_filter, component_filter):
    """Обновление содержимого логов из буфера памяти"""
    from utils.logger import get_log_buffer, clear_log_buffer

    ctx_triggered = ctx.triggered_id
    if ctx_triggered == "clear-logs-btn" and clear_clicks:
        clear_log_buffer()
        return html.P("Логи очищены", className="text-muted")

    log_buffer = get_log_buffer()
    lines = list(log_buffer)

    if not lines:
        return html.P("Нет записей в журнале", className="text-muted")

    COMPONENT_COLORS = {
        'SMART_BROKER': '#00ff88', 'RISK_MANAGER': '#ffaa00', 'TECH_CORE': '#00aaff',
        'TRAINER': '#aa00ff', 'MOEX_FETCHER': '#ff8800', 'NEWS_FETCHER_OPT': '#00ccff',
        'PORTFOLIO': '#ff00aa', 'MAIN': '#ffffff', 'SYSTEM': '#cccccc',
        'DEEP_DIAG': '#888888', 'TEST_FIXES': '#888888', 'TEST_8_STAGES': '#888888'
    }

    log_entries = []
    for entry in lines:
        name = entry.get('name', 'SYSTEM')
        level = entry.get('level', 'INFO')
        message = entry.get('message', '')
        timestamp = entry.get('timestamp', '')

        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
        else:
            time_str = str(timestamp)[:8]

        if level_filter == 'TRADES':
            trade_keywords = ['BUY', 'SELL', 'STOP_LOSS', 'TAKE_PROFIT', 'ДНЕВНАЯ ПРИБЫЛЬ', 'КУПЛЕНО', 'ПРОДАНО']
            if not any(kw in message for kw in trade_keywords):
                continue
        elif level_filter != 'ALL' and level != level_filter:
            continue

        if component_filter != 'ALL' and name != component_filter:
            continue

        color = COMPONENT_COLORS.get(name, '#cccccc')
        font_weight = 'normal'
        bg_color = 'transparent'

        if any(kw in message for kw in ['BUY', 'КУПЛЕНО']):
            font_weight = 'bold'; color = '#00ff88'
        elif any(kw in message for kw in ['SELL', 'ПРОДАНО']):
            font_weight = 'bold'; color = '#ff4444'
        elif 'STOP_LOSS' in message:
            font_weight = 'bold'; color = '#ff8800'
        elif 'TAKE_PROFIT' in message:
            font_weight = 'bold'; color = '#ffaa00'
        elif 'ДНЕВНАЯ ПРИБЫЛЬ' in message:
            font_weight = 'bold'; color = '#00ccff'
        elif level in ['ERROR', 'CRITICAL']:
            bg_color = '#330000'

        log_entries.append(
            html.Div([
                html.Span(f"{time_str} ", style={'color': '#888888'}),
                html.Span(f"[{name}] ", style={'color': color, 'fontWeight': 'bold'}),
                html.Span(message, style={'color': color, 'fontWeight': font_weight})
            ], style={
                'margin': '1px 0', 'fontSize': '11px',
                'backgroundColor': bg_color, 'padding': '1px 4px', 'borderRadius': '2px'
            })
        )

    log_entries.reverse()
    return log_entries


# Коллбэк статистики
@app.callback(
    [Output("system-stats", "children"),
     Output("components-status", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_system_stats(n_intervals):
    """Обновление статистики системы с реальными статусами"""
    if broker_instance is None:
        return [html.P("Нет данных"), html.P("Нет данных")]

    try:
        summary = broker_instance.get_portfolio_summary()
        risk_metrics = summary.get('risk_metrics', {})
        reserved_cash = summary.get('reserved_cash', 0)

        stats_html = html.Div([
            html.P(f"🔢 Циклов: {broker_instance.cycle_count}"),
            html.P(f"📈 Сигналов: {len(broker_instance.signals_cache)}"),
            html.P(f"💼 Позиций: {summary.get('positions_count', 0)}"),
            html.P(f"📊 PnL день: {risk_metrics.get('daily_pnl', 0):+,.0f}₽"),
            html.P(f"🎯 Сделок день: {risk_metrics.get('daily_trades', 0)}"),
            html.P(f"🔄 Обучение: {len(broker_instance.model.memory)} опытов"),
            html.P(f"💎 Накоплено: {reserved_cash:+,.0f}₽")
        ])

        model_ok = broker_instance.model is not None
        news_ok = hasattr(broker_instance, 'news_fetcher') and broker_instance.news_fetcher is not None
        tech_ok = hasattr(broker_instance, 'technical_core') and broker_instance.technical_core is not None
        risk_ok = hasattr(broker_instance, 'risk_manager') and broker_instance.risk_manager is not None
        sched_ok = hasattr(broker_instance, 'scheduler') and broker_instance.scheduler is not None
        portf_ok = hasattr(broker_instance, 'portfolio') and broker_instance.portfolio is not None
        trainer_ok = hasattr(broker_instance, 'trainer') and broker_instance.trainer is not None
        trainer_active = trainer_ok and getattr(broker_instance.trainer, 'training_enabled', False)

        components_html = html.Div([
            html.P(f"🤖 AI модель: {'✅' if model_ok else '❌'} {'Активна' if model_ok else 'Недоступна'}"),
            html.P(f"📰 Новости: {'✅' if news_ok else '❌'} {'Активны' if news_ok else 'Недоступны'}"),
            html.P(f"📊 Тех.анализ: {'✅' if tech_ok else '❌'} {'Активен' if tech_ok else 'Недоступен'}"),
            html.P(f"⚖️ Риск-менеджер: {'✅' if risk_ok else '❌'} {'Активен' if risk_ok else 'Недоступен'}"),
            html.P(f"⏰ Планировщик: {'✅' if sched_ok else '❌'} {'Активен' if sched_ok else 'Недоступен'}"),
            html.P(f"💰 Портфель: {'✅' if portf_ok else '❌'} {'Активен' if portf_ok else 'Недоступен'}"),
            html.P(f"🔄 Trainer: {'✅' if trainer_active else '❌'} {'Активен' if trainer_active else 'Остановлен'}")
        ])

        return [stats_html, components_html]

    except Exception as e:
        logger.error(f"Ошибка обновления статистики: {e}")
        return [html.P(f"Ошибка: {e}"), html.P("Ошибка загрузки")]


# 🆕 Коллбэк для новых модулей (Entry F, Rolling D, Hawkes, Chaos)
@app.callback(
    [Output("entry-confirmer-status", "children"),
     Output("rolling-exit-status", "children"),
     Output("hawkes-status", "children"),
     Output("chaos-metrics-status", "children"),
     Output("rolling-exit-positions-table", "children"),
     Output("chaos-metrics-table", "children")],
    [Input("interval-component", "n_intervals")]
)
def update_advanced_modules(n_intervals):
    """Обновление карточек новых модулей"""
    if broker_instance is None:
        return [html.P("Нет данных")] * 6

    try:
        adv = broker_instance.get_advanced_modules_summary()

        # === Entry Confirmer ===
        ec = adv.get('entry_confirmer')
        if ec:
            ec_stats = ec.get('stats', {})
            ec_cfg = ec.get('config', {})
            pc = ec_cfg.get('portfolio_constraints', {})
            cf = ec_cfg.get('chaos_filter', {})
            ht = ec_cfg.get('hawkes_trigger', {})
            tc = ec_cfg.get('technical_confirmation', {})
            phase_dist = ec_stats.get('phase_distribution', {})
            entry_html = html.Div([
                html.P([
                    html.Span(f"{'✅ Активен' if ec['enabled'] else '❌ Отключён'}",
                              className="text-success" if ec['enabled'] else "text-danger"),
                    html.Span(f" | Трекинг: {ec_stats.get('tracked_tickers', 0)} тикеров",
                              className="ms-2 text-muted"),
                ]),
                html.P(f"🎯 Фазы: {dict(phase_dist) if phase_dist else 'нет активных'}"),
                html.Hr(),
                html.P([
                    html.Strong("Hawkes-триггер: "),
                    f"bull/bear ≥ {ht.get('bull_to_bear_ratio', 1.5)}, "
                    f"min_bull ≥ {ht.get('min_bull_expected', 0.5)}, "
                    f"P(bull) ≥ {ht.get('min_prob_bull', 0.5)}"
                ], className="small"),
                html.P([
                    html.Strong("Тех. подтверждение: "),
                    f"RSI [{tc.get('rsi_min', 50)}-{tc.get('rsi_max', 70)}], "
                    f"BB [{tc.get('bb_position_min', 0.4)}-{tc.get('bb_position_max', 0.8)}], "
                    f"wait={tc.get('wait_minutes', 0)}м"
                ], className="small"),
                html.P([
                    html.Strong("Хаос-фильтр: "),
                    f"DET≥{cf.get('min_rqa_DET', 0.25)}, "
                    f"L_max≥{cf.get('min_rqa_L_max', 4)}, "
                    f"kurt≤{cf.get('max_kurtosis', 100)}, "
                    f"η≥{cf.get('min_hawkes_branching_ratio', 0.3)}"
                ], className="small"),
                html.P([
                    html.Strong("Портфель: "),
                    f"max_pos={pc.get('max_positions', 5)}, "
                    f"max_weight={pc.get('max_position_weight_pct', 20)}%, "
                    f"max_per_sector={pc.get('max_per_sector', 2)}, "
                    f"min_conf={pc.get('min_confidence', 0.45)}"
                ], className="small"),
                html.P(f"⏱ Кулдаун: {ec_cfg.get('cooldown_seconds', 60)}с",
                       className="small text-muted"),
            ])
        else:
            entry_html = html.P("Модуль не инициализирован", className="text-muted")

        # === Rolling Exit ===
        re_data = adv.get('rolling_exit')
        if re_data:
            re_stats = re_data.get('stats', {})
            re_cfg = re_data.get('config', {})
            th = re_cfg.get('thresholds_by_hold_time', {})
            hs = re_cfg.get('hard_stops', {})
            ph = re_cfg.get('phase_exit', {})
            phase_dist = re_stats.get('phase_distribution', {})
            rolling_html = html.Div([
                html.P([
                    html.Span(f"{'✅ Активен' if re_data['enabled'] else '❌ Отключён'}",
                              className="text-success" if re_data['enabled'] else "text-danger"),
                    html.Span(f" | Позиций: {re_stats.get('active_positions', 0)}",
                              className="ms-2 text-muted"),
                ]),
                html.P(f"🔄 Фазы: {dict(phase_dist) if phase_dist else 'нет активных'}"),
                html.Hr(),
                html.P([
                    html.Strong("Пороги по hold_time: "),
                    f"early({th.get('early_hours', 4)}ч)={th.get('early_threshold', 0.65)}, "
                    f"mid({th.get('mid_hours', 24)}ч)={th.get('mid_threshold', 0.55)}, "
                    f"late({th.get('late_hours', 72)}ч)={th.get('late_threshold', 0.45)}, "
                    f"force={th.get('force_threshold', 0.30)}"
                ], className="small"),
                html.P([
                    html.Strong("Hard stops: "),
                    f"base_SL={hs.get('base_stop_loss_pct', 2.5)}%, "
                    f"kurt_pen≤{hs.get('kurtosis_penalty_max', 5.0)}%, "
                    f"PT={hs.get('profit_taking_threshold_pct', 5.0)}%"
                ], className="small"),
                html.P([
                    html.Strong("Фазовый выход: "),
                    f"P1={ph.get('phase1_ratio', 0.5)*100:.0f}%, "
                    f"P2={ph.get('phase2_ratio', 0.3)*100:.0f}%, "
                    f"P3={ph.get('phase3_ratio', 0.2)*100:.0f}%, "
                    f"trailing_ATR×{ph.get('phase3_trailing_atr_mult', 1.5)}"
                ], className="small"),
                html.P(f"⏱ Min hold: {re_cfg.get('minimum_hold_hours_before_sell', 0)}ч",
                       className="small text-muted"),
            ])
        else:
            rolling_html = html.P("Модуль не инициализирован", className="text-muted")

        # === Hawkes ===
        hawkes_data = adv.get('hawkes')
        if hawkes_data:
            hs = hawkes_data.get('stats', {})
            per_ticker = hawkes_data.get('per_ticker', [])
            hawkes_html = html.Div([
                html.P([
                    html.Span(f"📊 Трекинг: {hs.get('tickers_tracked', 0)} тикеров",
                              className="text-info"),
                    html.Span(f" | Обучено: {hs.get('tickers_fitted', 0)}",
                              className="ms-2 text-muted"),
                    html.Span(f" | Per-ticker: {hawkes_data.get('per_ticker_count', 0)}",
                              className="ms-2 text-muted"),
                ]),
                html.P([
                    html.Span(f"🟢 Bullish events: {hs.get('total_bullish_events', 0)}",
                              className="text-success"),
                    html.Span(f" | 🔴 Bearish events: {hs.get('total_bearish_events', 0)}",
                              className="text-danger ms-2"),
                ]),
                html.P(f"⚙ Default threshold: {hawkes_data.get('default_threshold', 0.005)}",
                       className="small text-muted"),
                html.Hr(),
                html.P(html.Strong("Топ-тикеры по η (branching ratio):"),
                       className="small mb-1"),
                html.Div([
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Тикер", className="small"),
                            html.Th("Vol %", className="small"),
                            html.Th("Порог", className="small"),
                            html.Th("η_bull", className="small"),
                            html.Th("η_bear", className="small"),
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(t.get('ticker', ''), className="small"),
                                html.Td(f"{t.get('volatility_pct', 0):.2f}", className="small"),
                                html.Td(f"{t.get('threshold', 0):.4f}", className="small"),
                                html.Td(f"{t.get('eta_bull', 0):.2f}", className="small text-success"),
                                html.Td(f"{t.get('eta_bear', 0):.2f}", className="small text-danger"),
                            ]) for t in per_ticker[:8]
                        ])
                    ], className="table table-sm table-hover")
                ]) if per_ticker else html.P("Нет обученных тикеров",
                                              className="text-muted small"),
            ])
        else:
            hawkes_html = html.P("Hawkes не инициализирован", className="text-muted")

        # === Chaos Metrics ===
        chaos_data = adv.get('chaos_metrics')
        if chaos_data and isinstance(chaos_data, dict) and chaos_data.get('total_tickers', 0) > 0:
            total = chaos_data.get('total_tickers', 0)
            top = chaos_data.get('top_by_volatility', [])
            chaos_status_html = html.Div([
                html.P([
                    html.Span(f"🌀 Рассчитано: {total} тикеров",
                              className="text-success"),
                    html.Span(f" | Метрик: Hurst, D₂, RQA (RR/DET/L_max/LAM), kurtosis, ATR",
                              className="ms-2 text-muted small"),
                ]),
                html.Hr(),
                html.P(html.Strong("Топ-5 по волатильности:"), className="small mb-1"),
                html.Div([
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Тикер", className="small"),
                            html.Th("Vol %", className="small"),
                            html.Th("Hurst", className="small"),
                            html.Th("D₂", className="small"),
                            html.Th("DET", className="small"),
                            html.Th("L_max", className="small"),
                            html.Th("Kurt", className="small"),
                            html.Th("ATR %", className="small"),
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(t.get('ticker', ''), className="small"),
                                html.Td(f"{t.get('volatility_pct', 0):.2f}", className="small"),
                                html.Td(_hurst_color(t.get('hurst', 0.5)), className="small"),
                                html.Td(f"{t.get('fractal_dim', 0):.2f}", className="small"),
                                html.Td(f"{t.get('rqa_DET', 0):.2f}", className="small"),
                                html.Td(str(t.get('rqa_L_max', 0)), className="small"),
                                html.Td(_kurt_color(t.get('kurtosis', 0)), className="small"),
                                html.Td(f"{t.get('atr_pct', 0):.2f}", className="small"),
                            ]) for t in top[:5]
                        ])
                    ], className="table table-sm table-hover")
                ]),
            ])
        else:
            chaos_status_html = html.P("Хаос-метрики не рассчитаны",
                                        className="text-muted")

        # === Rolling Exit Positions Table ===
        re_positions = re_data.get('active_positions', {}) if re_data else {}
        if re_positions:
            re_rows = []
            for ticker, info in sorted(re_positions.items(),
                                         key=lambda x: -x[1].get('hold_hours', 0)):
                phase = info.get('phase', 1)
                phase_class = {1: 'badge bg-success', 2: 'badge bg-warning',
                               3: 'badge bg-danger', 0: 'badge bg-secondary'}.get(phase, 'badge bg-secondary')
                phase_name = {1: 'Full', 2: '50% sold', 3: 'Trailing', 0: 'Closed'}.get(phase, str(phase))
                score = info.get('last_sell_score', 0)
                score_class = "text-success" if score < 0.4 else "text-warning" if score < 0.6 else "text-danger"
                re_rows.append(html.Tr([
                    html.Td(html.Strong(ticker)),
                    html.Td(html.Span(phase_name, className=phase_class)),
                    html.Td(f"{info.get('hold_hours', 0):.1f}ч"),
                    html.Td(f"{info.get('entry_price', 0):.2f}"),
                    html.Td(f"{info.get('stop_loss_price', 0):.2f}"),
                    html.Td(f"{info.get('dynamic_target', 0):.2f}"),
                    html.Td(html.Span(f"{score:.2f}", className=score_class)),
                    html.Td(f"H={info.get('hurst', 0.5):.2f}"),
                    html.Td(f"DET={info.get('rqa_det', 0):.2f}"),
                    html.Td(f"L={info.get('rqa_l_max', 0)}"),
                    html.Td(f"k={info.get('kurtosis', 0):.0f}"),
                ]))
            re_table = html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Тикер"), html.Th("Фаза"), html.Th("Hold"),
                        html.Th("Entry"), html.Th("Stop"), html.Th("Target"),
                        html.Th("Score"), html.Th("Hurst"), html.Th("DET"),
                        html.Th("L_max"), html.Th("Kurt"),
                    ])),
                    html.Tbody(re_rows)
                ], className="table table-sm table-hover table-striped")
            ], className="table-responsive")
        else:
            re_table = html.P("Нет активных позиций в rolling exit",
                               className="text-muted text-center p-3")

        # === Chaos Metrics Full Table ===
        if chaos_data and chaos_data.get('all_tickers'):
            chaos_rows = []
            for t in chaos_data['all_tickers'][:20]:  # топ-20
                hurst = t.get('hurst', 0.5)
                kurt = t.get('kurtosis', 0)
                regime = _classify_hurst(hurst)
                chaos_rows.append(html.Tr([
                    html.Td(html.Strong(t.get('ticker', ''))),
                    html.Td(f"{t.get('last_price', 0):.2f}"),
                    html.Td(f"{t.get('volatility_pct', 0):.2f}"),
                    html.Td(_hurst_color(hurst)),
                    html.Td(regime),
                    html.Td(f"{t.get('fractal_dim', 0):.2f}"),
                    html.Td(f"{t.get('rqa_DET', 0):.2f}"),
                    html.Td(str(t.get('rqa_L_max', 0))),
                    html.Td(f"{t.get('rqa_LAM', 0):.2f}"),
                    html.Td(_kurt_color(kurt)),
                    html.Td(f"{t.get('atr_pct', 0):.2f}"),
                    html.Td(str(t.get('n_points', 0))),
                ]))
            chaos_table = html.Div([
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Тикер"), html.Th("Цена"), html.Th("Vol %"),
                        html.Th("Hurst"), html.Th("Режим"), html.Th("D₂"),
                        html.Th("DET"), html.Th("L_max"), html.Th("LAM"),
                        html.Th("Kurt"), html.Th("ATR %"), html.Th("N"),
                    ])),
                    html.Tbody(chaos_rows)
                ], className="table table-sm table-hover table-striped")
            ], className="table-responsive")
        else:
            chaos_table = html.P("Хаос-метрики не рассчитаны", className="text-muted")

        return [entry_html, rolling_html, hawkes_html, chaos_status_html,
                re_table, chaos_table]

    except Exception as e:
        logger.error(f"Ошибка обновления новых модулей: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return [html.P(f"Ошибка: {e}", className="text-danger")] * 6


def _hurst_color(h):
    """Цветной Херст."""
    if h > 0.55:
        return html.Span(f"{h:.3f}", className="text-success")  # персистентный
    elif h < 0.45:
        return html.Span(f"{h:.3f}", className="text-danger")  # антиперсистентный
    return html.Span(f"{h:.3f}", className="text-muted")


def _kurt_color(k):
    """Цветной kurtosis."""
    if k > 50:
        return html.Span(f"{k:.0f}", className="text-danger")  # экстремальные хвосты
    elif k > 10:
        return html.Span(f"{k:.0f}", className="text-warning")
    return html.Span(f"{k:.0f}", className="text-success")


def _classify_hurst(h):
    """Классификация режима по Херсту."""
    if h > 0.55:
        return html.Span("персист.", className="text-success")
    elif h < 0.45:
        return html.Span("антиперс.", className="text-danger")
    return html.Span("нейтр.", className="text-muted")
# Часть 4 из 4: коллбэки графиков, портфеля, утилиты

# Коллбэк графика цены и объема
@app.callback(
    Output("price-volume-chart", "figure"),
    [Input("ticker-selector", "value"),
     Input("period-selector", "value"),
     Input("interval-selector", "value"),
     Input("interval-component", "n_intervals")]
)
def update_price_volume_chart(ticker, period, interval_str, n_intervals):
    """Обновление графика цены и объема"""
    try:
        if broker_instance is None:
            return dashboard_viz._create_empty_chart("Система не инициализирована")

        interval_map = {"1min": 1, "5min": 10, "15min": 10, "30min": 10, "1h": 60, "1d": 24}
        period_count_map = {"1d": 24, "1w": 7 * 24, "1m": 30 * 24, "3m": 90 * 24, "6m": 180 * 24}

        interval = interval_map.get(interval_str, 60)
        count = period_count_map.get(period, 100)

        if interval == 24:
            count = min(count // 24, 365)

        try:
            moex = broker_instance.moex
            candles_data = moex.get_candles(ticker=ticker, interval=interval, count=count)
        except Exception as e:
            logger.error(f"Ошибка получения данных {ticker}: {e}")
            candles_data = None

        if candles_data is None or candles_data.empty:
            base_prices = {
                'SBER': 280, 'GAZP': 160, 'LKOH': 7500, 'YNDX': 3500, 'VTBR': 0.025,
                'ROSN': 580, 'GMKN': 16000, 'NVTK': 1800, 'TCSG': 4500, 'MOEX': 150,
                'ALRS': 100, 'PHOR': 2800
            }
            base_price = base_prices.get(ticker, 100)

            if interval == 24:
                dates = pd.date_range(end=datetime.now(), periods=count, freq='D')
            else:
                dates = pd.date_range(end=datetime.now(), periods=count, freq='h')

            trend = np.random.uniform(-0.01, 0.01, count).cumsum()
            prices = base_price * (1 + trend)
            volatility = base_price * 0.02
            noise = np.random.randn(count) * volatility
            final_prices = prices + noise

            candles_data = pd.DataFrame({
                'Open': final_prices - np.random.rand(count) * (base_price * 0.01),
                'High': final_prices + np.random.rand(count) * (base_price * 0.015),
                'Low': final_prices - np.random.rand(count) * (base_price * 0.01),
                'Close': final_prices,
                'Volume': np.random.randint(10000, 1000000, count)
            }, index=dates)

        return dashboard_viz.create_price_volume_chart(candles_data, ticker)

    except Exception as e:
        logger.error(f"Ошибка обновления графика цены: {e}")
        return dashboard_viz._create_empty_chart(f"Ошибка: {str(e)}")


# Коллбэк графика индикаторов
@app.callback(
    Output("indicators-chart", "figure"),
    [Input("ticker-selector", "value"),
     Input("period-selector", "value"),
     Input("interval-selector", "value"),
     Input("interval-component", "n_intervals")]
)
def update_indicators_chart(ticker, period, interval_str, n_intervals):
    """График индикаторов из свечных данных"""
    try:
        try:
            import talib
            HAS_TALIB = True
        except ImportError:
            HAS_TALIB = False

        if not HAS_TALIB:
            return dashboard_viz._create_empty_chart(
                "TA-Lib не установлен.\nУстановите: pip install TA-Lib"
            )

        if broker_instance is None:
            return dashboard_viz._create_empty_chart("Система не инициализирована")

        interval_map = {"1min": 1, "5min": 10, "15min": 10, "30min": 10, "1h": 60, "1d": 24}
        interval = interval_map.get(interval_str, 60)
        count = 50

        candles_data = broker_instance.moex.get_candles(ticker=ticker, interval=interval, count=count)

        if candles_data is None or candles_data.empty:
            return dashboard_viz._create_empty_chart(f"Нет данных для {ticker}")

        if 'Close' in candles_data.columns:
            close_prices = candles_data['Close'].values
        elif 'close' in candles_data.columns:
            close_prices = candles_data['close'].values
        else:
            close_prices = candles_data['Open'].values

        if len(close_prices) < 20:
            return dashboard_viz._create_empty_chart(f"Мало данных для расчета индикаторов")

        indicators_data = {'prices': close_prices.tolist()}

        # RSI
        try:
            rsi_values = talib.RSI(close_prices, timeperiod=14)
            indicators_data['rsi'] = rsi_values.tolist()
        except:
            indicators_data['rsi'] = (50 + 20 * np.sin(np.linspace(0, 4 * np.pi, len(close_prices)))).tolist()

        # MACD
        try:
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            indicators_data['macd'] = macd.tolist()
            indicators_data['macd_signal'] = macd_signal.tolist()
            indicators_data['macd_hist'] = macd_hist.tolist()
        except:
            indicators_data['macd'] = (np.sin(np.linspace(0, 6 * np.pi, len(close_prices))) * 2).tolist()
            indicators_data['macd_signal'] = (np.sin(np.linspace(0, 6 * np.pi, len(close_prices)) - 0.5) * 1.8).tolist()
            indicators_data['macd_hist'] = (np.random.randn(len(close_prices)) * 0.5).tolist()

        # Bollinger Bands
        try:
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20)
            indicators_data['bb_upper'] = upper.tolist()
            indicators_data['bb_middle'] = middle.tolist()
            indicators_data['bb_lower'] = lower.tolist()
        except:
            base = np.mean(close_prices)
            indicators_data['bb_upper'] = (base + base * 0.05 + np.sin(np.linspace(0, 4 * np.pi, len(close_prices))) * base * 0.02).tolist()
            indicators_data['bb_middle'] = (base + np.sin(np.linspace(0, 4 * np.pi, len(close_prices))) * base * 0.01).tolist()
            indicators_data['bb_lower'] = (base - base * 0.05 + np.sin(np.linspace(0, 4 * np.pi, len(close_prices))) * base * 0.02).tolist()

        return dashboard_viz.create_indicators_chart(indicators_data, ticker)

    except Exception as e:
        logger.error(f"Ошибка обновления графика индикаторов: {e}")
        return dashboard_viz._create_empty_chart(f"Ошибка: {str(e)}")


# Коллбэк графика сентимента
@app.callback(
    Output("sentiment-chart", "figure"),
    [Input("ticker-selector", "value"),
     Input("interval-component", "n_intervals")]
)
def update_sentiment_chart(ticker, n_intervals):
    """Обновление графика новостного сентимента"""
    try:
        if broker_instance is None:
            return dashboard_viz._create_empty_chart("Система не инициализирована")

        sentiment_data = []

        if hasattr(broker_instance, 'get_sentiment_history'):
            all_sentiment = broker_instance.get_sentiment_history(limit=50)
            for item in all_sentiment:
                item_ticker = item.get('ticker', 'MARKET')
                if item_ticker == ticker or (ticker == 'MOEX' and item_ticker == 'MARKET'):
                    sentiment_data.append(item)
            if not sentiment_data and ticker != 'MOEX':
                sentiment_data = [item for item in all_sentiment if item.get('ticker') == 'MARKET']

        if not sentiment_data:
            now = datetime.now()
            base_sentiment = 0.2 if ticker == 'SBER' else 0.0
            sentiment_data = [
                {
                    'timestamp': (now - timedelta(hours=i)).isoformat(),
                    'sentiment': base_sentiment + 0.1 * np.sin(i * 0.5),
                    'source': 'TEST',
                    'ticker': ticker if ticker != 'MOEX' else 'MARKET'
                }
                for i in range(20, 0, -1)
            ]

        if not sentiment_data:
            return dashboard_viz._create_empty_chart(f"Нет данных по сентименту для {ticker}")

        return dashboard_viz.create_sentiment_chart(sentiment_data)

    except Exception as e:
        logger.error(f"Ошибка обновления графика сентимента: {e}")
        return dashboard_viz._create_empty_chart(f"Ошибка: {str(e)}")


# Коллбэки страницы портфеля
@app.callback(
    [Output("detailed-positions-table", "children"),
     Output("sector-pie-chart", "figure"),
     Output("portfolio-value-chart", "figure"),
     Output("trade-history-table", "children")],
    [Input("interval-component", "n_intervals"),
     Input("nav-portfolio", "n_clicks")]
)
def update_portfolio_page(n_intervals, nav_clicks):
    """Обновление данных на странице портфеля"""
    if broker_instance is None:
        empty_chart = dashboard_viz._create_empty_chart("Система не инициализирована")
        return ["Нет данных", empty_chart, empty_chart, "Нет данных"]

    try:
        summary = get_portfolio_summary_safe()
        positions = summary.get('positions', [])
        positions_table = create_real_positions_table(positions)
        sector_chart = create_real_sector_chart(positions)
        value_chart = create_real_portfolio_value_chart(summary)
        trade_history = get_real_trade_history()
        trade_table = create_real_trade_history_table(trade_history)
        return [positions_table, sector_chart, value_chart, trade_table]

    except Exception as e:
        logger.error(f"Ошибка обновления страницы портфеля: {e}")
        empty_chart = dashboard_viz._create_empty_chart(f"Ошибка загрузки")
        return ["Ошибка загрузки", empty_chart, empty_chart, "Ошибка загрузки"]


def get_portfolio_summary_safe():
    """Безопасное получение сводки портфеля"""
    try:
        if broker_instance and hasattr(broker_instance, 'get_portfolio_summary'):
            summary = broker_instance.get_portfolio_summary()
            if 'positions' in summary and broker_instance.portfolio:
                try:
                    with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
                        portfolio_data = json.load(f)
                        saved_positions = portfolio_data.get('positions', {})
                    for i, pos in enumerate(summary['positions']):
                        ticker = pos.get('ticker')
                        if ticker and ticker in saved_positions:
                            saved_strategy = saved_positions[ticker].get('strategy')
                            if saved_strategy and pos.get('strategy', 'unknown') == 'unknown':
                                summary['positions'][i]['strategy'] = saved_strategy
                except:
                    pass
            return summary
        return get_portfolio_summary_fallback()
    except:
        return get_portfolio_summary_fallback()


def get_portfolio_summary_fallback():
    """Резервный метод получения сводки портфеля"""
    try:
        portfolio = broker_instance.portfolio if broker_instance else None
        if not portfolio:
            return {"positions": []}
        prices = broker_instance._get_current_prices() if hasattr(broker_instance, '_get_current_prices') else {}
        positions_detail = []
        total_value = 0
        for ticker, pos in portfolio.positions.items():
            current_price = prices.get(ticker, pos.get('avg_price', 0))
            quantity = pos.get('qty', 0)
            avg_price = pos.get('avg_price', 0)
            position_value = quantity * current_price
            total_value += position_value
            positions_detail.append({
                'ticker': ticker, 'quantity': quantity, 'avg_price': avg_price,
                'current_price': current_price, 'position_value': position_value,
                'pnl': (current_price - avg_price) * quantity,
                'pnl_percent': ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0,
                'weight': (position_value / total_value * 100) if total_value > 0 else 0,
                'buy_time': pos.get('buy_time'), 'strategy': pos.get('strategy', 'unknown')
            })
        return {
            'positions': positions_detail,
            'total_value': total_value + (portfolio.cash if hasattr(portfolio, 'cash') else 0),
            'cash': portfolio.cash if hasattr(portfolio, 'cash') else 0,
            'positions_count': len(portfolio.positions),
            'initial_capital': portfolio.initial_capital if hasattr(portfolio, 'initial_capital') else 10000
        }
    except:
        return {"positions": []}


def create_real_positions_table(positions):
    """Создание таблицы позиций"""
    if not positions:
        return html.Div([
            html.P("Нет открытых позиций", className="text-muted text-center"),
            html.Small("Система ожидает торговых сигналов", className="text-muted d-block text-center")
        ], className="p-4")

    positions_sorted = sorted(positions, key=lambda x: x.get('position_value', 0), reverse=True)

    header = html.Thead(html.Tr([
        html.Th("Тикер"), html.Th("Стратегия"), html.Th("Кол-во"), html.Th("Ср. цена"),
        html.Th("Тек. цена"), html.Th("Стоимость"), html.Th("PnL ₽"), html.Th("PnL %"),
        html.Th("Вес"), html.Th("Дней")
    ]))

    strategy_colors = {
        'news_aggressive': 'badge bg-danger', 'momentum': 'badge bg-warning',
        'mean_reversion': 'badge bg-info', 'balanced': 'badge bg-primary',
        'conservative': 'badge bg-success', 'unknown': 'badge bg-secondary',
        'cascading_entry': 'badge bg-info', 'price_prediction': 'badge bg-primary',
        'swing': 'badge bg-warning', 'breakout': 'badge bg-danger',
        'tech_conservative': 'badge bg-success',
    }

    rows = []
    for pos in positions_sorted:
        ticker = pos.get('ticker', 'N/A')
        strategy = pos.get('strategy', 'unknown')
        days_held = 0
        buy_time = pos.get('buy_time')
        if buy_time:
            try:
                if isinstance(buy_time, (int, float)):
                    buy_dt = datetime.fromtimestamp(buy_time)
                elif isinstance(buy_time, str):
                    buy_dt = datetime.fromisoformat(buy_time.replace('Z', '+00:00'))
                else:
                    buy_dt = datetime.now()
                days_held = max(0, (datetime.now() - buy_dt).days)
            except:
                days_held = 0

        pnl = pos.get('pnl', 0)
        pnl_percent = pos.get('pnl_percent', 0)
        pnl_class = "text-success" if pnl >= 0 else "text-danger"
        strategy_class = strategy_colors.get(strategy, 'badge bg-secondary')

        rows.append(html.Tr([
            html.Td(html.Strong(ticker)),
            html.Td(html.Span(strategy, className=f"{strategy_class} badge-sm")),
            html.Td(f"{pos.get('quantity', 0):,}"),
            html.Td(f"{pos.get('avg_price', 0):.2f}"),
            html.Td(f"{pos.get('current_price', 0):.2f}"),
            html.Td(f"{pos.get('position_value', 0):,.0f} ₽"),
            html.Td(html.Span(f"{pnl:+,.0f} ₽", className=pnl_class)),
            html.Td(html.Span(f"{pnl_percent:+.1f}%", className="text-success" if pnl_percent >= 0 else "text-danger")),
            html.Td(f"{pos.get('weight', 0):.1f}%"),
            html.Td(f"{days_held}")
        ]))

    total_positions_value = sum(p.get('position_value', 0) for p in positions_sorted)
    total_pnl = sum(p.get('pnl', 0) for p in positions_sorted)
    total_pnl_class = "text-success" if total_pnl >= 0 else "text-danger"

    summary = html.Div([
        html.Hr(),
        html.Div([
            html.Span("Всего позиций: ", className="text-muted"),
            html.Strong(f"{len(positions)}", className="ms-2"),
            html.Span("Общая стоимость: ", className="text-muted ms-4"),
            html.Strong(f"{total_positions_value:,.0f} ₽", className="ms-2"),
            html.Span("Общий PnL: ", className="text-muted ms-4"),
            html.Strong(f"{total_pnl:+,.0f} ₽", className=f"ms-2 {total_pnl_class}")
        ], className="d-flex justify-content-between")
    ])

    return html.Div([html.Table([header, html.Tbody(rows)], className="table table-sm table-hover table-striped"), summary], className="table-responsive")


def create_real_sector_chart(positions):
    """График распределения по секторам"""
    if not positions:
        return dashboard_viz._create_empty_chart("Нет открытых позиций")

    try:
        sector_info = load_sector_info()
        sector_values = {}
        for pos in positions:
            ticker = pos.get('ticker', '')
            position_value = pos.get('position_value', 0)
            sector = sector_info['sectors'].get(ticker, 'Не определен')
            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += position_value

        total_value = sum(sector_values.values())
        filtered_sectors = {}
        other_value = 0
        for sector, value in sector_values.items():
            if total_value > 0 and (value / total_value) >= 0.01:
                filtered_sectors[sector] = value
            else:
                other_value += value
        if other_value > 0:
            filtered_sectors['Другое'] = other_value

        if not filtered_sectors:
            return dashboard_viz._create_empty_chart("Недостаточно данных по секторам")

        sectors = list(filtered_sectors.keys())
        values = list(filtered_sectors.values())
        sector_colors = sector_info['sector_colors']
        colors = [sector_colors.get(s, '#cccccc') for s in sectors]

        fig = go.Figure(data=[go.Pie(
            labels=sectors, values=values, hole=0.4, marker=dict(colors=colors),
            textinfo='label+percent', textposition='inside',
            hovertemplate="<b>%{label}</b><br>Стоимость: %{value:,.0f}₽<br>Доля: %{percent}<extra></extra>"
        )])

        fig.update_layout(
            title=dict(text='📊 Распределение по секторам', font=dict(size=16, color='white')),
            template='plotly_dark', height=400, margin=dict(l=20, r=20, t=60, b=20),
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        return fig

    except Exception as e:
        logger.error(f"Ошибка создания графика секторов: {e}")
        return dashboard_viz._create_empty_chart("Ошибка загрузки секторов")


def load_sector_info():
    """Загрузка информации о секторах"""
    try:
        with open('config/ticker_sectors.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            'sectors': {
                'SBER': 'Финансы', 'VTBR': 'Финансы', 'GAZP': 'Нефть и газ',
                'LKOH': 'Нефть и газ', 'ROSN': 'Нефть и газ', 'GMKN': 'Металлургия',
                'MTSS': 'Телеком', 'MGNT': 'Потребительские товары', 'PHOR': 'Химия', 'IRAO': 'Энергетика'
            },
            'sector_colors': {
                'Финансы': '#00ff88', 'Нефть и газ': '#0088ff', 'Металлургия': '#ff4444',
                'Телеком': '#ffaa00', 'Потребительские товары': '#00cc66', 'Химия': '#00aaff',
                'Энергетика': '#ff8800', 'Не определен': '#cccccc', 'Другое': '#999999'
            }
        }


def create_real_portfolio_value_chart(summary):
    """График стоимости портфеля"""
    try:
        history_data = load_portfolio_history()
        if not history_data:
            history_data = [{
                'timestamp': datetime.now().isoformat(),
                'total_value': summary.get('total_value', 0),
                'cash': summary.get('cash', 0),
                'positions_value': summary.get('total_value', 0) - summary.get('cash', 0)
            }]

        dates, total_values, cash_values, positions_values = [], [], [], []
        for point in history_data[-30:]:
            try:
                dates.append(datetime.fromisoformat(point.get('timestamp', '').replace('Z', '+00:00')) if point.get('timestamp') else datetime.now())
                total_values.append(point.get('total_value', 0))
                cash_values.append(point.get('cash', 0))
                positions_values.append(point.get('positions_value', 0))
            except:
                continue

        if not dates:
            return dashboard_viz._create_empty_chart("Нет исторических данных")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=total_values, mode='lines+markers', name='Общая стоимость',
                                 line=dict(color='#00ff88', width=3), marker=dict(size=6)))
        if any(v > 0 for v in positions_values):
            fig.add_trace(go.Scatter(x=dates, y=positions_values, mode='lines', name='Стоимость позиций',
                                     line=dict(color='#0088ff', width=2, dash='dash'),
                                     fill='tonexty', fillcolor='rgba(0, 136, 255, 0.1)'))
        if any(v > 0 for v in cash_values):
            fig.add_trace(go.Scatter(x=dates, y=cash_values, mode='lines', name='Кэш',
                                     line=dict(color='#ffaa00', width=2, dash='dot'),
                                     fill='tonexty', fillcolor='rgba(255, 170, 0, 0.1)'))

        initial_capital = summary.get('initial_capital', total_values[0] if total_values else 0)
        fig.update_layout(
            title=dict(text='📈 Динамика стоимости портфеля', font=dict(size=16, color='white')),
            xaxis_title="Дата", yaxis_title="Стоимость, ₽", template='plotly_dark',
            hovermode='x unified', showlegend=True, height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        if initial_capital > 0:
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="#ff4444",
                          annotation_text=f"Начальный капитал: {initial_capital:,.0f}₽",
                          annotation_position="bottom right", annotation_font=dict(size=10))
        current_value = total_values[-1] if total_values else 0
        fig.add_hline(y=current_value, line_dash="dot", line_color="#00ff88",
                      annotation_text=f"Текущая: {current_value:,.0f}₽",
                      annotation_position="top right", annotation_font=dict(size=10))
        fig.update_yaxes(tickprefix="₽", tickformat=",")
        fig.update_xaxes(tickformat="%d.%m")
        return fig

    except Exception as e:
        logger.error(f"Ошибка создания графика стоимости: {e}")
        return dashboard_viz._create_empty_chart("Ошибка создания графика")


def load_portfolio_history():
    """Загрузка истории портфеля"""
    for file_path in ['data/portfolio_history.json', 'data/daily_report.json']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif 'history' in data:
                    return data['history']
        except:
            continue
    return create_history_from_trades()


def create_history_from_trades():
    """Создание истории из сделок"""
    try:
        if not broker_instance or not hasattr(broker_instance.portfolio, 'trade_history'):
            return []
        trades = broker_instance.portfolio.trade_history
        if not trades:
            return []
        history = []
        current_cash = broker_instance.portfolio.initial_capital
        positions_value = 0
        for trade in sorted(trades, key=lambda x: x.get('timestamp', '')):
            if trade.get('action') == 'BUY':
                current_cash -= trade.get('cost', 0)
                positions_value += trade.get('cost', 0)
            elif trade.get('action') == 'SELL':
                current_cash += trade.get('revenue', 0)
                positions_value -= trade.get('revenue', 0)
            history.append({
                'timestamp': trade.get('timestamp', ''),
                'total_value': current_cash + positions_value,
                'cash': current_cash, 'positions_value': positions_value
            })
        return history[-30:]
    except:
        return []


def get_real_trade_history():
    """Получение реальной истории сделок"""
    try:
        if broker_instance and hasattr(broker_instance.portfolio, 'trade_history'):
            return broker_instance.portfolio.trade_history[-50:]
        elif broker_instance and hasattr(broker_instance.portfolio, 'get_trade_history_summary'):
            return broker_instance.portfolio.get_trade_history_summary(limit=50)
        else:
            with open('data/portfolio_state.json', 'r', encoding='utf-8') as f:
                return json.load(f).get('trade_history', [])[-50:]
    except:
        return []


def create_real_trade_history_table(trades):
    """Создание таблицы истории сделок"""
    if not trades:
        return html.Div([
            html.P("Нет истории сделок", className="text-muted text-center"),
            html.Small("Сделки появятся после начала торговли", className="text-muted d-block text-center")
        ], className="p-4")

    trades_sorted = sorted(trades, key=lambda x: x.get('timestamp', ''), reverse=True)

    header = html.Thead(html.Tr([
        html.Th("Время"), html.Th("Тикер"), html.Th("Действие"), html.Th("Кол-во"),
        html.Th("Цена"), html.Th("Сумма"), html.Th("PnL"), html.Th("Стратегия")
    ]))

    action_map = {
        'BUY': ("badge bg-success", "↗"), 'SELL': ("badge bg-danger", "↘"),
        'STOP_LOSS': ("badge bg-warning", "⛔"), 'TAKE_PROFIT': ("badge bg-info", "✅")
    }
    strategy_colors = {
        'news_aggressive': 'badge-sm bg-danger', 'momentum': 'badge-sm bg-warning',
        'mean_reversion': 'badge-sm bg-info', 'balanced': 'badge-sm bg-primary',
        'conservative': 'badge-sm bg-success', 'unknown': 'badge-sm bg-secondary'
    }

    rows = []
    for trade in trades_sorted[:20]:
        timestamp = trade.get('timestamp', '')
        time_str = ''
        if timestamp:
            try:
                if isinstance(timestamp, (int, float)):
                    dt = datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = datetime.now()
                time_str = dt.strftime('%d.%m %H:%M')
            except:
                time_str = str(timestamp)[:16]

        action = trade.get('action', '')
        action_class, action_icon = action_map.get(action, ("badge bg-secondary", "➡"))
        pnl = trade.get('pnl', 0)
        pnl_text = f"{pnl:+,.0f} ₽" if pnl != 0 else "-"
        amount = trade.get('cost', trade.get('revenue', 0))
        strategy = trade.get('strategy', 'unknown')
        strategy_class = strategy_colors.get(strategy, 'badge-sm bg-secondary')

        rows.append(html.Tr([
            html.Td(time_str), html.Td(html.Strong(trade.get('ticker', ''))),
            html.Td(html.Span(f"{action_icon} {action}", className=f"{action_class}")),
            html.Td(f"{trade.get('quantity', 0):,}"), html.Td(f"{trade.get('price', 0):.2f}"),
            html.Td(f"{amount:,.0f} ₽"),
            html.Td(html.Span(pnl_text, className="text-success" if pnl >= 0 else "text-danger")),
            html.Td(html.Span(strategy, className=f"{strategy_class}"))
        ]))

    buy_count = sum(1 for t in trades_sorted if t.get('action') == 'BUY')
    sell_count = sum(1 for t in trades_sorted if t.get('action') == 'SELL')
    total_pnl = sum(t.get('pnl', 0) for t in trades_sorted)

    summary = html.Div([
        html.Hr(),
        html.Div([
            html.Span("Всего сделок: ", className="text-muted"),
            html.Strong(f"{len(trades_sorted)}", className="ms-2"),
            html.Span("Покупки: ", className="text-muted ms-4"),
            html.Strong(f"{buy_count}", className="ms-2 text-success"),
            html.Span("Продажи: ", className="text-muted ms-4"),
            html.Strong(f"{sell_count}", className="ms-2 text-danger"),
            html.Span("Общий PnL: ", className="text-muted ms-4"),
            html.Strong(f"{total_pnl:+,.0f} ₽", className=f"ms-2 {'text-success' if total_pnl >= 0 else 'text-danger'}")
        ], className="d-flex justify-content-between flex-wrap")
    ])

    return html.Div([html.Table([header, html.Tbody(rows)], className="table table-sm table-hover table-striped"), summary], className="table-responsive")


def save_portfolio_history():
    """Сохранение текущего состояния портфеля в историю"""
    if broker_instance is None:
        return
    try:
        history_file = 'data/portfolio_history.json'
        history_data = []
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                if not isinstance(history_data, list):
                    history_data = []
        except:
            history_data = []
        summary = get_portfolio_summary_safe()
        history_data.append({
            'timestamp': datetime.now().isoformat(),
            'total_value': summary.get('total_value', 0),
            'cash': summary.get('cash', 0),
            'positions_value': summary.get('total_value', 0) - summary.get('cash', 0),
            'positions_count': summary.get('positions_count', 0),
            'pnl_total': summary.get('total_value', 0) - summary.get('initial_capital', 0)
        })
        history_data = history_data[-100:]
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, default=str)
        logger.debug(f"История портфеля сохранена ({len(history_data)} записей)")
    except Exception as e:
        logger.error(f"Ошибка сохранения истории портфеля: {e}")


@app.callback(
    Output("ticker-selector", "options"),
    [Input("interval-component", "n_intervals")],
    prevent_initial_call=False
)
def update_ticker_dropdown_simple(n_intervals):
    """Обновление выпадающего списка тикеров"""
    try:
        portfolio_tickers = get_portfolio_tickers()
        if not portfolio_tickers:
            return [{'label': "MOEX (Индекс Мосбиржи)", 'value': 'MOEX'}] + [
                {'label': ticker, 'value': ticker} for ticker in ['SBER', 'GAZP', 'LKOH', 'VTBR', 'ROSN', 'GMKN']
            ]
        options = []
        for ticker in sorted(portfolio_tickers):
            if ticker == 'MOEX':
                options.append({'label': "MOEX (Московская биржа)", 'value': ticker})
            else:
                options.append({'label': f"{ticker} (в портфеле)", 'value': ticker})
        return options
    except:
        return [{'label': 'SBER (Сбербанк)', 'value': 'SBER'}]


# Динамическое обновление списка компонентов в фильтре логов
@app.callback(
    Output("log-component-filter", "options"),
    Input("interval-component", "n_intervals")
)
def update_component_filter(n_intervals):
    """Обновление выпадающего списка компонентов на основе буфера логов"""
    from utils.logger import get_log_buffer

    log_buffer = get_log_buffer()
    names = set()
    for entry in log_buffer:
        name = entry.get('name', '')
        if name:
            names.add(name)

    options = [{'label': 'Все', 'value': 'ALL'}]
    options += [{'label': name, 'value': name} for name in sorted(names)]

    return options

@app.callback(
    Output("reload-configs-btn", "children"),
    Input("reload-configs-btn", "n_clicks")
)
def reload_configs(n_clicks):
    """Перезагрузка конфигов на лету"""
    if n_clicks and broker_instance is not None:
        try:
            broker_instance.reload_configs()
            logger.info("Конфиги перезагружены через дашборд")
            return "✅ Конфиги обновлены!"
        except Exception as e:
            logger.error(f"Ошибка перезагрузки конфигов: {e}")
            return "❌ Ошибка!"
    return "🔧 Обновить конфиги"


# ============================================================
# API для LLM-коуча
# ============================================================
@app.server.route("/api/ollama/models")
def get_ollama_models():
    """Возвращает список загруженных моделей Ollama"""
    try:
        import requests as req
        resp = req.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"models": models, "status": "ok"}
        return {"models": [], "status": "error", "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"models": [], "status": "error", "error": str(e)}


@app.server.route("/api/coach/model", methods=["POST"])
def set_coach_model():
    """Сохраняет выбранную модель коуча в rl_config.json"""
    try:
        import flask
        data = flask.request.json
        model_name = data.get("model", "")

        if not model_name:
            return {"status": "error", "error": "Не указана модель"}

        config_path = "config/rl_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "llm_coach" not in config:
            config["llm_coach"] = {}
        if "provider" not in config["llm_coach"]:
            config["llm_coach"]["provider"] = {}

        config["llm_coach"]["provider"]["type"] = "ollama"
        config["llm_coach"]["provider"]["model"] = model_name
        config["llm_coach"]["provider"]["url"] = config["llm_coach"]["provider"].get("url", "http://localhost:11434")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Модель коуча обновлена: {model_name}")
        return {"status": "ok", "model": model_name}

    except Exception as e:
        logger.error(f"Ошибка сохранения модели коуча: {e}")
        return {"status": "error", "error": str(e)}


@app.server.route("/api/coach/model", methods=["GET"])
def get_coach_model():
    """Возвращает текущую модель коуча из конфига"""
    try:
        config_path = "config/rl_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        coach_config = config.get("llm_coach", {})
        provider = coach_config.get("provider", {})
        return {
            "status": "ok",
            "model": provider.get("model", ""),
            "type": provider.get("type", "ollama"),
            "enabled": coach_config.get("enabled", False)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================
# Коллбэки LLM-коуча (один комбинированный коллбэк)
# ============================================================
@app.callback(
    [Output("coach-model-dropdown", "options"),
     Output("coach-model-dropdown", "value"),
     Output("coach-model-status", "children"),
     Output("coach-enabled-check", "value"),
     Output("coach-interval-input", "value"),
     Output("coach-timeout-input", "value"),
     Output("coach-weight-slider", "value")],
    [Input("refresh-models-btn", "n_clicks"),
     Input("interval-component", "n_intervals")],
    prevent_initial_call=False
)
def update_coach_ui(refresh_clicks, n_intervals):
    """Обновляет весь UI коуча: список моделей, статус, настройки"""
    # Получаем список моделей и статус
    try:
        import requests as req
        resp = req.get("http://localhost:11434/api/tags", timeout=5)
        models = []
        status = "✅ Ollama доступна"

        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if not models:
                status = "⚠️ Нет загруженных моделей"
        else:
            status = f"❌ Ошибка HTTP {resp.status_code}"
    except Exception as e:
        models = []
        status = f"❌ Ollama недоступна"

    # Получаем настройки из конфига
    try:
        with open("config/rl_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        coach = config.get("llm_coach", {})
        current_model = coach.get("provider", {}).get("model", "")
        enabled = ["enabled"] if coach.get("enabled", False) else []
        interval = coach.get("coach_interval_cycles", 20)
        timeout = coach.get("timeout_seconds", 30)
        weight = coach.get("coach_action_weight", 0.3)
    except:
        current_model = ""
        enabled = []
        interval = 20
        timeout = 30
        weight = 0.3

    # Всегда добавляем текущую модель в опции, даже если она не загружена
    options = [{"label": m, "value": m} for m in models]
    if current_model and current_model not in models:
        options.insert(0, {"label": f"{current_model} (не загружена)", "value": current_model})

    return options, current_model, status, enabled, interval, timeout, weight


@app.callback(
    Output("save-coach-settings-btn", "children"),
    [Input("save-coach-settings-btn", "n_clicks")],
    [State("coach-enabled-check", "value"),
     State("coach-model-dropdown", "value"),
     State("coach-interval-input", "value"),
     State("coach-timeout-input", "value"),
     State("coach-weight-slider", "value")]
)
def save_coach_settings(n_clicks, enabled, model, interval, timeout, weight):
    """Сохранение настроек коуча в rl_config.json"""
    if not n_clicks:
        return "💾 Сохранить настройки коуча"

    try:
        config_path = "config/rl_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "llm_coach" not in config:
            config["llm_coach"] = {}

        config["llm_coach"]["enabled"] = "enabled" in enabled
        config["llm_coach"]["coach_interval_cycles"] = interval
        config["llm_coach"]["timeout_seconds"] = timeout
        config["llm_coach"]["coach_action_weight"] = weight

        if "provider" not in config["llm_coach"]:
            config["llm_coach"]["provider"] = {}

        config["llm_coach"]["provider"]["type"] = "ollama"
        config["llm_coach"]["provider"]["model"] = model
        config["llm_coach"]["provider"]["timeout"] = timeout
        config["llm_coach"]["provider"]["url"] = config["llm_coach"]["provider"].get("url", "http://localhost:11434")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Настройки коуча сохранены: enabled={config['llm_coach']['enabled']}, model={model}")
        return "✅ Настройки коуча сохранены!"

    except Exception as e:
        logger.error(f"Ошибка сохранения настроек коуча: {e}")
        return "❌ Ошибка сохранения"


@app.callback(
    Output("coach-card-header", "children"),
    Input("coach-model-dropdown", "value")
)
def update_coach_header(model_name):
    """Обновляет заголовок карточки с названием модели"""
    if model_name:
        return f"🧠 LLM-Коуч — {model_name}"
    return "🧠 LLM-Коуч"

# ============================================================
# Коллбэки для вкладки "Работа моделей"
# ============================================================
@app.callback(
    [Output("coach-recommendations-table", "children"),
     Output("model-actions-table", "children")],
    Input("interval-component", "n_intervals")
)
def update_models_logs(n_intervals):
    """Обновляет таблицы рекомендаций коуча и действий модели"""
    coach_rows = []
    model_rows = []

    if broker_instance and hasattr(broker_instance, 'coach_log'):
        for entry in broker_instance.coach_log[-20:]:
            coach_rows.append(html.Tr([
                html.Td(entry.get('time', ''), style={'fontSize': '11px'}),
                html.Td(html.Strong(entry.get('ticker', '')), style={'fontSize': '11px'}),
                html.Td(html.Span(entry.get('action', ''), className=f"badge {'bg-danger' if entry.get('action') == 'SELL' else 'bg-success' if entry.get('action') == 'BUY' else 'bg-warning'}"), style={'fontSize': '11px'}),
                html.Td(entry.get('rule', ''), style={'fontSize': '11px'}),
                html.Td(f"{entry.get('confidence', 0):.2f}", style={'fontSize': '11px'}),
                html.Td(entry.get('rationale', ''), style={'fontSize': '11px', 'maxWidth': '300px', 'wordBreak': 'break-word'})
            ]))

    if broker_instance and hasattr(broker_instance, 'model_log'):
        for entry in broker_instance.model_log[-20:]:
            coach_advice = entry.get('coach_advice', '—')
            match = '✅' if entry.get('matched', False) else '❌'
            match_color = 'text-success' if entry.get('matched', False) else 'text-danger'
            action_str = str(entry.get('action', ''))
            # 🆕 v14: Цвета для новых типов действий
            if 'ENTRY_PASS' in action_str:
                action_class = 'badge bg-success'
            elif 'REJECT' in action_str:
                action_class = 'badge bg-danger'
            elif 'EXIT_HOLD' in action_str:
                action_class = 'badge bg-info'
            elif 'SELL' in action_str:
                action_class = 'badge bg-danger'
            elif 'BUY' in action_str:
                action_class = 'badge bg-success'
            else:
                action_class = 'badge bg-warning'
            model_rows.append(html.Tr([
                html.Td(entry.get('time', ''), style={'fontSize': '11px'}),
                html.Td(html.Strong(entry.get('ticker', '')), style={'fontSize': '11px'}),
                html.Td(html.Span(action_str, className=action_class), style={'fontSize': '11px'}),
                html.Td(f"{entry.get('state_value', 0):.2f}", style={'fontSize': '11px'}),
                html.Td(coach_advice, style={'fontSize': '11px', 'maxWidth': '300px', 'wordBreak': 'break-word'}),
                html.Td(html.Span(match, className=match_color), style={'fontSize': '11px', 'textAlign': 'center'})
            ]))

    coach_header = html.Thead(html.Tr([
        html.Th("Время"), html.Th("Тикер"), html.Th("Совет"), html.Th("Правило"),
        html.Th("Уверенность"), html.Th("Обоснование")
    ]))

    model_header = html.Thead(html.Tr([
        html.Th("Время"), html.Th("Тикер"), html.Th("Действие"), html.Th("Value/Conf"),
        html.Th("Причина / Параметры"), html.Th("OK?")
    ]))

    coach_table = html.Table([coach_header, html.Tbody(coach_rows)], className="table table-sm table-hover table-striped") if coach_rows else html.P("Нет рекомендаций коуча", className="text-muted")
    model_table = html.Table([model_header, html.Tbody(model_rows)], className="table table-sm table-hover table-striped") if model_rows else html.P("Нет действий модели", className="text-muted")

    return coach_table, model_table


@app.callback(
    Output("model-actions-header", "children"),
    Input("model-actions-table", "children")
)
def update_model_actions_header(table_children):
    """Обновляет заголовок с процентом совпадений"""
    if not table_children or not hasattr(table_children, 'props'):
        return "🤖 Действия модели"

    # Подсчитываем совпадения из логов
    if broker_instance and hasattr(broker_instance, 'model_log') and broker_instance.model_log:
        logs = broker_instance.model_log
        total = len(logs)
        matched = sum(1 for entry in logs if entry.get('matched', False))
        if total > 0:
            pct = (matched / total) * 100
            return f"🤖 Действия модели — совпадений: {matched}/{total} ({pct:.1f}%)"

    return "🤖 Действия модели"

# ============================================================
# 📰 Коллбэки вкладки "Новости и сентимент"
# ============================================================

@app.callback(
    Output("news-ticker-filter", "options"),
    Input("interval-component", "n_intervals"),
    prevent_initial_call=False
)
def update_news_ticker_filter_options(n_intervals):
    """Обновление выпадающего списка тикеров на основе ленты новостей."""
    if broker_instance is None:
        return [{'label': 'Все тикеры', 'value': 'ALL'}]
    try:
        # Получаем ленту, чтобы собрать все тикеры
        feed = broker_instance.get_news_sentiment_feed(limit=100, min_abs_sentiment=0.0)
        tickers_seen = set()
        for item in feed:
            for t in item.get('tickers', []):
                tickers_seen.add(t)

        options = [{'label': 'Все тикеры', 'value': 'ALL'}]
        options += [{'label': t, 'value': t} for t in sorted(tickers_seen)]
        return options
    except Exception as e:
        logger.error(f"Ошибка обновления фильтра тикеров новостей: {e}")
        return [{'label': 'Все тикеры', 'value': 'ALL'}]


@app.callback(
    [Output("news-sentiment-stats", "children"),
     Output("news-top-positive", "children"),
     Output("news-top-negative", "children")],
    Input("interval-component", "n_intervals"),
    prevent_initial_call=False
)
def update_news_sentiment_overview(n_intervals):
    """Обновление сводной статистики и топов позитивных/негативных новостей."""
    if broker_instance is None:
        return [
            html.P("Брокер не инициализирован", className="text-muted"),
            html.P("Нет данных", className="text-muted"),
            html.P("Нет данных", className="text-muted"),
        ]

    try:
        stats = broker_instance.get_news_sentiment_stats()

        # —— Сводка ——
        total = stats.get('total', 0)
        pos = stats.get('positive', 0)
        neg = stats.get('negative', 0)
        neu = stats.get('neutral', 0)
        avg = stats.get('avg_sentiment', 0.0)

        avg_class = "text-success" if avg > 0.05 else "text-danger" if avg < -0.05 else "text-muted"

        stats_html = html.Div([
            html.Div([
                html.H5(f"{total}", className="mb-0"),
                html.Small("Всего новостей", className="text-muted")
            ], className="text-center p-2"),
            html.Div([
                html.H5(f"🟢 {pos}", className="mb-0 text-success"),
                html.Small("Позитивных", className="text-muted")
            ], className="text-center p-2"),
            html.Div([
                html.H5(f"🔴 {neg}", className="mb-0 text-danger"),
                html.Small("Негативных", className="text-muted")
            ], className="text-center p-2"),
            html.Div([
                html.H5(f"⚪ {neu}", className="mb-0 text-muted"),
                html.Small("Нейтральных", className="text-muted")
            ], className="text-center p-2"),
            html.Div([
                html.H5(html.Span(f"{avg:+.3f}", className=avg_class), className="mb-0"),
                html.Small("Средний сентимент", className="text-muted")
            ], className="text-center p-2"),
        ], className="d-flex flex-wrap gap-3 justify-content-around")

        # —— Топ-3 позитивных ——
        top_pos = stats.get('top_positive', [])
        if top_pos:
            pos_html = html.Div([
                _render_news_card(n, positive=True) for n in top_pos
            ])
        else:
            pos_html = html.P("Нет позитивных новостей", className="text-muted")

        # —— Топ-3 негативных ——
        top_neg = stats.get('top_negative', [])
        if top_neg:
            neg_html = html.Div([
                _render_news_card(n, positive=False) for n in top_neg
            ])
        else:
            neg_html = html.P("Нет негативных новостей", className="text-muted")

        return [stats_html, pos_html, neg_html]

    except Exception as e:
        logger.error(f"Ошибка обновления обзорной статистики новостей: {e}")
        return [
            html.P(f"Ошибка: {e}", className="text-danger"),
            html.P("Ошибка", className="text-danger"),
            html.P("Ошибка", className="text-danger"),
        ]


def _render_news_card(news: Dict, positive: bool = True) -> html.Div:
    """Рендер одной новости в виде компактной карточки."""
    sentiment = news.get('sentiment', 0.0)
    title = news.get('title', '(без заголовка)')
    source = news.get('source', '—')
    tickers = news.get('tickers', [])
    timestamp = news.get('timestamp', '')

    # Парсим время
    time_str = ''
    if timestamp:
        try:
            dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
            time_str = dt.strftime('%d.%m %H:%M')
        except Exception:
            time_str = str(timestamp)[:16]

    border_color = '#28a745' if positive else '#dc3545'
    sent_color = 'text-success' if positive else 'text-danger'

    ticker_badges = html.Div([
        html.Span(t, className="badge bg-info me-1") for t in tickers[:3]
    ], className="mt-1") if tickers else None

    return html.Div([
        html.Div([
            html.Span(f"{sentiment:+.3f}", className=f"{sent_color} fw-bold me-2"),
            html.Small(f"{time_str} · {source}", className="text-muted")
        ], className="d-flex justify-content-between align-items-center"),
        html.P(title, className="mb-1 small"),
        ticker_badges
    ], className="mb-2 p-2", style={
        'borderLeft': f'3px solid {border_color}',
        'backgroundColor': 'rgba(255,255,255,0.03)',
        'borderRadius': '4px'
    })


@app.callback(
    [Output("news-feed-table", "children"),
     Output("news-count-badge", "children")],
    [Input("interval-component", "n_intervals"),
     Input("news-ticker-filter", "value"),
     Input("news-label-filter", "value"),
     Input("news-min-sentiment-slider", "value")],
    prevent_initial_call=False
)
def update_news_feed_table(n_intervals, ticker_filter, label_filter, min_sentiment):
    """Обновление таблицы ленты новостей с сентиментом."""
    if broker_instance is None:
        return [html.P("Брокер не инициализирован", className="text-muted p-3"), "0"]

    try:
        feed = broker_instance.get_news_sentiment_feed(
            limit=100,
            min_abs_sentiment=float(min_sentiment or 0.0),
            ticker_filter=ticker_filter,
            label_filter=label_filter or 'ALL'
        )

        if not feed:
            return [
                html.Div([
                    html.P("Нет новостей по выбранным фильтрам", className="text-muted text-center p-4"),
                    html.Small("Попробуйте уменьшить мин. |sentiment| или изменить фильтр",
                               className="text-muted d-block text-center")
                ], className="p-3"),
                "0"
            ]

        header = html.Thead(html.Tr([
            html.Th("Время"),
            html.Th("Источник"),
            html.Th("Тикеры"),
            html.Th("Заголовок"),
            html.Th("Сентимент"),
            html.Th("Лейбл"),
            html.Th("Коррекция"),  # 🆕 v16.4
        ]))

        rows = []
        for news in feed:
            sentiment = news.get('sentiment', 0.0)
            label = news.get('sentiment_label', 'NEUTRAL')
            override = news.get('sentiment_override', '')

            # Цвет сентимента
            if label == 'POSITIVE':
                sent_class = "text-success fw-bold"
                label_badge = html.Span("🟢 POS", className="badge bg-success")
            elif label == 'NEGATIVE':
                sent_class = "text-danger fw-bold"
                label_badge = html.Span("🔴 NEG", className="badge bg-danger")
            else:
                sent_class = "text-muted"
                label_badge = html.Span("⚪ NEU", className="badge bg-secondary")

            # 🆕 Бейдж context-override
            override_badge = None
            if override:
                override_badge = html.Span(
                    "⚡ override",
                    className="badge bg-warning ms-1",
                    title=override,
                    style={'fontSize': '9px'}
                )

            # Время
            timestamp = news.get('timestamp', '')
            time_str = ''
            if timestamp:
                try:
                    dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                    time_str = dt.strftime('%d.%m %H:%M')
                except Exception:
                    time_str = str(timestamp)[:16]

            # Тикеры
            tickers = news.get('tickers', [])
            if tickers:
                tickers_html = html.Div([
                    html.Span(t, className="badge bg-info me-1 mb-1") for t in tickers[:4]
                ])
            else:
                tickers_html = html.Span("—", className="text-muted")

            # Заголовок (с tooltip)
            title = news.get('title', '(без заголовка)')
            summary = news.get('summary', '')
            link = news.get('link', '')

            # Тримим заголовок до 120 символов
            title_display = title[:120] + ('…' if len(title) > 120 else '')

            if link:
                title_html = html.A(title_display, href=link, target="_blank",
                                     className="text-decoration-none",
                                     style={'color': '#00ccff'})
            else:
                title_html = html.Span(title_display)

            # Tooltip с summary
            if summary:
                tooltip_text = summary[:300] + ('…' if len(summary) > 300 else '')
                title_html = html.Div([
                    title_html,
                    html.Small(tooltip_text, className="text-muted d-block",
                               style={'fontSize': '10px', 'maxHeight': '40px',
                                      'overflow': 'hidden', 'marginTop': '2px'})
                ])

            # 🆕 v16.4: Кнопки коррекции сентимента
            # Используем data attributes для передачи данных новости
            news_data_json = json.dumps({
                'title': title[:200],
                'summary': summary[:300] if summary else '',
                'source': news.get('source', ''),
                'original_label': label,
                'original_sentiment': sentiment,
            }, ensure_ascii=False)

            correction_buttons = html.Div([
                html.Button("🟢", id={'type': 'sent-correct', 'index': hash(title) % 100000},
                           title="Позитивный",
                           className="btn btn-sm btn-outline-success p-1",
                           style={'fontSize': '10px', 'padding': '2px 4px'},
                           **{'data-label': 'POSITIVE', 'data-news': news_data_json}),
                html.Button("🔴", id={'type': 'sent-correct', 'index': hash(title) % 100000 + 1},
                           title="Негативный",
                           className="btn btn-sm btn-outline-danger p-1 ms-1",
                           style={'fontSize': '10px', 'padding': '2px 4px'},
                           **{'data-label': 'NEGATIVE', 'data-news': news_data_json}),
                html.Button("⚪", id={'type': 'sent-correct', 'index': hash(title) % 100000 + 2},
                           title="Нейтральный",
                           className="btn btn-sm btn-outline-secondary p-1 ms-1",
                           style={'fontSize': '10px', 'padding': '2px 4px'},
                           **{'data-label': 'NEUTRAL', 'data-news': news_data_json}),
            ], className="d-flex")

            rows.append(html.Tr([
                html.Td(html.Small(time_str), className="text-muted"),
                html.Td(html.Small(news.get('source', '—'))),
                html.Td(tickers_html),
                html.Td(title_html, style={'maxWidth': '400px'}),
                html.Td(html.Span(f"{sentiment:+.3f}", className=sent_class)),
                html.Td(html.Div([label_badge, override_badge] if override_badge else [label_badge])),
                html.Td(correction_buttons),  # 🆕 v16.4
            ]))

        table = html.Table([header, html.Tbody(rows)],
                           className="table table-sm table-hover table-striped")

        count_badge = f"{len(feed)} новост."

        return [table, count_badge]

    except Exception as e:
        logger.error(f"Ошибка обновления таблицы новостей: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return [html.P(f"Ошибка: {e}", className="text-danger p-3"), "0"]


# ============================================================
# 🆕 v16.4: API для дообучения новостной модели сентимента
# ============================================================

@app.server.route("/api/sentiment/correct", methods=["POST"])
def sentiment_correct():
    """Сохранение коррекции сентимента пользователем."""
    try:
        import flask
        from core.sentiment_finetuner import save_correction, get_corrections_count

        data = flask.request.json
        title = data.get("title", "")
        summary = data.get("summary", "")
        source = data.get("source", "")
        original_label = data.get("original_label", "NEUTRAL")
        original_sentiment = data.get("original_sentiment", 0.0)
        corrected_label = data.get("corrected_label", "NEUTRAL")

        if not title:
            return {"status": "error", "error": "title is required"}, 400

        success = save_correction(
            news_title=title,
            news_summary=summary,
            source=source,
            original_label=original_label,
            original_sentiment=original_sentiment,
            corrected_label=corrected_label,
        )

        return {
            "status": "ok" if success else "error",
            "total_corrections": get_corrections_count(),
        }
    except Exception as e:
        logger.error(f"API sentiment/correct error: {e}")
        return {"status": "error", "error": str(e)}, 500


@app.server.route("/api/sentiment/corrections", methods=["GET"])
def sentiment_corrections_list():
    """Получение списка всех коррекций."""
    try:
        from core.sentiment_finetuner import get_corrections, get_stats
        corrections = get_corrections()
        stats = get_stats()
        return {"corrections": corrections, "stats": stats}
    except Exception as e:
        logger.error(f"API sentiment/corrections error: {e}")
        return {"error": str(e)}, 500


@app.server.route("/api/sentiment/finetune", methods=["POST"])
def sentiment_finetune():
    """Запуск дообучения модели."""
    try:
        import flask
        from core.sentiment_finetuner import finetune_model, get_corrections_count

        data = flask.request.json or {}
        epochs = data.get("epochs", 3)
        batch_size = data.get("batch_size", 8)
        learning_rate = data.get("learning_rate", 2e-5)

        result = finetune_model(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        return result
    except Exception as e:
        logger.error(f"API sentiment/finetune error: {e}")
        return {"success": False, "message": f"Ошибка: {e}"}, 500


@app.server.route("/api/sentiment/reload-analyzer", methods=["POST"])
def sentiment_reload_analyzer():
    """Перезапуск NewsAnalyzer без перезапуска системы."""
    try:
        if broker_instance is None:
            return {"success": False, "message": "Брокер не инициализирован"}, 500
        result = broker_instance.reload_news_analyzer()
        return result
    except Exception as e:
        logger.error(f"API sentiment/reload-analyzer error: {e}")
        return {"success": False, "message": f"Ошибка: {e}"}, 500


@app.server.route("/api/sentiment/stats", methods=["GET"])
def sentiment_stats():
    """Статистика по коррекциям и модели."""
    try:
        from core.sentiment_finetuner import get_stats
        return get_stats()
    except Exception as e:
        return {"error": str(e)}, 500


# ============================================================
# 🆕 v16.4: Callback для UI дообучения в вкладке Новости
# ============================================================

@app.callback(
    [Output("sentiment-ft-stats", "children"),
     Output("sentiment-ft-button", "disabled"),
     Output("sentiment-ft-button", "children"),
     Output("sentiment-ft-status", "children"),
     Output("sentiment-ft-result", "children"),
     Output("sentiment-corrections-list", "children")],
    [Input("interval-component", "n_intervals"),
     Input("sentiment-ft-button", "n_clicks"),
     Input("refresh-corrections-btn", "n_clicks")],
    prevent_initial_call=False
)
def update_sentiment_finetune_ui(n_intervals, ft_clicks, refresh_clicks):
    """Обновление UI блока дообучения."""
    from core.sentiment_finetuner import get_stats, get_corrections, finetune_model

    triggered = ctx.triggered_id

    # Обработка нажатия "Дообучить"
    ft_result = ""
    if triggered == "sentiment-ft-button" and ft_clicks:
        result = finetune_model()
        if result['success']:
            ft_result = html.P(f"✅ {result['message']}", className="text-success small")
        else:
            ft_result = html.P(f"❌ {result['message']}", className="text-danger small")

    stats = get_stats()
    total = stats['total_corrections']
    min_needed = stats['min_for_finetune']
    finetuned = stats['finetuned_available']

    # Статистика
    by_label = stats['by_label']
    stats_html = html.Div([
        html.P([
            html.Span(f"Коррекций: {total}/{min_needed} мин. ", className="text-info"),
            html.Span(f"| 🟢 {by_label.get('POSITIVE', 0)} ", className="text-success"),
            html.Span(f"🔴 {by_label.get('NEGATIVE', 0)} ", className="text-danger"),
            html.Span(f"⚪ {by_label.get('NEUTRAL', 0)}", className="text-muted"),
        ]),
        html.P(f"Дообученная модель: {'✅ доступна' if finetuned else '❌ нет'}",
               className="small text-muted"),
    ])

    # Кнопка
    button_disabled = total < min_needed
    button_text = f"🎓 Дообучить модель ({total} коррекций)"

    # Статус
    status_text = ""
    if button_disabled:
        status_text = f"Соберите ещё {min_needed - total} коррекций для дообучения"
    elif finetuned:
        status_text = "Дообученная модель активна. Перезапустите систему после нового дообучения."

    # Список коррекций
    corrections = get_corrections()
    if corrections:
        corr_items = []
        for c in corrections[-20:]:  # последние 20
            corr_items.append(html.Div([
                html.Span(f"🟢" if c.get('corrected_label') == 'POSITIVE' else
                          f"🔴" if c.get('corrected_label') == 'NEGATIVE' else "⚪",
                          className="me-1"),
                html.Span(c.get('title', '')[:60] + ('…' if len(c.get('title', '')) > 60 else ''),
                          className="text-muted"),
                html.Small(f" (было: {c.get('original_label', '?')})",
                          className="text-muted ms-1"),
            ], className="mb-1"))
        corr_list = html.Div(corr_items)
    else:
        corr_list = html.P("Нет коррекций", className="text-muted small")

    return [stats_html, button_disabled, button_text, status_text, ft_result, corr_list]


# 🆕 v16.7: Callback для перезапуска NewsAnalyzer
@app.callback(
    Output("reload-analyzer-status", "children"),
    Input("reload-news-analyzer-btn", "n_clicks"),
    prevent_initial_call=True
)
def reload_news_analyzer(n_clicks):
    """Перезапуск NewsAnalyzer через API."""
    if not n_clicks:
        return ""

    try:
        import requests
        # Вызываем API напрямую (broker_instance может быть в другом потоке)
        resp = requests.post(
            f"http://localhost:{broker_instance.settings.get('web_port', 8050) if broker_instance else 8050}/api/sentiment/reload-analyzer",
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get('success'):
                return html.P(f"✅ {data['message']}", className="text-success small")
            else:
                return html.P(f"❌ {data.get('message', 'ошибка')}", className="text-danger small")
        else:
            return html.P(f"❌ HTTP {resp.status_code}", className="text-danger small")
    except Exception as e:
        # Прямой вызов через broker_instance (fallback)
        if broker_instance:
            result = broker_instance.reload_news_analyzer()
            if result.get('success'):
                return html.P(f"✅ {result['message']}", className="text-success small")
            else:
                return html.P(f"❌ {result.get('message', 'ошибка')}", className="text-danger small")
        return html.P(f"❌ Ошибка: {e}", className="text-danger small")


# Клиентский скрипт для коррекции сентимента
@app.server.route("/api/sentiment/correct", methods=["GET"])
def sentiment_correct_help():
    """Справка по API."""
    return {
        "usage": "POST /api/sentiment/correct",
        "fields": {
            "title": "str (required)",
            "summary": "str",
            "source": "str",
            "original_label": "POSITIVE|NEGATIVE|NEUTRAL",
            "original_sentiment": "float",
            "corrected_label": "POSITIVE|NEGATIVE|NEUTRAL",
        }
    }


if __name__ == '__main__':
    print("Запуск веб-интерфейса...")
    app.run(debug=True, port=8050, host='127.0.0.1')